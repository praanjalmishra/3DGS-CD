# Use HLoc for camrea relocalization

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pycolmap
import pyquaternion
import torch
from hloc import (extract_features, localize_sfm, match_features,
                  matchers, pairs_from_retrieval)
from hloc.utils.base_model import dynamic_load
from hloc.utils.io import get_keypoints
from tqdm import tqdm

retrieval_conf = extract_features.confs["netvlad"]
feature_conf = extract_features.confs["sift"]
matcher_conf = match_features.confs["adalam"]
pycolmap_conf = {
    "estimation": {"ransac": {"max_error": 2.0}}, 
    "refinement": {'refine_focal_length': False}
}


def colmap_to_opengl(qvec, tvec):
    """
    Convert colmap pose to opengl pose

    Args:
        qvec (4-list): quaternion (qw, qx, qy, qz)
        tvec (3-list): translation (x, y, z)

    Returns:
        pose (4x4 array): opengl pose
    """
    pose = pyquaternion.Quaternion(qvec).transformation_matrix
    pose[:3, 3] = tvec
    pose = np.linalg.inv(pose)
    # Convert from COLMAP's camera coordinate system (OpenCV) to ours (OpenGL)
    pose[0:3, 1:3] *= -1
    pose = pose[np.array([1, 0, 2, 3]), :]
    pose[2, :] *= -1
    return pose


def image_retrieval(descriptors, num_matched, db_model, db_descriptors):
    """
    Global image retrieval

    Args:
        descriptors (Path): path to global descriptors
        num_matched (int): number of retrieved images
        db_model (Path): path to the database model
        db_descriptors (Path): path to the database descriptors
    """
    if isinstance(db_descriptors, (Path, str)):
        db_descriptors = [db_descriptors]
    name2db = {n: i for i, p in enumerate(db_descriptors)
               for n in pairs_from_retrieval.list_h5_names(p)}
    query_names_h5 = pairs_from_retrieval.list_h5_names(descriptors)

    images = pairs_from_retrieval.read_images_binary(db_model / 'images.bin')
    db_names = [i.name for i in images.values()]

    if len(db_names) == 0:
        raise ValueError('Could not find any database image.')
    query_names = pairs_from_retrieval.parse_names(None, None, query_names_h5)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    db_desc = pairs_from_retrieval.get_descriptors(
        db_names, db_descriptors, name2db
    )
    query_desc = pairs_from_retrieval.get_descriptors(query_names, descriptors)
    sim = torch.einsum('id,jd->ij', query_desc.to(device), db_desc.to(device))

    # Avoid self-matching
    self = np.array(query_names)[:, None] == np.array(db_names)[None]
    pairs = pairs_from_retrieval.pairs_from_score_matrix(
        sim, self, num_matched, min_score=0
    )
    pairs = [(query_names[i], db_names[j]) for i, j in pairs]
    pairs_dict = {}
    for k, v in pairs:
        if k not in pairs_dict:
            pairs_dict[k] = [v]
        else:
            pairs_dict[k].append(v)
    return pairs_dict


def feature_matching(conf, pairs, features, features_ref):
    """
    Match features between query and reference images

    Args:
        conf (dict): matcher configs
        pairs (dict): image pairs
        features (Path): path to features
        features_ref (Path): path to reference features

    Returns:
        matches (list): matches
        scores (list): matching scores
    """
    assert isinstance(features, Path) or Path(features).exists()
    assert features_ref is not None and features_ref.exists()
    assert features.exists()

    pairs = [(q, r) for q, rs in pairs.items() for r in rs]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Model = dynamic_load(matchers, conf['model']['name'])
    model = Model(conf['model']).eval().to(device)

    dataset = match_features.FeaturePairsDataset(
        pairs, features, features_ref
    )
    loader = torch.utils.data.DataLoader(
        dataset, num_workers=5, batch_size=1, shuffle=False, pin_memory=True
    )

    all_matches, all_scores = defaultdict(dict), defaultdict(dict)
    for ii, data in enumerate(tqdm(loader, smoothing=.1)):
        data = {k: v if k.startswith('image')
                else v.to(device, non_blocking=True) for k, v in data.items()}
        pred = model(data)
        matches = pred['matches0'][0].cpu().short().numpy()
        scores = pred['matching_scores0'][0].cpu().half().numpy()
        idx = np.where(matches != -1)[0]
        matches = np.stack([idx, matches[idx]], -1)
        scores = scores[idx]
        all_matches[pairs[ii][0]][pairs[ii][1]] = matches
        all_scores[pairs[ii][0]][pairs[ii][1]] = scores
    return all_matches, all_scores


def pose_from_cluster(
    localizer, qname, query_camera, db_ids, features_path, match_dict,
    **kwargs
):
    """
    Localize query image from a cluster of database images

    Args:
        localizer (QueryLocalizer): query localizer
        qname (str): query image name
        query_camera (pycolmap.Camera): query camera
        db_ids (list): database image ids
        features_path (Path): path to features
        match_dict (Nx2-array): matches
        kwargs (dict): keyword arguments

    Returns:
        ret (dict): localization results
    """

    kpq = get_keypoints(features_path, qname)
    kpq += 0.5  # COLMAP coordinates

    kp_idx_to_3D = defaultdict(list)
    kp_idx_to_3D_to_db = defaultdict(lambda: defaultdict(list))
    num_matches = 0
    for i, db_id in enumerate(db_ids):
        image = localizer.reconstruction.images[db_id]
        if image.num_points3D() == 0:
            print(f'No 3D points found for {image.name}.')
            continue
        points3D_ids = np.array([p.point3D_id if p.has_point3D() else -1
                                 for p in image.points2D])

        matches = match_dict[image.name]
        matches = matches[points3D_ids[matches[:, 1]] != -1]
        num_matches += len(matches)
        for idx, m in matches:
            id_3D = points3D_ids[m]
            kp_idx_to_3D_to_db[idx][id_3D].append(i)
            # avoid duplicate observations
            if id_3D not in kp_idx_to_3D[idx]:
                kp_idx_to_3D[idx].append(id_3D)

    idxs = list(kp_idx_to_3D.keys())
    mkp_idxs = [i for i in idxs for _ in kp_idx_to_3D[i]]
    mp3d_ids = [j for i in idxs for j in kp_idx_to_3D[i]]
    ret = localizer.localize(kpq, mkp_idxs, mp3d_ids, query_camera, **kwargs)
    ret['camera'] = {
        'model': query_camera.model_name,
        'width': query_camera.width,
        'height': query_camera.height,
        'params': query_camera.params,
    }
    return ret


def localize(
    reference_sfm, queries, retrieval_dict, features, match_dict,
    ransac_thresh=12, width=1024, height=1024, intrin=[1422, 1422, 512, 512],
    dist_params=[0.0, 0.0, 0.0, 0.0]
):
    """
    Camera localization for sparse view images w/ SolvePnP + RANSAC
    
    Args:
        reference_sfm (pycolmap.Reconstruction): reference reconstruction
        queries (list): list of query image names
        retrieval_dict (dict): image pairs {"qimg":["dbimg1", "dbimg2"...]...}
        features (Path): path to features
        match_dict (Nx2-array-dict-dict): matches of query-reference image pairs
        ransac_thresh (float): RANSAC threshold
        width (int): image width
        height (int): image height
        intrin (4-list): camera intrinsics (fx, fy, cx, cy)
        dist_params (4-list): camera distortion params (k1, k2, p1, p2)

    Returns:
        poses (list): camera poses (4x4 arrays)
    """
    assert features.exists(), features

    cam = pycolmap.Camera(
        "OPENCV", int(width), int(height), params =[*intrin, *dist_params]
    )
    queries = [(q, cam) for q in queries]

    print('Reading the 3D model...')
    if not isinstance(reference_sfm, pycolmap.Reconstruction):
        reference_sfm = pycolmap.Reconstruction(reference_sfm)
    db_name_to_id = {img.name: i for i, img in reference_sfm.images.items()}

    config = {
        "estimation": {"ransac": {"max_error": ransac_thresh}},
        "refinement": {'refine_focal_length': False}
    }
    localizer = localize_sfm.QueryLocalizer(reference_sfm, config)

    poses = {}
    print('Starting localization...')
    for idx, (qname, qcam) in enumerate(queries):
        if qname not in retrieval_dict:
            print(f'No images retrieved for query image {qname}. Skipping...')
            continue
        db_names = retrieval_dict[qname]
        db_ids = []
        for n in db_names:
            if n not in db_name_to_id:
                print(f'Image {n} was retrieved but not in database')
                continue
            db_ids.append(db_name_to_id[n])
        ret = pose_from_cluster(
            localizer, qname, qcam, db_ids, features, match_dict[qname]
        )
        if ret['success']:
            poses[qname] = (ret['qvec'], ret['tvec'])
        else:
            closest = reference_sfm.images[db_ids[0]]
            poses[qname] = (closest.qvec, closest.tvec)

    print(f'Localized {len(poses)} / {len(queries)} images.')
    res = {}
    for q in poses:
        qvec, tvec = poses[q]
        pose = colmap_to_opengl(qvec, tvec)
        name = q.split("/")[-1]
        res[name] = pose
    return res


def read_cam_params_from_json(transforms_json):
    """
    Read Camera parameters from transforms.json

    Args:
        transforms_json (Path or str): path to transforms.json

    Returns:
        width, height (int): Image dimensions
        fx, fy, cx, cy: Camera intrinsic params
        k1, k2, p1, p2: Camera distortion params
    """
    transforms_json = Path(transforms_json)
    with open(transforms_json, 'r') as infile:
        data = json.load(infile)
    return (
        data["w"], data["h"],
        data["fl_x"], data["fl_y"], data["cx"], data["cy"],
        data["k1"], data["k2"], data["p1"], data["p2"]
    )


def save_poses(
    poses, json_path, width, height, intrin, dist_params, prefix="rgb_new"
):
    """
    Save estimated camera poses to transforms.json
    
    Args:
        poses (dict): camera poses
        json_path (Path): path to save json
        width (int): image width
        height (int): image height
        intrin (4-list): camera intrinsics (fx, fy, cx, cy)
        dist_params (4-list): camera distortion params (k1, k2, p1, p2)
        prefix (str): prefix for image file paths
    """
    json_dict = {
        "fl_x": intrin[0], "fl_y": intrin[1], "cx": intrin[2], "cy": intrin[3],
        "w": width, "h": height, "camera_model": "OPENCV", 
        "k1": dist_params[0], "k2": dist_params[1],
        "p1": dist_params[2], "p2": dist_params[3], "frames": []
    }
    for file_path, pose in poses.items():
        frame_dict = {
            "file_path": f"{prefix}/{file_path}",
            "transform_matrix": pose.tolist()
        }
        json_dict["frames"].append(frame_dict)
    
    with open(json_path, "w") as f:
        json.dump(json_dict, f, indent=4)


def camloc(
    ref_sfm_dir, query_img_dir, 
    transforms_filename="transforms.json", ransac_thresh=12
):
    """
    Localize the sparse view cameras
    
    Args:
        ref_sfm_dir (Path): Path to colmap SFM result
        query_img_dir (Path): Path to sparse view RGB images
        output_dir (Path): Path to save feature extraction results
        ransac_thresh (float): RANSAC threshold in pixels

    Returns:
        poses (list): camera poses (4x4 arrays)
    """
    ref_sfm_dir = Path(ref_sfm_dir)
    query_img_dir = Path(query_img_dir)
    transforms_json = ref_sfm_dir.parent / transforms_filename
    # --- Read camera parameters ---
    width, height, fx, fy, cx, cy, k1, k2, p1, p2 = \
        read_cam_params_from_json(transforms_json)
    intrin = [fx, fy, cx, cy]
    dist_params = [k1, k2, p1, p2]
    # --- Read the reference 3D reconstruction ---
    ref_sfm = pycolmap.Reconstruction(ref_sfm_dir / "sparse" / "0")
    # --- Extract local and global features from query images --- 
    for f in query_img_dir.iterdir():  # Delete existing feature files
        if f.suffix == ".h5":
            f.unlink()
    query_imgs = [
        p.relative_to(query_img_dir).as_posix()
        for p in query_img_dir.iterdir() if p.suffix == ".png"
    ]
    extract_features.main(
        feature_conf, query_img_dir,
        image_list=query_imgs, feature_path=query_img_dir / "features.h5"
    )
    global_features = extract_features.main(
        retrieval_conf, query_img_dir, query_img_dir
    )
    # --- Global localization to retrieve reference images ---
    loc_pairs = image_retrieval(
        global_features, num_matched=10,
        db_model=ref_sfm_dir / "sparse" / "0",
        db_descriptors=ref_sfm_dir / "global-feats-netvlad.h5"
    )
    # --- Feature matching ---
    loc_matches, _ = feature_matching(
        matcher_conf, loc_pairs, query_img_dir / "features.h5",
        features_ref=ref_sfm_dir / "features.h5"
    )
    # --- Camera Localization ---
    poses = localize(
        ref_sfm, query_imgs, loc_pairs, query_img_dir / "features.h5",
        loc_matches, ransac_thresh=ransac_thresh,
        width=width, height=height, intrin=intrin
    )
    for f in query_img_dir.iterdir():  # Delete extracted feature files
        if f.suffix == ".h5":
            f.unlink()
    # Save poses to transforms.json
    save_poses(
        poses, query_img_dir / "transforms.json",
        width, height, intrin, dist_params
    )
    return poses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--image_dir', type=Path,
        default=Path("/home/ziqi/Desktop/test/test/")
    )
    parser.add_argument(
        '-sfm', '--colmap_path', type=str,
        default=Path("/home/ziqi/Desktop/test/masks/colmap/")
    )
    parser.add_argument(
        '-t', '--transforms_json', type=str, default="transforms.json",
    )
    parser.add_argument(
        '-rt', '--ransac_thresh', type=float, default=12.0
    )
    args = parser.parse_args()

    res = camloc(
        args.colmap_path, args.image_dir,
        transforms_filename=args.transforms_json, ransac_thresh=args.ransac_thresh
    )
