import json

def simple_merge_transforms(pre_change_file, post_change_file, output_file, post_change_indices):
    # Read pre-change transforms
    with open(pre_change_file, 'r') as f:
        pre_data = json.load(f)
    
    # Read post-change transforms
    with open(post_change_file, 'r') as f:
        post_data = json.load(f)
    
    # Select specific post-change frames
    selected_post_frames = [post_data['frames'][i] for i in post_change_indices]
    
    # Make sure file paths are correct
    for frame in selected_post_frames:
        if 'file_path' in frame:
            # Ensure the path includes 'rgb_new/' prefix
            if not frame['file_path'].startswith('rgb_new/'):
                frame['file_path'] = 'rgb_new/' + frame['file_path'].split('/')[-1]
    
    # Combine pre-change and selected post-change frames
    merged_data = pre_data.copy()
    merged_data['frames'].extend(selected_post_frames)
    
    # Write the merged transforms
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=2)
    
    print(f"Merged {len(pre_data['frames'])} pre-change frames with {len(selected_post_frames)} post-change frames.")
    print(f"Output saved to {output_file}")

# Example usage
simple_merge_transforms(
    '/local/home/pmishra/isaac-nav-suite/scripts/dlab/render_data/transforms.json',
    '/local/home/pmishra/isaac-nav-suite/scripts/dlab/new/render_data_open/transforms.json',
    '/local/home/pmishra/cvg/3dgscd/data/dlab/transforms.json',
    [0, 2, 5, 6]
)