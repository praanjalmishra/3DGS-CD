---
name: Bug report
about: Create a report to help us improve
title: ''
labels: ''
assignees: ''

---

**Please add debug information**

1. Add `--debug` to the end of [this line](https://github.com/520xyxyzq/3DGS-CD/blob/4aca35166c9a20a2b79ab79f74d5b6d6f3ca1722/scripts/real_gsplat_train.sh#L173)
2. Comment out [these lines](https://github.com/520xyxyzq/3DGS-CD/blob/main/scripts/real_gsplat_train.sh#L36-L167).
3. Replace `${current_time}` on [this line](https://github.com/520xyxyzq/3DGS-CD/blob/4aca35166c9a20a2b79ab79f74d5b6d6f3ca1722/scripts/real_gsplat_train.sh#L171) with the actual folder name.
4. Replace the `debug_dir` on [this line](https://github.com/520xyxyzq/3DGS-CD/blob/853b8621ce41715e366b456bebe28b34a8ad0340/nerfstudio/scripts/change_det.py#L76) with your favorite folder.
5. Run the script again
6. Share the debug images saved in your `debug_dir`.

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected behavior**
A clear and concise description of what you expected to happen.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Additional context**
Add any other context about the problem here.
