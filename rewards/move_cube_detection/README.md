# Tuning files for the move-cube-detection task.
#
#   reward_weights.py  — reward / penalty scalar weights only
#   variables.py       — geometry, thresholds, vision HSV / depth
#
# Vision observation (policy):
#   OpenCV red bbox → center (x, y) → [x/W, y/H, median_depth/max_depth]
#
# Imported by:
#   src/thesis_rl/tasks/move_cube_detection/config/ur5e/env_cfg.py
