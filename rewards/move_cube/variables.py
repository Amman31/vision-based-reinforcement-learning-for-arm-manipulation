"""Geometry and thresholds for UR5e cube push-on-table."""

# Scene
CUBE_START_POS = (-1.4, -0.2, 0.93357)
CUBE_SPAWN_RADIUS = 0.15

# Any of these four pads can register a push.
FINGER_BODY_NAMES = ("left_inner_finger", "right_inner_finger", "left_outer_finger", "right_outer_finger")
FINGER_SENSOR_NAMES = (
    "left_inner_finger_contact",
    "right_inner_finger_contact",
    "left_outer_finger_contact",
    "right_outer_finger_contact",
)

# Success: any finger touched the cube, then it moved at least 1 cm in XY from spawn.
SUCCESS_XY_DISPLACEMENT = 0.01
SUCCESS_DWELL_STEPS = 1
CONTACT_FORCE_THRESHOLD = 0.05
CONTACT_DISTANCE = 0.045

# Approach
APPROACH_DISTANCE_SCALE = 10.0

# Gripper close command (same calibration as pick-and-place).
GRASP_FINGER_TARGET = 0.480

# Vision
DEPTH_MAX_METERS = 3.0

# Episode
EPISODE_LENGTH_S = 4.0
