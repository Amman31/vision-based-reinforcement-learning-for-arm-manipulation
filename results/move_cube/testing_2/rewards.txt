# Approach fingers toward the cube.
APPROACH = 1.0

# Any finger touching the cube.
CONTACT = 3.0

# XY displacement from spawn, gated on any-finger contact.
PUSH = 7.0

# Episode success: cube moved >= 1 cm in XY after any finger touched it.
SUCCESS = 15.0

# Regularization
STEP_PENALTY = -2.0e-4
ACTION_RATE = -2.0e-3