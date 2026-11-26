# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task registrations for the generated project."""

from thesis_rl.auto_export import install_auto_export_hook

from isaaclab_tasks.utils import import_packages

# Hook RSL-RL learn() so results/plots/model are written after isaaclab train.
install_auto_export_hook()

import_packages(__name__, ["utils", ".mdp"])
