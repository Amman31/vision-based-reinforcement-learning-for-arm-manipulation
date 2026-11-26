#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train with Isaac Lab (auto-export happens inside thesis_rl after learn).

Plain ``isaaclab train`` already exports via the package hook. This helper
just forwards arguments to ``isaaclab`` for convenience.

Example::

    python scripts/train_and_export.py -- \\
        train --rl_library rsl_rl --task ThesisRl-Move-Cube-UR5E \\
        --num_envs 256 --max_iterations 50 --enable_cameras --headless \\
        physics=isaacsim_physx
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Forward to isaaclab; thesis_rl auto-exports after train"
    )
    parser.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to ``isaaclab`` (include leading train ...)",
    )
    args = parser.parse_args()

    train_argv = list(args.train_args)
    if train_argv and train_argv[0] == "--":
        train_argv = train_argv[1:]
    if not train_argv:
        raise SystemExit(
            "Pass isaaclab arguments after -- , e.g.\n"
            "  python scripts/train_and_export.py -- "
            "train --rl_library rsl_rl --task ThesisRl-Move-Cube-UR5E ..."
        )

    train_cmd = ["isaaclab", *train_argv]
    print(f"[INFO] Training : {' '.join(train_cmd)}", flush=True)
    raise SystemExit(subprocess.call(train_cmd))


if __name__ == "__main__":
    main()
