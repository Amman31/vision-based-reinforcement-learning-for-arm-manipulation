import argparse
import sys

import gymnasium as gym

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

import isaaclab_tasks  # noqa: F401
import thesis_rl.tasks  # noqa: F401


TASK_NAME = "ThesisRl-Move-Cube-UR5E"


def main():

    parser = argparse.ArgumentParser(
        description="Find the exact runtime prim path of wrist_3_link."
    )

    parser.add_argument(
        "--task",
        type=str,
        default=TASK_NAME,
    )

    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
    )

    add_launcher_args(parser)

    args_cli, hydra_args = setup_preset_cli(parser)
    sys.argv = [sys.argv[0]] + hydra_args

    env_cfg, _ = resolve_task_config(
        args_cli.task,
        "",
    )

    env_cfg.scene.num_envs = args_cli.num_envs

    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    with launch_simulation(env_cfg, args_cli):

        # Import Isaac Sim USD tools only after Kit has launched.
        import omni.usd

        env = gym.make(
            args_cli.task,
            cfg=env_cfg,
        )

        env.reset()

        stage = omni.usd.get_context().get_stage()

        print("\n======================================")
        print(" WRIST_3_LINK PRIM PATHS")
        print("======================================")

        paths = []

        for prim in stage.Traverse():

            if prim.GetName() == "wrist_3_link":
                paths.append(str(prim.GetPath()))

        if len(paths) == 0:
            print("No wrist_3_link prim found.")
        else:
            for path in paths:
                print(path)

        print("======================================\n")

        env.close()


if __name__ == "__main__":
    main()