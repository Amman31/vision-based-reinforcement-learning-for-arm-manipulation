#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Compare ground-truth cube XYZ vs ZED X camera detection.

Runs ``ThesisRl-Pick-Cube-Detection-UR5E``, samples several resets/steps, and
reports error statistics. Also probes ROS / world / OpenGL camera-frame
conventions so we can fix ``cube_position_from_camera`` if needed.

Example::

    python scripts/compare_cube_detection.py \\
        --task ThesisRl-Pick-Cube-Detection-UR5E \\
        --num_envs 4 --samples 64 --headless \\
        physics=isaacsim_physx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.utils.math import quat_apply, quat_apply_inverse
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

import isaaclab_tasks  # noqa: F401
import thesis_rl.tasks  # noqa: F401

from thesis_rl.tasks.pick_cube_detection.mdp.observations import cube_position_from_camera
from thesis_rl.tasks.pick_cube_detection.mdp.rewards import cube_position

DEFAULT_TASK = "ThesisRl-Pick-Cube-Detection-UR5E"
_TASK_ALIASES = {
    "pick_cube_detection": DEFAULT_TASK,
}


def _resolve_task_id(task: str) -> str:
    key = task.strip()
    return _TASK_ALIASES.get(key, key)


def _camera_quat(camera, convention: str) -> torch.Tensor:
    if convention == "ros":
        return camera.data.quat_w_ros.torch
    if convention == "world":
        return camera.data.quat_w_world.torch
    if convention == "opengl":
        return camera.data.quat_w_opengl.torch
    raise ValueError(convention)


def _detect_with_convention(env, convention: str) -> tuple[torch.Tensor, dict]:
    """Same bbox pipeline as observations.py, but selectable camera-frame convention."""
    camera = env.scene["depth_camera"]
    images = camera.data.output

    rgb = images["rgb"].torch[..., :3].float()
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    depth = images["distance_to_image_plane"].torch.float()
    if depth.ndim == 4:
        depth = depth[..., 0]
    depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

    r, g, b = rgb.unbind(dim=-1)
    mask = (r > 0.35) & (r > g + 0.15) & (r > b + 0.15) & (g < 0.40) & (b < 0.40)

    n_envs, height, width = mask.shape
    device, dtype = mask.device, rgb.dtype
    vs = torch.arange(height, device=device, dtype=dtype).view(1, height, 1).expand_as(mask)
    us = torch.arange(width, device=device, dtype=dtype).view(1, 1, width).expand_as(mask)

    u_min = torch.where(mask, us, torch.full_like(us, float("inf"))).amin(dim=(1, 2))
    u_max = torch.where(mask, us, torch.full_like(us, float("-inf"))).amax(dim=(1, 2))
    v_min = torch.where(mask, vs, torch.full_like(vs, float("inf"))).amin(dim=(1, 2))
    v_max = torch.where(mask, vs, torch.full_like(vs, float("-inf"))).amax(dim=(1, 2))
    has_det = mask.any(dim=(1, 2))
    zero = torch.zeros_like(u_min)
    u_min = torch.where(has_det, u_min, zero)
    u_max = torch.where(has_det, u_max, zero)
    v_min = torch.where(has_det, v_min, zero)
    v_max = torch.where(has_det, v_max, zero)
    u = 0.5 * (u_min + u_max)
    v = 0.5 * (v_min + v_max)

    in_bbox = (
        (us >= u_min[:, None, None])
        & (us <= u_max[:, None, None])
        & (vs >= v_min[:, None, None])
        & (vs <= v_max[:, None, None])
        & has_det[:, None, None]
    )
    depth_ok = (depth > 0.05) & (depth < 3.0)
    depth_mask_f = (in_bbox & depth_ok).to(dtype)
    depth_count = depth_mask_f.sum(dim=(1, 2)).clamp_min(1.0)
    z = (depth * depth_mask_f).sum(dim=(1, 2)) / depth_count
    valid = has_det & (depth_mask_f.sum(dim=(1, 2)) > 0)

    k = camera.data.intrinsic_matrices.torch
    fx, fy, cx, cy = k[:, 0, 0], k[:, 1, 1], k[:, 0, 2], k[:, 1, 2]
    p_cam = torch.stack(((u - cx) * z / fx, (v - cy) * z / fy, z), dim=-1)
    p_world = quat_apply(_camera_quat(camera, convention), p_cam) + camera.data.pos_w.torch
    p_env = p_world - env.scene.env_origins
    p_env = torch.where(valid.unsqueeze(-1), p_env, torch.full_like(p_env, float("nan")))

    stats = {
        "mask_pixels": mask.to(dtype).sum(dim=(1, 2)),
        "valid": valid,
        "u": u,
        "v": v,
        "z": z,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
    }
    return p_env, stats


def _project_gt_to_uv(env) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project GT cube (env frame) into CameraLeft pixel coords with ROS convention."""
    camera = env.scene["depth_camera"]
    p_world = cube_position(env) + env.scene.env_origins
    p_cam = quat_apply_inverse(camera.data.quat_w_ros.torch, p_world - camera.data.pos_w.torch)
    k = camera.data.intrinsic_matrices.torch
    fx, fy, cx, cy = k[:, 0, 0], k[:, 1, 1], k[:, 0, 2], k[:, 1, 2]
    z = p_cam[:, 2].clamp_min(1e-4)
    u = fx * p_cam[:, 0] / z + cx
    v = fy * p_cam[:, 1] / z + cy
    return u, v, z


def _err_stats(name: str, err: torch.Tensor, valid: torch.Tensor, unit: str = "m") -> None:
    if valid.any():
        e = err[valid]
        finite = torch.isfinite(e)
        if not finite.any():
            print(f"  {name:8s}  n={int(valid.sum()):4d}  (all non-finite)")
            return
        e = e[finite]
        print(
            f"  {name:8s}  n={int(finite.sum()):4d}  "
            f"mean={e.mean():.4f}  median={e.median():.4f}  "
            f"p95={e.quantile(0.95):.4f}  max={e.max():.4f} {unit}"
        )
    else:
        print(f"  {name:8s}  n=0 (no valid detections)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare GT vs camera-detected cube XYZ.")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--samples", type=int, default=64, help="Number of env-steps to accumulate.")
    parser.add_argument("--steps_per_reset", type=int, default=8)
    parser.add_argument("--tol", type=float, default=0.03, help="Pass threshold on median L2 error (m).")
    add_launcher_args(parser)
    args_cli, hydra_args = setup_preset_cli(parser)
    sys.argv = [sys.argv[0]] + hydra_args

    task_id = _resolve_task_id(args_cli.task)
    env_cfg, _ = resolve_task_config(task_id, "")
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    print(f"[INFO] Task     : {task_id}", flush=True)
    print(f"[INFO] Envs     : {args_cli.num_envs}", flush=True)
    print(f"[INFO] Samples  : {args_cli.samples}", flush=True)

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(task_id, cfg=env_cfg)
        try:
            unwrapped = env.unwrapped
            env.reset()
            action = torch.zeros(env.action_space.shape, device=unwrapped.device)

            gt_all: list[torch.Tensor] = []
            cam_all = {c: [] for c in ("ros", "world", "opengl")}
            obs_all: list[torch.Tensor] = []
            valid_all = {c: [] for c in ("ros", "world", "opengl")}
            mask_pix: list[torch.Tensor] = []
            uv_err: list[torch.Tensor] = []
            depth_err: list[torch.Tensor] = []
            intrinsics_logged = False

            collected = 0
            while collected < args_cli.samples:
                env.reset()
                for _ in range(args_cli.steps_per_reset):
                    env.step(action)
                    camera = unwrapped.scene["depth_camera"]
                    if hasattr(camera, "update"):
                        camera.update(0.0)

                    if not intrinsics_logged:
                        k = camera.data.intrinsic_matrices.torch[0]
                        print(
                            f"[INFO] Intrinsics env0: fx={k[0,0]:.3f} fy={k[1,1]:.3f} "
                            f"cx={k[0,2]:.3f} cy={k[1,2]:.3f}",
                            flush=True,
                        )
                        print(f"[INFO] Camera pos_w0 : {camera.data.pos_w.torch[0].tolist()}", flush=True)
                        intrinsics_logged = True

                    gt = cube_position(unwrapped)
                    obs_cam = cube_position_from_camera(unwrapped)
                    gt_all.append(gt)
                    obs_all.append(obs_cam)

                    u_gt, v_gt, z_gt = _project_gt_to_uv(unwrapped)
                    for conv in cam_all:
                        est, stats = _detect_with_convention(unwrapped, conv)
                        cam_all[conv].append(est)
                        valid_all[conv].append(stats["valid"])
                        if conv == "ros":
                            mask_pix.append(stats["mask_pixels"])
                            uv_err.append(
                                torch.stack(
                                    (stats["u"] - u_gt, stats["v"] - v_gt), dim=-1
                                ).norm(dim=-1)
                            )
                            depth_err.append((stats["z"] - z_gt).abs())

                    collected += gt.shape[0]
                    if collected >= args_cli.samples:
                        break

            gt_t = torch.cat(gt_all, dim=0)
            obs_t = torch.cat(obs_all, dim=0)
            print("\n==========================================")
            print(" CUBE DETECTION vs GROUND TRUTH")
            print("==========================================")
            mask_t = torch.cat(mask_pix, dim=0)
            print(
                f"Red mask pixels: mean={mask_t.float().mean():.1f}  "
                f"min={int(mask_t.min())}  max={int(mask_t.max())}  "
                f"zero_frac={(mask_t == 0).float().mean():.2%}"
            )
            uv_t = torch.cat(uv_err, dim=0)
            z_t = torch.cat(depth_err, dim=0)
            ros_valid = torch.cat(valid_all["ros"], dim=0)
            _err_stats("UV(px)", uv_t, ros_valid, unit="px")
            _err_stats("depth", z_t, ros_valid, unit="m")

            print("\nL2 error by camera-frame convention:")
            best_name, best_med = None, float("inf")
            for conv in ("ros", "world", "opengl"):
                est = torch.cat(cam_all[conv], dim=0)
                valid = torch.cat(valid_all[conv], dim=0)
                err = (est - gt_t).norm(dim=-1)
                _err_stats(conv, err, valid)
                if valid.any():
                    med = float(err[valid].median())
                    if med < best_med:
                        best_med, best_name = med, conv

            # Production observation function (uses ROS today).
            obs_valid = obs_t.isfinite().all(dim=-1) & (obs_t.norm(dim=-1) > 1e-6)
            obs_err = (obs_t - gt_t).norm(dim=-1)
            print("\nProduction cube_position_from_camera (ROS):")
            _err_stats("obs", obs_err, obs_valid)
            if obs_valid.any():
                axis = (obs_t[obs_valid] - gt_t[obs_valid]).abs().mean(dim=0)
                print(f"  mean |dx,dy,dz| = [{axis[0]:.4f}, {axis[1]:.4f}, {axis[2]:.4f}] m")
                print("  sample GT vs cam (first 5 valid):")
                idx = obs_valid.nonzero(as_tuple=False).flatten()[:5]
                for i in idx.tolist():
                    g, c = gt_t[i], obs_t[i]
                    print(
                        f"    gt=[{g[0]:.3f},{g[1]:.3f},{g[2]:.3f}]  "
                        f"cam=[{c[0]:.3f},{c[1]:.3f},{c[2]:.3f}]  "
                        f"err={obs_err[i]:.3f} m"
                    )

            print("------------------------------------------")
            if best_name is None:
                print("RESULT: FAIL — no valid red-cube detections. Check lighting / thresholds.")
                code = 2
            elif best_med <= args_cli.tol:
                print(
                    f"RESULT: PASS — best convention={best_name}, "
                    f"median L2={best_med:.4f} m <= tol={args_cli.tol:.3f} m"
                )
                code = 0
            else:
                print(
                    f"RESULT: FAIL — best convention={best_name}, "
                    f"median L2={best_med:.4f} m > tol={args_cli.tol:.3f} m"
                )
                code = 1
            print(f"RECOMMEND_CONVENTION={best_name}")
            print("==========================================\n")
            raise SystemExit(code)
        finally:
            env.close()


if __name__ == "__main__":
    main()
