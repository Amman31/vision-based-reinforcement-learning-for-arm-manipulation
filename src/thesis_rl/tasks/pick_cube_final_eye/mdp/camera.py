"""Camera acquisition: read RGB + depth from the simulated ZED X CameraLeft."""

from __future__ import annotations

import torch


def read_rgb_depth(env, sensor_name: str = "depth_camera"):
    """Return ``(rgb, depth)`` as float tensors.

    * ``rgb``: (N, H, W, 3) in [0, 1]
    * ``depth``: (N, H, W) meters, NaN/Inf cleared to 0
    """
    camera = env.scene[sensor_name]
    images = camera.data.output
    if images is None or "rgb" not in images or "distance_to_image_plane" not in images:
        raise RuntimeError(
            f"Camera '{sensor_name}' missing rgb/depth. Check CameraCfg.data_types."
        )

    rgb = images["rgb"].torch[..., :3].float()
    if rgb.max() > 1.5:
        rgb = rgb / 255.0

    depth = images["distance_to_image_plane"].torch.float()
    if depth.ndim == 4:
        depth = depth[..., 0]
    depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    return rgb, depth
