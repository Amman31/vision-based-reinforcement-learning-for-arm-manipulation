"""ZED X RGB-D → red-cube bounding box → env-frame XYZ for the policy."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from isaaclab.utils.math import quat_apply

# Red-cube PreviewSurface is diffuse (0.8, 0.015, 0.015). Thresholds are in [0, 1].
# Detection is pure PyTorch color thresholding (no OpenCV / no neural detector).
_RED_MIN = 0.35
_RED_MARGIN = 0.15
_MAX_DEPTH_M = 3.0
_MIN_DEPTH_M = 0.05
# Kill thin RTX "red lag" streaks before fitting the bbox.
_OPEN_KERNEL = 3
_MIN_COMPONENT_PIXELS = 8
_STRUCTURE_4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)


def _red_mask(rgb: torch.Tensor) -> torch.Tensor:
    """Boolean red-cube mask, shape (N, H, W)."""
    r, g, b = rgb.unbind(dim=-1)
    return (
        (r > _RED_MIN)
        & (r > g + _RED_MARGIN)
        & (r > b + _RED_MARGIN)
        & (g < 0.40)
        & (b < 0.40)
    )


def _morph_open(mask: torch.Tensor, kernel: int = _OPEN_KERNEL) -> torch.Tensor:
    """Binary opening (erode → dilate) to drop 1–2 px lag noise / thin streaks."""
    if kernel < 2:
        return mask
    x = mask.unsqueeze(1).float()
    eroded = -F.max_pool2d(-x, kernel_size=kernel, stride=1, padding=kernel // 2)
    opened = F.max_pool2d(eroded, kernel_size=kernel, stride=1, padding=kernel // 2)
    return opened.squeeze(1) > 0.5


def _largest_component_mask(mask: torch.Tensor) -> torch.Tensor:
    """Keep only the largest 4-connected red blob per env (drops distant lag islands)."""
    try:
        from scipy import ndimage
    except ImportError:
        return mask

    out = torch.zeros_like(mask)
    mask_np = mask.detach().cpu().numpy()
    for i in range(mask_np.shape[0]):
        labeled, n_labels = ndimage.label(mask_np[i], structure=_STRUCTURE_4)
        if n_labels == 0:
            continue
        counts = ndimage.sum(mask_np[i], labeled, index=list(range(1, n_labels + 1)))
        best = int(np.argmax(counts)) + 1
        if float(counts[best - 1]) < _MIN_COMPONENT_PIXELS:
            continue
        out[i] = torch.as_tensor(labeled == best, device=mask.device, dtype=torch.bool)
    return out


def _clean_detection_mask(rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
    """Red mask cleaned of sim lag: require valid depth, open, keep largest blob."""
    mask = _red_mask(rgb)
    depth_ok = (depth > _MIN_DEPTH_M) & (depth < _MAX_DEPTH_M)
    # Ghost / lag pixels often have invalid depth — drop them first.
    mask = mask & depth_ok
    mask = _morph_open(mask)
    return _largest_component_mask(mask)


def _axis_aligned_bbox(mask: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Axis-aligned bbox from a boolean mask.

    Returns ``(u_min, u_max, v_min, v_max, has_det)`` each shape (N,).
    Center is ``0.5*(min+max)`` for u and v.
    """
    n_envs, height, width = mask.shape
    device = mask.device
    dtype = torch.float32

    vs = torch.arange(height, device=device, dtype=dtype).view(1, height, 1).expand(n_envs, height, width)
    us = torch.arange(width, device=device, dtype=dtype).view(1, 1, width).expand(n_envs, height, width)

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
    return u_min, u_max, v_min, v_max, has_det


def cube_position_from_camera(env):
    """Estimate cube XYZ in the env frame from ZED X RGB + depth.

    Pipeline (same as intended sim/real deployment)::

        RGB → red threshold ∩ valid depth → morph open → largest blob
        (u, v) = bbox center of that blob
        Z = mean depth on blob pixels (not empty bbox interior)
        P_cam = [(u-cx)Z/fx, (v-cy)Z/fy, Z]
        P_world = R_ros(P_cam) + t
        P_env = P_world - env_origin

    Cleaning rejects RTX red-lag noise that would otherwise inflate the AABB.
    """
    camera = env.scene["depth_camera"]
    images = camera.data.output

    rgb = images["rgb"].torch[..., :3].float()
    if rgb.max() > 1.5:
        rgb = rgb / 255.0

    depth = images["distance_to_image_plane"].torch.float()
    if depth.ndim == 4:
        depth = depth[..., 0]
    # Important: nan*0 == nan in PyTorch, so clear invalids before masking.
    depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

    mask = _clean_detection_mask(rgb, depth)
    u_min, u_max, v_min, v_max, has_det = _axis_aligned_bbox(mask)
    u = 0.5 * (u_min + u_max)
    v = 0.5 * (v_min + v_max)

    dtype = rgb.dtype
    # Depth from cube pixels only (not empty space inside an oversized AABB).
    depth_mask_f = mask.to(dtype)
    depth_count = depth_mask_f.sum(dim=(1, 2)).clamp_min(1.0)
    z = (depth * depth_mask_f).sum(dim=(1, 2)) / depth_count

    valid = has_det & (depth_mask_f.sum(dim=(1, 2)) > 0)

    k = camera.data.intrinsic_matrices.torch
    fx, fy = k[:, 0, 0], k[:, 1, 1]
    cx, cy = k[:, 0, 2], k[:, 1, 2]

    x_cam = (u - cx) * z / fx
    y_cam = (v - cy) * z / fy
    p_cam = torch.stack((x_cam, y_cam, z), dim=-1)

    p_world = quat_apply(camera.data.quat_w_ros.torch, p_cam) + camera.data.pos_w.torch
    p_env = p_world - env.scene.env_origins

    return torch.where(valid.unsqueeze(-1), p_env, torch.zeros_like(p_env))
