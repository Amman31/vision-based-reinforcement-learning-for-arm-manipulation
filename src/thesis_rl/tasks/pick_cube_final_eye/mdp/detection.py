"""Red-cube detection in RGB (bbox center + depth on the blob).

Uses the same cleaning pipeline as ``pick_cube_detection`` (valid-depth ∩ morph
open ∩ largest connected component) so RTX red-lag noise does not inflate the
bbox. Batched on GPU via PyTorch (no OpenCV dependency for vectorized envs).
"""

from __future__ import annotations

import torch

from thesis_rl.tasks.pick_cube_detection.mdp.observations import (
    _axis_aligned_bbox,
    _clean_detection_mask,
)

_MAX_DEPTH_M = 3.0
_MIN_DEPTH_M = 0.05


def detect_red_cube(rgb: torch.Tensor, depth: torch.Tensor):
    """Detect the red cube and return image location + depth.

    Returns
    -------
    u, v : (N,) float
        Bbox-center pixel coordinates.
    z : (N,) float
        Mean valid depth (m) on the cleaned red blob.
    valid : (N,) bool
        True when a blob with valid depth was found.
    mask : (N, H, W) bool
        Cleaned detection mask (largest red blob).
    """
    mask = _clean_detection_mask(rgb, depth)
    u_min, u_max, v_min, v_max, has_det = _axis_aligned_bbox(mask)
    u = 0.5 * (u_min + u_max)
    v = 0.5 * (v_min + v_max)

    depth_ok = (depth > _MIN_DEPTH_M) & (depth < _MAX_DEPTH_M)
    depth_mask = mask & depth_ok
    depth_f = depth_mask.to(rgb.dtype)
    count = depth_f.sum(dim=(1, 2)).clamp_min(1.0)
    z = (depth * depth_f).sum(dim=(1, 2)) / count
    valid = has_det & (depth_f.sum(dim=(1, 2)) > 0)
    return u, v, z, valid, mask
