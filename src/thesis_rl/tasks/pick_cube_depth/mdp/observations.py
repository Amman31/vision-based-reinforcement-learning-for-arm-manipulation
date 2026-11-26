"""Direct RGB-D camera observations for the visual policy."""

import torch


def camera_rgbd(env):
    """Return float32 RGB-D images as (N, 4, H, W), scaled to [0, 1].

    RGB retains all colors. Depth is image-plane distance in meters, clipped
    to the camera's 3 m range; missing depth is represented as maximum range.
    """
    images = env.scene["depth_camera"].data.output
    rgb = images["rgb"].torch[..., :3].float() / 255.0
    depth = images["distance_to_image_plane"].torch.float()
    depth = torch.nan_to_num(depth, nan=3.0, posinf=3.0, neginf=0.0).clamp(0.0, 3.0) / 3.0
    return torch.cat((rgb, depth), dim=-1).permute(0, 3, 1, 2).contiguous()
