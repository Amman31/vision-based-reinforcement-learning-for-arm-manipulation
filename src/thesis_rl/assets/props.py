"""Shared tabletop scene props: table, cube, spawn-outline markers."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

from thesis_rl.assets.ur5e import ASSET_DIR, TABLE_HEIGHT

# Default cube XY and half-extents of the uniform spawn box (shared by EventCfg + outline).
CUBE_INIT_XY = (0.20, 0.0)
CUBE_SPAWN_HALF_XY = (0.15, 0.15)

_SPAWN_LINE_W = 0.006
_SPAWN_LINE_H = 0.002
_SPAWN_OUTLINE_Z = TABLE_HEIGHT + 0.001

TABLE_CFG = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/Table",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(ASSET_DIR / "table.usd"),
        scale=(0.0014, 0.00181, 0.001),
    ),
    init_state=AssetBaseCfg.InitialStateCfg(
        pos=(0.0538809793, 0.0, TABLE_HEIGHT - 0.7465264463),
    ),
)

CUBE_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Cube",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(ASSET_DIR / "cube_5cm.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.08),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.015, 0.015), roughness=0.7),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(CUBE_INIT_XY[0], CUBE_INIT_XY[1], TABLE_HEIGHT + 0.027),
    ),
)

def _white_bar(name: str, size_xy: tuple[float, float], pos_xy: tuple[float, float]) -> AssetBaseCfg:
    """Visual-only white bar (no collision) for one edge of the spawn outline."""
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CuboidCfg(
            size=(size_xy[0], size_xy[1], _SPAWN_LINE_H),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0), roughness=0.4),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(pos_xy[0], pos_xy[1], _SPAWN_OUTLINE_Z)),
    )


# White outline of the cube spawn square (visual only).
SPAWN_OUTLINE_X_NEG_CFG = _white_bar(
    "SpawnOutline_XNeg",
    (_SPAWN_LINE_W, 2.0 * CUBE_SPAWN_HALF_XY[1]),
    (CUBE_INIT_XY[0] - CUBE_SPAWN_HALF_XY[0], CUBE_INIT_XY[1]),
)
SPAWN_OUTLINE_X_POS_CFG = _white_bar(
    "SpawnOutline_XPos",
    (_SPAWN_LINE_W, 2.0 * CUBE_SPAWN_HALF_XY[1]),
    (CUBE_INIT_XY[0] + CUBE_SPAWN_HALF_XY[0], CUBE_INIT_XY[1]),
)
SPAWN_OUTLINE_Y_NEG_CFG = _white_bar(
    "SpawnOutline_YNeg",
    (2.0 * CUBE_SPAWN_HALF_XY[0], _SPAWN_LINE_W),
    (CUBE_INIT_XY[0], CUBE_INIT_XY[1] - CUBE_SPAWN_HALF_XY[1]),
)
SPAWN_OUTLINE_Y_POS_CFG = _white_bar(
    "SpawnOutline_YPos",
    (2.0 * CUBE_SPAWN_HALF_XY[0], _SPAWN_LINE_W),
    (CUBE_INIT_XY[0], CUBE_INIT_XY[1] + CUBE_SPAWN_HALF_XY[1]),
)

__all__ = [
    "CUBE_CFG",
    "CUBE_INIT_XY",
    "CUBE_SPAWN_HALF_XY",
    "SPAWN_OUTLINE_X_NEG_CFG",
    "SPAWN_OUTLINE_X_POS_CFG",
    "SPAWN_OUTLINE_Y_NEG_CFG",
    "SPAWN_OUTLINE_Y_POS_CFG",
    "TABLE_CFG",
]
