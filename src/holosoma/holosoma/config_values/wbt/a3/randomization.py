"""Domain randomization preset for A3 T3D0 WBT."""

from holosoma.config_values.wbt.g1.randomization import g1_29dof_wbt_randomization

# These randomization terms operate through RobotConfig names and dimensions.
a3_t3d0_wbt_randomization = g1_29dof_wbt_randomization

__all__ = ["a3_t3d0_wbt_randomization"]
