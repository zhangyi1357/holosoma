"""Asymmetric actor/critic WBT observations for the A3 T3D0."""

from holosoma.config_values.wbt.g1.observation import g1_29dof_wbt_observation

# WBT observation terms obtain their dimensions from the active robot and
# tracked-body command, so the same term graph is valid for A3's 29 policy DoFs.
a3_t3d0_wbt_observation = g1_29dof_wbt_observation

__all__ = ["a3_t3d0_wbt_observation"]
