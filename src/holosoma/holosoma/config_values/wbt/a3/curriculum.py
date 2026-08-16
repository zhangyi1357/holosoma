"""Whole-body tracking curriculum for the A3 T3D0."""

from holosoma.config_values.wbt.g1.curriculum import g1_29dof_wbt_curriculum

# The curriculum contains only the simulator-agnostic average episode tracker.
a3_t3d0_wbt_curriculum = g1_29dof_wbt_curriculum

__all__ = ["a3_t3d0_wbt_curriculum"]
