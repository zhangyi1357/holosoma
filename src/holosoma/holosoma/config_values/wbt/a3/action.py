"""Joint-position action preset for the A3 T3D0."""

from holosoma.config_types.action import ActionManagerCfg, ActionTermCfg

a3_t3d0_joint_pos = ActionManagerCfg(
    terms={
        "joint_control": ActionTermCfg(
            func="holosoma.managers.action.terms.joint_control:JointPositionActionTerm",
            params={},
            scale=1.0,
            clip=None,
        ),
    }
)

__all__ = ["a3_t3d0_joint_pos"]
