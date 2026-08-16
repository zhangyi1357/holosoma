"""Whole-body tracking termination preset for the A3 T3D0."""

from holosoma.config_types.termination import TerminationManagerCfg, TerminationTermCfg
from holosoma.config_values.wbt.a3.command import A3_T3D0_TRACKED_BODY_NAMES

a3_t3d0_wbt_termination = TerminationManagerCfg(
    terms={
        "timeout": TerminationTermCfg(
            func="holosoma.managers.termination.terms.common:timeout_exceeded",
            is_timeout=True,
        ),
        "bad_tracking": TerminationTermCfg(
            func="holosoma.managers.termination.terms.wbt:BadTrackingZOnly",
            params={
                "bad_ref_pos_threshold": 0.5,
                "bad_ref_ori_threshold": 0.8,
                "bad_motion_body_pos_threshold": 0.30,
                "body_names_to_track": A3_T3D0_TRACKED_BODY_NAMES,
                "bad_motion_body_pos_body_names": [
                    "left_ankle_roll_Link",
                    "right_ankle_roll_Link",
                    "left_wrist_yaw_Link",
                    "right_wrist_yaw_Link",
                ],
                "bad_object_pos_threshold": 0.25,
                "bad_object_ori_threshold": 0.8,
            },
        ),
    }
)

__all__ = ["a3_t3d0_wbt_termination"]
