"""FastSAC WBT reward preset for the A3 T3D0."""

from dataclasses import replace

from holosoma.config_types.reward import RewardTermCfg
from holosoma.config_values.wbt.g1.reward import g1_29dof_wbt_fast_sac_reward

a3_t3d0_wbt_fast_sac_reward = replace(
    g1_29dof_wbt_fast_sac_reward,
    terms={
        **g1_29dof_wbt_fast_sac_reward.terms,
        "undesired_contacts": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:UndesiredContacts",
            params={
                "threshold": 1.0,
                "undesired_contacts_body_names": (
                    "^(?!left_ankle_roll_Link$)(?!right_ankle_roll_Link$)"
                    "(?!left_wrist_yaw_Link$)(?!right_wrist_yaw_Link$).+$"
                ),
            },
            weight=-0.1,
        ),
    },
)

__all__ = ["a3_t3d0_wbt_fast_sac_reward"]
