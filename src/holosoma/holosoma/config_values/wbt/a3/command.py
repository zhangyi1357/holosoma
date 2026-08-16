"""Whole-body tracking motion command for the A3 T3D0."""

from holosoma.config_types.command import CommandManagerCfg, CommandTermCfg, MotionConfig, NoiseToInitialPoseConfig

A3_T3D0_TRACKED_BODY_NAMES = [
    "pelvis_link",
    "left_hip_roll_Link",
    "left_knee_Link",
    "left_ankle_roll_Link",
    "right_hip_roll_Link",
    "right_knee_Link",
    "right_ankle_roll_Link",
    "torso_Link",
    "left_shoulder_roll_Link",
    "left_elbow_Link",
    "left_wrist_yaw_Link",
    "right_shoulder_roll_Link",
    "right_elbow_Link",
    "right_wrist_yaw_Link",
]

motion_config = MotionConfig(
    motion_file=("holosoma/data/motions/a3_t3d0/whole_body_tracking/lafan1/training_clips/walk1_subject1_0s_30s.npz"),
    body_names_to_track=A3_T3D0_TRACKED_BODY_NAMES,
    body_name_ref=["torso_Link"],
    use_adaptive_timesteps_sampler=True,
    noise_to_initial_pose=NoiseToInitialPoseConfig(
        overall_noise_scale=1.0,
        dof_pos=0.08,
        root_pos=[0.05, 0.05, 0.01],
        root_rot=[0.1, 0.1, 0.2],
        root_lin_vel=[0.5, 0.5, 0.2],
        root_ang_vel=[0.52, 0.52, 0.78],
        object_pos=[0.0, 0.0, 0.0],
    ),
)

a3_t3d0_wbt_command = CommandManagerCfg(
    params={},
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={"motion_config": motion_config},
        ),
    },
    reset_terms={
        "motion_command": CommandTermCfg(func="holosoma.managers.command.terms.wbt:MotionCommand"),
    },
    step_terms={
        "motion_command": CommandTermCfg(func="holosoma.managers.command.terms.wbt:MotionCommand"),
    },
)

__all__ = ["A3_T3D0_TRACKED_BODY_NAMES", "a3_t3d0_wbt_command"]
