"""FastSAC whole-body tracking experiment for A3 T3D0 on Isaac Sim."""

from dataclasses import replace

from holosoma.config_types.experiment import ExperimentConfig, NightlyConfig, TrainingConfig
from holosoma.config_values import algo, simulator, terrain
from holosoma.config_values.robots.a3_t3d0 import a3_t3d0_29dof
from holosoma.config_values.wbt.a3.action import a3_t3d0_joint_pos
from holosoma.config_values.wbt.a3.command import a3_t3d0_wbt_command
from holosoma.config_values.wbt.a3.curriculum import a3_t3d0_wbt_curriculum
from holosoma.config_values.wbt.a3.observation import a3_t3d0_wbt_observation
from holosoma.config_values.wbt.a3.randomization import a3_t3d0_wbt_randomization
from holosoma.config_values.wbt.a3.reward import a3_t3d0_wbt_fast_sac_reward
from holosoma.config_values.wbt.a3.termination import a3_t3d0_wbt_termination

a3_t3d0_wbt_fast_sac = ExperimentConfig(
    training=TrainingConfig(
        project="A3WholeBodyTracking",
        name="a3_t3d0_29dof_wbt_fast_sac",
        num_envs=4096,
    ),
    env_class="holosoma.envs.wbt.wbt_manager.WholeBodyTrackingManager",
    algo=replace(
        algo.fast_sac,
        config=replace(
            algo.fast_sac.config,
            num_learning_iterations=400000,
            v_max=20.0,
            v_min=-20.0,
            gamma=0.99,
            num_steps=1,
            num_updates=4,
            num_atoms=501,
            policy_frequency=2,
            target_entropy_ratio=0.5,
            tau=0.05,
            use_symmetry=False,
        ),
    ),
    # A3 is deliberately supported only through Isaac Sim in this integration.
    simulator=replace(
        simulator.isaacsim,
        config=replace(
            simulator.isaacsim.config,
            sim=replace(simulator.isaacsim.config.sim, max_episode_length_s=10.0),
        ),
    ),
    robot=a3_t3d0_29dof,
    terrain=terrain.terrain_locomotion_plane,
    observation=a3_t3d0_wbt_observation,
    action=a3_t3d0_joint_pos,
    termination=a3_t3d0_wbt_termination,
    randomization=a3_t3d0_wbt_randomization,
    command=a3_t3d0_wbt_command,
    curriculum=a3_t3d0_wbt_curriculum,
    reward=a3_t3d0_wbt_fast_sac_reward,
    nightly=NightlyConfig(
        iterations=200000,
        metrics={
            "Episode/rew_motion_global_ref_position_error_exp": [0.30, "inf"],
            "Episode/rew_motion_relative_body_position_error_exp": [0.80, "inf"],
        },
    ),
)

__all__ = ["a3_t3d0_wbt_fast_sac"]
