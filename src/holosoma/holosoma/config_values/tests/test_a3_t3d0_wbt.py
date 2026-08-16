"""Pure-CPU checks for the A3 T3D0 Isaac Sim WBT integration."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from holosoma.config_values.robots.a3_t3d0 import (
    A3_T3D0_BODY_NAMES,
    A3_T3D0_DOF_NAMES,
    a3_t3d0_29dof,
)
from holosoma.config_values.wbt.a3.command import A3_T3D0_TRACKED_BODY_NAMES, motion_config
from holosoma.config_values.wbt.a3.experiment import a3_t3d0_wbt_fast_sac
from holosoma.utils.path import resolve_asset_path, resolve_data_file_path


def _urdf_path() -> Path:
    return Path(resolve_asset_path(a3_t3d0_29dof.asset.urdf_file, a3_t3d0_29dof.asset.asset_root))


def test_a3_training_urdf_fixes_neck_and_exposes_29_dofs():
    root = ET.parse(_urdf_path()).getroot()  # noqa: S314 - trusted bundled URDF
    joint_types = {joint.attrib["name"]: joint.attrib["type"] for joint in root.findall("joint")}
    active_joint_names = [name for name, joint_type in joint_types.items() if joint_type != "fixed"]

    assert joint_types["head_yaw_joint"] == "fixed"
    assert joint_types["head_pitch_joint"] == "fixed"
    assert len(active_joint_names) == 29
    assert set(active_joint_names) == set(A3_T3D0_DOF_NAMES)


def test_a3_robot_config_dimensions_and_asset_names_are_consistent():
    lengths = {
        len(a3_t3d0_29dof.dof_names),
        len(a3_t3d0_29dof.dof_pos_lower_limit_list),
        len(a3_t3d0_29dof.dof_pos_upper_limit_list),
        len(a3_t3d0_29dof.dof_vel_limit_list),
        len(a3_t3d0_29dof.dof_effort_limit_list),
        len(a3_t3d0_29dof.dof_armature_list),
        len(a3_t3d0_29dof.dof_joint_friction_list),
        len(a3_t3d0_29dof.init_state.default_joint_angles),
    }
    assert lengths == {29}
    assert a3_t3d0_29dof.actions_dim == 29
    assert a3_t3d0_29dof.dof_obs_size == 29
    assert a3_t3d0_29dof.num_bodies == len(A3_T3D0_BODY_NAMES) == 30
    assert set(A3_T3D0_TRACKED_BODY_NAMES).issubset(A3_T3D0_BODY_NAMES)

    root = ET.parse(_urdf_path()).getroot()  # noqa: S314 - trusted bundled URDF
    child_for_active_joint = {
        joint.find("child").attrib["link"] for joint in root.findall("joint") if joint.attrib["type"] != "fixed"
    }
    assert set(A3_T3D0_BODY_NAMES) == {"pelvis_link", *child_for_active_joint}


def test_bundled_a3_motion_matches_robot_and_wbt_command():
    motion_path = Path(resolve_data_file_path(motion_config.motion_file))
    with np.load(motion_path) as motion:
        assert motion["joint_pos"].shape[1] == 7 + 29
        assert motion["joint_vel"].shape[1] == 6 + 29
        assert motion["body_pos_w"].shape[1] == 30
        assert motion["body_quat_w"].shape[1] == 30
        assert motion["joint_names"].tolist() == A3_T3D0_DOF_NAMES
        assert motion["body_names"].tolist() == A3_T3D0_BODY_NAMES
        assert np.isfinite(motion["joint_pos"]).all()
        assert np.isfinite(motion["body_pos_w"]).all()


def test_a3_experiment_is_isaacsim_fast_sac_only():
    assert a3_t3d0_wbt_fast_sac.simulator.config.name == "isaacsim"
    assert a3_t3d0_wbt_fast_sac.algo._target_.endswith("fast_sac_agent.FastSACAgent")
    assert a3_t3d0_wbt_fast_sac.robot is a3_t3d0_29dof
    assert a3_t3d0_wbt_fast_sac.training.num_envs == 4096


def test_a3_pd_gains_match_hope_training_plant_baseline():
    assert a3_t3d0_29dof.control.stiffness == {
        "hip_pitch": 80.0,
        "hip_roll": 120.0,
        "hip_yaw": 80.0,
        "knee": 250.0,
        "ankle_pitch": 50.0,
        "ankle_roll": 50.0,
        "waist_yaw": 85.0,
        "waist_roll": 50.0,
        "waist_pitch": 50.0,
        "shoulder_pitch": 40.0,
        "shoulder_roll": 40.0,
        "shoulder_yaw": 30.0,
        "elbow": 30.0,
        "wrist_roll": 30.0,
        "wrist_pitch": 20.0,
        "wrist_yaw": 20.0,
    }
    assert a3_t3d0_29dof.control.damping == {
        "hip_pitch": 4.0,
        "hip_roll": 5.0,
        "hip_yaw": 4.0,
        "knee": 10.0,
        "ankle_pitch": 4.0,
        "ankle_roll": 4.0,
        "waist_yaw": 4.0,
        "waist_roll": 2.5,
        "waist_pitch": 2.8,
        "shoulder_pitch": 4.5,
        "shoulder_roll": 4.5,
        "shoulder_yaw": 3.0,
        "elbow": 3.0,
        "wrist_roll": 3.0,
        "wrist_pitch": 3.0,
        "wrist_yaw": 3.0,
    }
