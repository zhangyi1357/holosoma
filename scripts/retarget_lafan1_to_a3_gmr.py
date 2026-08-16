#!/usr/bin/env python3
"""Retarget a LAFAN1 BVH clip to the fixed-neck A3 T3D0 with GMR.

The script uses an external GMR checkout without modifying it, builds a
temporary floating-base URDF from the Isaac Sim training asset, runs GMR
headlessly, and writes a Holosoma whole-body-tracking NPZ plus diagnostics.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import types
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import mujoco as mj
import numpy as np
from prepare_a3_t3d0_wbt_motion import (
    A3_BODY_NAMES,
    A3_DOF_NAMES,
    DEFAULT_URDF,
    _angular_velocity,
    _differentiate,
    _forward_kinematics,
    _matrix_to_quat_wxyz,
    _parse_urdf,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "retargeting" / "gmr" / "bvh_lafan1_to_a3_t3d0.json"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "src"
    / "holosoma"
    / "holosoma"
    / "data"
    / "motions"
    / "a3_t3d0"
    / "whole_body_tracking"
    / "lafan1"
    / "walk1_subject1.npz"
)
ROBOT_KEY = "agibot_a3_t3d0"


def _load_gmr(gmr_root: Path):
    """Load only GMR's headless retargeting modules.

    Upstream GMR imports its viewer and torch kinematics from package
    ``__init__``.  Those components are unrelated to offline BVH retargeting,
    so a namespace package is used here to keep the dependency set minimal.
    """
    package_dir = gmr_root / "general_motion_retargeting"
    if not (package_dir / "motion_retarget.py").is_file():
        raise FileNotFoundError(f"Not a GMR checkout: {gmr_root}")

    package = types.ModuleType("general_motion_retargeting")
    package.__path__ = [str(package_dir)]
    package.__package__ = "general_motion_retargeting"
    sys.modules["general_motion_retargeting"] = package

    from general_motion_retargeting.motion_retarget import GeneralMotionRetargeting
    from general_motion_retargeting.params import IK_CONFIG_DICT, ROBOT_XML_DICT
    from general_motion_retargeting.utils.lafan1 import load_bvh_file

    return GeneralMotionRetargeting, IK_CONFIG_DICT, ROBOT_XML_DICT, load_bvh_file


def _prepare_floating_urdf(source: Path, destination: Path) -> None:
    tree = ET.parse(source)  # noqa: S314 - caller explicitly selects a local robot asset.
    robot = tree.getroot()

    if any(link.attrib.get("name") == "world" for link in robot.findall("link")):
        raise ValueError("Source URDF already contains a world link")

    compiler_parent = ET.Element("mujoco")
    ET.SubElement(
        compiler_parent,
        "compiler",
        {
            "balanceinertia": "true",
            "discardvisual": "false",
            "fusestatic": "false",
            "strippath": "false",
        },
    )
    robot.insert(0, compiler_parent)

    # The generated file lives in a temporary directory, so make mesh paths
    # absolute before moving it away from the training URDF.
    for mesh in robot.iter("mesh"):
        filename = mesh.attrib.get("filename")
        if filename:
            mesh.attrib["filename"] = str((source.parent / filename).resolve())

    world = ET.Element("link", {"name": "world"})
    floating = ET.Element("joint", {"name": "floating_base_joint", "type": "floating"})
    ET.SubElement(floating, "parent", {"link": "world"})
    ET.SubElement(floating, "child", {"link": "pelvis_link"})
    robot.insert(1, world)
    robot.insert(2, floating)
    tree.write(destination, encoding="utf-8", xml_declaration=True)


def _prepare_config(source: Path, destination: Path | None, drop_fixed_head_target: bool) -> Path:
    with source.open("r", encoding="utf-8") as file:
        config = json.load(file)

    if not drop_fixed_head_target:
        return source

    config["ik_match_table1"].pop("head_yaw_Link", None)
    config["ik_match_table2"].pop("head_yaw_Link", None)
    config["human_scale_table"].pop("Head", None)
    assert destination is not None
    destination.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return destination


def _read_bvh_fps(path: Path) -> float:
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip().lower().startswith("frame time:"):
                frame_time = float(line.split(":", maxsplit=1)[1].strip())
                if frame_time <= 0.0:
                    break
                return 1.0 / frame_time
    raise ValueError(f"Could not read a positive Frame Time from {path}")


def _read_joint_velocity_limits(path: Path) -> dict[str, float]:
    root = ET.parse(path).getroot()  # noqa: S314 - caller explicitly selects a local robot asset.
    limits = {}
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        if joint.attrib.get("type") in {"revolute", "continuous"} and limit is not None:
            if "velocity" in limit.attrib:
                limits[joint.attrib["name"]] = float(limit.attrib["velocity"])
    return limits


def _git_revision(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _quat_angle_error(first: np.ndarray, second: np.ndarray) -> float:
    first = first / np.linalg.norm(first)
    second = second / np.linalg.norm(second)
    return float(2.0 * np.arccos(np.clip(abs(np.dot(first, second)), 0.0, 1.0)))


def _retarget(
    bvh_path: Path,
    output_path: Path,
    config_path: Path,
    urdf_path: Path,
    gmr_root: Path,
    start_frame: int,
    end_frame: int | None,
    max_frames: int | None,
    warmup_iterations: int,
    solver: str,
    enforce_velocity_limits: bool,
    drop_fixed_head_target: bool,
    force: bool,
) -> None:
    if output_path.exists() and not force:
        raise FileExistsError(f"Output already exists: {output_path}. Pass --force to replace it.")

    GeneralMotionRetargeting, ik_configs, robot_models, load_bvh_file = _load_gmr(gmr_root)
    frames, actual_human_height = load_bvh_file(str(bvh_path), format="lafan1")
    fps = _read_bvh_fps(bvh_path)

    stop = len(frames) if end_frame is None else min(end_frame, len(frames))
    if max_frames is not None:
        stop = min(stop, start_frame + max_frames)
    if start_frame < 0 or stop <= start_frame:
        raise ValueError(f"Invalid frame range [{start_frame}, {stop}) for {len(frames)} BVH frames")
    if warmup_iterations < 0:
        raise ValueError("--warmup-iterations must be non-negative")
    frames = frames[start_frame:stop]

    root_link, urdf_joints = _parse_urdf(urdf_path)
    velocity_limits = _read_joint_velocity_limits(urdf_path)
    active_joints = {joint.name: joint for joint in urdf_joints if joint.joint_type in {"revolute", "continuous"}}
    if set(active_joints) != set(A3_DOF_NAMES):
        missing = sorted(set(A3_DOF_NAMES) - set(active_joints))
        extra = sorted(set(active_joints) - set(A3_DOF_NAMES))
        raise ValueError(f"A3 model must expose exactly 29 joints; missing={missing}, extra={extra}")

    with tempfile.TemporaryDirectory(prefix="a3-gmr-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        floating_urdf = temp_dir / "a3_t3d0_29dof_floating.urdf"
        temporary_config = temp_dir / "bvh_lafan1_to_a3_t3d0_no_head.json"
        _prepare_floating_urdf(urdf_path, floating_urdf)
        active_config = _prepare_config(config_path, temporary_config, drop_fixed_head_target)

        robot_models[ROBOT_KEY] = floating_urdf
        ik_configs.setdefault("bvh_lafan1", {})[ROBOT_KEY] = active_config
        retargeter = GeneralMotionRetargeting(
            src_human="bvh_lafan1",
            tgt_robot=ROBOT_KEY,
            actual_human_height=actual_human_height,
            solver=solver,
            verbose=False,
            use_velocity_limit=False,
        )

        model = retargeter.model
        if model.nq != 7 + len(A3_DOF_NAMES):
            raise ValueError(f"Expected floating base + 29 joints (nq=36), got nq={model.nq}, nv={model.nv}")

        if enforce_velocity_limits:
            import mink

            missing_velocity_limits = sorted(set(A3_DOF_NAMES) - set(velocity_limits))
            if missing_velocity_limits:
                raise ValueError(f"URDF is missing velocity limits for {missing_velocity_limits}")
            model.opt.timestep = 1.0 / fps
            retargeter.ik_limits.append(
                mink.VelocityLimit(model, {name: velocity_limits[name] for name in A3_DOF_NAMES})
            )
            # One rate-limited IK update per BVH frame. Multiple inner updates
            # would each consume a full frame's velocity allowance.
            retargeter.max_iter = 0

        qpos_addresses: dict[str, int] = {}
        for name in A3_DOF_NAMES:
            joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"GMR model is missing joint {name}")
            qpos_addresses[name] = int(model.jnt_qposadr[joint_id])

        qpos_frames = []
        task_errors = []
        position_errors: dict[str, list[float]] = defaultdict(list)
        rotation_errors: dict[str, list[float]] = defaultdict(list)

        # GMR starts from MuJoCo's zero pose. Re-solving the first target before
        # recording avoids turning initialization convergence into a one-frame
        # joint-velocity spike in the training motion.
        for _ in range(warmup_iterations):
            retargeter.retarget(frames[0])

        for frame_index, human_frame in enumerate(frames):
            qpos = retargeter.retarget(human_frame)
            if not np.all(np.isfinite(qpos)):
                raise FloatingPointError(f"GMR returned NaN/Inf at source frame {start_frame + frame_index}")
            qpos_frames.append(qpos)
            task_errors.append(float(retargeter.error1()))

            mj.mj_forward(model, retargeter.configuration.data)
            for robot_body, entry in retargeter.ik_match_table1.items():
                human_body, pos_weight, rot_weight, _, _ = entry
                body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, robot_body)
                target_pos, target_quat = retargeter.scaled_human_data[human_body]
                if pos_weight:
                    position_errors[robot_body].append(
                        float(np.linalg.norm(retargeter.configuration.data.xpos[body_id] - target_pos))
                    )
                if rot_weight:
                    rotation_errors[robot_body].append(
                        _quat_angle_error(retargeter.configuration.data.xquat[body_id], target_quat)
                    )

        qpos_array = np.asarray(qpos_frames, dtype=np.float64)
        root_pos = qpos_array[:, :3].copy()
        root_quat_wxyz = qpos_array[:, 3:7].copy()
        root_quat_wxyz /= np.linalg.norm(root_quat_wxyz, axis=1, keepdims=True)

        joint_positions: dict[str, np.ndarray] = {}
        limit_violations = 0
        max_limit_violation = 0.0
        for name in A3_DOF_NAMES:
            values = qpos_array[:, qpos_addresses[name]]
            joint = active_joints[name]
            violation = np.maximum(joint.lower - values, values - joint.upper)
            positive_violation = np.maximum(violation, 0.0)
            limit_violations += int(np.count_nonzero(positive_violation > 1e-7))
            max_limit_violation = max(max_limit_violation, float(positive_violation.max(initial=0.0)))
            joint_positions[name] = np.clip(values, joint.lower, joint.upper)

    positions, rotations = _forward_kinematics(root_link, urdf_joints, root_pos, root_quat_wxyz, joint_positions)
    body_pos_w = np.stack([positions[name] for name in A3_BODY_NAMES], axis=1)
    body_quat_w = np.stack([_matrix_to_quat_wxyz(rotations[name]) for name in A3_BODY_NAMES], axis=1)
    dt = 1.0 / fps
    body_lin_vel_w = _differentiate(body_pos_w, dt)
    body_ang_vel_w = _angular_velocity(body_quat_w, dt)
    ordered_joint_pos = np.stack([joint_positions[name] for name in A3_DOF_NAMES], axis=1)
    ordered_joint_vel = _differentiate(ordered_joint_pos, dt)
    velocity_excess = {}
    velocity_violation_samples = 0
    for index, name in enumerate(A3_DOF_NAMES):
        limit = velocity_limits.get(name)
        if limit is None or limit <= 0.0:
            continue
        absolute_velocity = np.abs(ordered_joint_vel[:, index])
        count = int(np.count_nonzero(absolute_velocity > limit + 1e-6))
        velocity_violation_samples += count
        if count:
            velocity_excess[name] = {
                "samples": count,
                "limit_rad_s": limit,
                "max_rad_s": float(absolute_velocity.max()),
                "max_ratio": float(absolute_velocity.max() / limit),
            }
    root_lin_vel = _differentiate(root_pos, dt)
    root_ang_vel = body_ang_vel_w[:, A3_BODY_NAMES.index(root_link)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        fps=np.asarray([round(fps)], dtype=np.int64),
        joint_pos=np.concatenate([root_pos, root_quat_wxyz, ordered_joint_pos], axis=1).astype(np.float32),
        joint_vel=np.concatenate([root_lin_vel, root_ang_vel, ordered_joint_vel], axis=1).astype(np.float32),
        body_pos_w=body_pos_w.astype(np.float32),
        body_quat_w=body_quat_w.astype(np.float32),
        body_lin_vel_w=body_lin_vel_w.astype(np.float32),
        body_ang_vel_w=body_ang_vel_w.astype(np.float32),
        joint_names=np.asarray(A3_DOF_NAMES),
        body_names=np.asarray(A3_BODY_NAMES),
    )

    ankle_indices = [A3_BODY_NAMES.index("left_ankle_roll_Link"), A3_BODY_NAMES.index("right_ankle_roll_Link")]
    ankle_z = body_pos_w[:, ankle_indices, 2]
    report = {
        "source_bvh": bvh_path.name,
        "source_frame_range": [start_frame, stop],
        "output": output_path.name,
        "config": (
            str(config_path.relative_to(REPO_ROOT)) if config_path.is_relative_to(REPO_ROOT) else config_path.name
        ),
        "gmr_revision": _git_revision(gmr_root),
        "fps": fps,
        "frames": len(frames),
        "controlled_dofs": len(A3_DOF_NAMES),
        "neck_dofs": 0,
        "fixed_head_target_dropped": drop_fixed_head_target,
        "velocity_limits_enforced_in_ik": enforce_velocity_limits,
        "warmup_iterations": warmup_iterations,
        "human_height_reported_by_gmr": actual_human_height,
        "task_error": {"mean": float(np.mean(task_errors)), "max": float(np.max(task_errors))},
        "joint_limits": {"violations": limit_violations, "max_violation_rad": max_limit_violation},
        "joint_velocity_limits": {
            "violation_samples": velocity_violation_samples,
            "joints": velocity_excess,
        },
        "root_z_m": {"min": float(root_pos[:, 2].min()), "max": float(root_pos[:, 2].max())},
        "ankle_z_m": {"min": float(ankle_z.min()), "median": float(np.median(ankle_z)), "max": float(ankle_z.max())},
        "position_error_m": {
            name: {"mean": float(np.mean(values)), "max": float(np.max(values))}
            for name, values in position_errors.items()
        },
        "rotation_error_rad": {
            name: {"mean": float(np.mean(values)), "max": float(np.max(values))}
            for name, values in rotation_errors.items()
        },
    }
    report_path = output_path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote {output_path}")
    print(f"Wrote {report_path}")
    print(
        f"frames={len(frames)}, fps={fps:.3f}, controlled_dofs=29, neck_dofs=0, "
        f"limit_violations={limit_violations}, task_error_mean={np.mean(task_errors):.5f}"
    )
    print(
        f"root_z=[{root_pos[:, 2].min():.3f}, {root_pos[:, 2].max():.3f}] m, "
        f"ankle_z=[{ankle_z.min():.3f}, {ankle_z.max():.3f}] m"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bvh", type=Path, required=True, help="LAFAN1 BVH file")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Holosoma WBT NPZ output")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="GMR A3 IK JSON")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF, help="Fixed-neck A3 29-DoF URDF")
    parser.add_argument(
        "--gmr-root",
        type=Path,
        default=Path(os.environ["GMR_ROOT"]) if "GMR_ROOT" in os.environ else None,
        required="GMR_ROOT" not in os.environ,
        help="GMR checkout root (or set GMR_ROOT)",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--max-frames", type=int, help="Retarget at most this many frames")
    parser.add_argument("--warmup-iterations", type=int, default=30)
    parser.add_argument("--solver", default="daqp", help="Mink/qpsolvers solver")
    parser.add_argument(
        "--enforce-velocity-limits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply the A3 URDF joint velocity limits during each GMR frame",
    )
    parser.add_argument(
        "--drop-fixed-head-target",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove the Head IK target because both A3 neck joints are fixed",
    )
    parser.add_argument("--force", action="store_true", help="Replace existing output/report")
    args = parser.parse_args()

    _retarget(
        bvh_path=args.bvh.resolve(),
        output_path=args.output.resolve(),
        config_path=args.config.resolve(),
        urdf_path=args.urdf.resolve(),
        gmr_root=args.gmr_root.resolve(),
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        max_frames=args.max_frames,
        warmup_iterations=args.warmup_iterations,
        solver=args.solver,
        enforce_velocity_limits=args.enforce_velocity_limits,
        drop_fixed_head_target=args.drop_fixed_head_target,
        force=args.force,
    )


if __name__ == "__main__":
    main()
