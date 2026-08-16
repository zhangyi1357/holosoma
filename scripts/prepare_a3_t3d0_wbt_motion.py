#!/usr/bin/env python3
"""Rebuild a 29-DoF Holosoma motion clip with A3 T3D0 kinematics.

The A3 and G1 presets share the same 29 controlled joint names, but their link
geometry is different.  This tool keeps the root orientation and joint motion,
clips it to A3 limits, and recomputes every A3 body pose/velocity with URDF
forward kinematics.  It is a lightweight compatibility conversion, not a full
motion-retargeting optimizer.
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    REPO_ROOT
    / "src"
    / "holosoma"
    / "holosoma"
    / "data"
    / "motions"
    / "g1_29dof"
    / "whole_body_tracking"
    / "sub3_largebox_003_mj.npz"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "src"
    / "holosoma"
    / "holosoma"
    / "data"
    / "motions"
    / "a3_t3d0"
    / "whole_body_tracking"
    / "sub3_largebox_003_a3_fk.npz"
)
DEFAULT_URDF = (
    REPO_ROOT / "src" / "holosoma" / "holosoma" / "data" / "robots" / "a3_t3d0" / "urdf" / "a3_t3d0_29dof.urdf"
)

A3_DOF_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

A3_BODY_NAMES = [
    "pelvis_link",
    "left_hip_pitch_Link",
    "left_hip_roll_Link",
    "left_hip_yaw_Link",
    "left_knee_Link",
    "left_ankle_pitch_Link",
    "left_ankle_roll_Link",
    "right_hip_pitch_Link",
    "right_hip_roll_Link",
    "right_hip_yaw_Link",
    "right_knee_Link",
    "right_ankle_pitch_Link",
    "right_ankle_roll_Link",
    "waist_yaw_Link",
    "waist_roll_Link",
    "torso_Link",
    "left_shoulder_pitch_Link",
    "left_shoulder_roll_Link",
    "left_shoulder_yaw_Link",
    "left_elbow_Link",
    "left_wrist_roll_Link",
    "left_wrist_pitch_Link",
    "left_wrist_yaw_Link",
    "right_shoulder_pitch_Link",
    "right_shoulder_roll_Link",
    "right_shoulder_yaw_Link",
    "right_elbow_Link",
    "right_wrist_roll_Link",
    "right_wrist_pitch_Link",
    "right_wrist_yaw_Link",
]

REQUIRED_KEYS = {
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
    "body_names",
    "joint_names",
}


@dataclass(frozen=True)
class Joint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz: np.ndarray
    origin_rotation: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float


def _vector(text: str | None, default: str) -> np.ndarray:
    return np.asarray([float(value) for value in (text or default).split()], dtype=np.float64)


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(quat, -1, 0)
    output = np.empty(quat.shape[:-1] + (3, 3), dtype=np.float64)
    output[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    output[..., 0, 1] = 2.0 * (x * y - z * w)
    output[..., 0, 2] = 2.0 * (x * z + y * w)
    output[..., 1, 0] = 2.0 * (x * y + z * w)
    output[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    output[..., 1, 2] = 2.0 * (y * z - x * w)
    output[..., 2, 0] = 2.0 * (x * z - y * w)
    output[..., 2, 1] = 2.0 * (y * z + x * w)
    output[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return output


def _matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Convert one or more rotation matrices to normalized wxyz quaternions."""
    flat = matrix.reshape((-1, 3, 3))
    quaternions = np.empty((flat.shape[0], 4), dtype=np.float64)
    for index, rotation in enumerate(flat):
        trace = np.trace(rotation)
        if trace > 0.0:
            scale = 2.0 * np.sqrt(trace + 1.0)
            quat = np.asarray(
                [
                    0.25 * scale,
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                ]
            )
        else:
            diagonal = np.diag(rotation)
            axis = int(np.argmax(diagonal))
            if axis == 0:
                scale = 2.0 * np.sqrt(max(0.0, 1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]))
                quat = np.asarray(
                    [
                        (rotation[2, 1] - rotation[1, 2]) / scale,
                        0.25 * scale,
                        (rotation[0, 1] + rotation[1, 0]) / scale,
                        (rotation[0, 2] + rotation[2, 0]) / scale,
                    ]
                )
            elif axis == 1:
                scale = 2.0 * np.sqrt(max(0.0, 1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]))
                quat = np.asarray(
                    [
                        (rotation[0, 2] - rotation[2, 0]) / scale,
                        (rotation[0, 1] + rotation[1, 0]) / scale,
                        0.25 * scale,
                        (rotation[1, 2] + rotation[2, 1]) / scale,
                    ]
                )
            else:
                scale = 2.0 * np.sqrt(max(0.0, 1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]))
                quat = np.asarray(
                    [
                        (rotation[1, 0] - rotation[0, 1]) / scale,
                        (rotation[0, 2] + rotation[2, 0]) / scale,
                        (rotation[1, 2] + rotation[2, 1]) / scale,
                        0.25 * scale,
                    ]
                )
        quaternions[index] = quat / np.linalg.norm(quat)
    return quaternions.reshape(matrix.shape[:-2] + (4,))


def _axis_angle_matrices(axis: np.ndarray, angles: np.ndarray) -> np.ndarray:
    x, y, z = axis
    skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    identity = np.eye(3)
    return (
        identity[None, :, :]
        + np.sin(angles)[:, None, None] * skew[None, :, :]
        + (1.0 - np.cos(angles))[:, None, None] * (skew @ skew)[None, :, :]
    )


def _parse_urdf(path: Path) -> tuple[str, list[Joint]]:
    # The input is a local robot asset explicitly selected by the caller.
    root = ET.parse(path).getroot()  # noqa: S314
    links = {element.attrib["name"] for element in root.findall("link")}
    child_links: set[str] = set()
    joints: list[Joint] = []

    for element in root.findall("joint"):
        parent = element.find("parent").attrib["link"]
        child = element.find("child").attrib["link"]
        child_links.add(child)
        origin = element.find("origin")
        xyz = _vector(origin.attrib.get("xyz") if origin is not None else None, "0 0 0")
        rpy = _vector(origin.attrib.get("rpy") if origin is not None else None, "0 0 0")
        axis_element = element.find("axis")
        axis = _vector(axis_element.attrib.get("xyz") if axis_element is not None else None, "1 0 0")
        axis_norm = np.linalg.norm(axis)
        if axis_norm:
            axis = axis / axis_norm
        limit = element.find("limit")
        lower = float(limit.attrib["lower"]) if limit is not None and "lower" in limit.attrib else 0.0
        upper = float(limit.attrib["upper"]) if limit is not None and "upper" in limit.attrib else 0.0
        joints.append(
            Joint(
                name=element.attrib["name"],
                joint_type=element.attrib["type"],
                parent=parent,
                child=child,
                origin_xyz=xyz,
                origin_rotation=_rpy_matrix(rpy),
                axis=axis,
                lower=lower,
                upper=upper,
            )
        )

    roots = links - child_links
    if len(roots) != 1:
        raise ValueError(f"Expected one URDF root link, found {sorted(roots)}")
    root_link = next(iter(roots))

    # Put joints in parent-before-child order so FK is a single pass.
    ordered: list[Joint] = []
    available = {root_link}
    pending = joints.copy()
    while pending:
        ready = [joint for joint in pending if joint.parent in available]
        if not ready:
            raise ValueError("URDF joint graph is disconnected or cyclic")
        for joint in ready:
            ordered.append(joint)
            available.add(joint.child)
            pending.remove(joint)
    return root_link, ordered


def _forward_kinematics(
    root_link: str,
    joints: list[Joint],
    root_pos: np.ndarray,
    root_quat_wxyz: np.ndarray,
    joint_positions: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    frame_count = root_pos.shape[0]
    rotations: dict[str, np.ndarray] = {root_link: _quat_wxyz_to_matrix(root_quat_wxyz)}
    positions: dict[str, np.ndarray] = {root_link: root_pos.copy()}

    for joint in joints:
        parent_rotation = rotations[joint.parent]
        parent_position = positions[joint.parent]
        origin_rotation = np.broadcast_to(joint.origin_rotation, (frame_count, 3, 3))
        joint_frame_rotation = parent_rotation @ origin_rotation
        positions[joint.child] = parent_position + np.einsum("nij,j->ni", parent_rotation, joint.origin_xyz)

        if joint.joint_type in {"revolute", "continuous"}:
            angles = joint_positions[joint.name]
            motion_rotation = _axis_angle_matrices(joint.axis, angles)
            rotations[joint.child] = joint_frame_rotation @ motion_rotation
        elif joint.joint_type == "fixed":
            rotations[joint.child] = joint_frame_rotation
        else:
            raise ValueError(f"Unsupported joint type '{joint.joint_type}' for {joint.name}")
    return positions, rotations


def _differentiate(values: np.ndarray, dt: float) -> np.ndarray:
    edge_order = 2 if values.shape[0] > 2 else 1
    return np.gradient(values, dt, axis=0, edge_order=edge_order)


def _angular_velocity(quat_wxyz: np.ndarray, dt: float) -> np.ndarray:
    frame_count, body_count, _ = quat_wxyz.shape
    output = np.zeros((frame_count, body_count, 3), dtype=np.float64)
    if frame_count < 2:
        return output
    for body_index in range(body_count):
        rotations = _quat_wxyz_to_matrix(quat_wxyz[:, body_index])
        delta = rotations[1:] @ np.swapaxes(rotations[:-1], -1, -2)
        delta_quat = _matrix_to_quat_wxyz(delta)
        delta_quat[delta_quat[:, 0] < 0.0] *= -1.0
        vector = delta_quat[:, 1:]
        vector_norm = np.linalg.norm(vector, axis=1)
        angle = 2.0 * np.arctan2(vector_norm, np.clip(delta_quat[:, 0], 0.0, 1.0))
        scale = np.divide(angle, vector_norm, out=np.full_like(angle, 2.0), where=vector_norm > 1e-12)
        step_velocity = vector * scale[:, None] / dt
        output[0, body_index] = step_velocity[0]
        output[-1, body_index] = step_velocity[-1]
        if frame_count > 2:
            output[1:-1, body_index] = 0.5 * (step_velocity[:-1] + step_velocity[1:])
    return output


def _source_ankle_indices(body_names: list[str]) -> list[int]:
    normalized = {name.lower(): index for index, name in enumerate(body_names)}
    names = ["left_ankle_roll_link", "right_ankle_roll_link"]
    return [normalized[name] for name in names if name in normalized]


def convert(input_path: Path, output_path: Path, urdf_path: Path, align_feet: bool, force: bool) -> None:
    if output_path.exists() and not force:
        raise FileExistsError(f"Output already exists: {output_path}. Pass --force to replace it.")

    root_link, joints = _parse_urdf(urdf_path)
    active_joints = {joint.name: joint for joint in joints if joint.joint_type in {"revolute", "continuous"}}
    if set(active_joints) != set(A3_DOF_NAMES):
        missing = sorted(set(A3_DOF_NAMES) - set(active_joints))
        extra = sorted(set(active_joints) - set(A3_DOF_NAMES))
        raise ValueError(f"A3 URDF must expose exactly 29 controlled joints; missing={missing}, extra={extra}")

    with np.load(input_path) as source:
        missing_keys = REQUIRED_KEYS - set(source.files)
        if missing_keys:
            raise ValueError(f"Input is not a Holosoma motion file; missing keys: {sorted(missing_keys)}")
        if "object_pos_w" in source.files:
            raise ValueError("Object-interaction clips are not supported by this robot-only A3 converter")

        source_joint_names = source["joint_names"].tolist()
        source_body_names = source["body_names"].tolist()
        source_joint_pos = np.asarray(source["joint_pos"], dtype=np.float64)
        if source_joint_pos.shape[1] != len(source_joint_names) + 7:
            raise ValueError("joint_pos must contain xyz+wxyz root columns followed by named joints")
        if not set(A3_DOF_NAMES).issubset(source_joint_names):
            missing = sorted(set(A3_DOF_NAMES) - set(source_joint_names))
            raise ValueError(f"Input motion is missing A3 joints: {missing}")

        fps = float(np.asarray(source["fps"]).reshape(-1)[0])
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        dt = 1.0 / fps

        source_angles = source_joint_pos[:, 7:]
        joint_positions: dict[str, np.ndarray] = {}
        clipped_count = 0
        for name in A3_DOF_NAMES:
            joint = active_joints[name]
            values = source_angles[:, source_joint_names.index(name)]
            clipped = np.clip(values, joint.lower, joint.upper)
            clipped_count += int(np.count_nonzero(np.abs(values - clipped) > 1e-9))
            joint_positions[name] = clipped

        root_pos = source_joint_pos[:, :3].copy()
        root_quat_wxyz = source_joint_pos[:, 3:7].copy()
        quat_norm = np.linalg.norm(root_quat_wxyz, axis=1, keepdims=True)
        if np.any(quat_norm < 1e-8):
            raise ValueError("Input contains a zero-length root quaternion")
        root_quat_wxyz /= quat_norm

        positions, rotations = _forward_kinematics(
            root_link,
            joints,
            root_pos,
            root_quat_wxyz,
            joint_positions,
        )

        body_pos_w = np.stack([positions[name] for name in A3_BODY_NAMES], axis=1)
        if align_feet:
            source_ankles = _source_ankle_indices(source_body_names)
            if len(source_ankles) != 2:
                raise ValueError("--align-feet requires left/right ankle-roll bodies in the input motion")
            source_ankle_z = np.asarray(source["body_pos_w"], dtype=np.float64)[:, source_ankles, 2].mean(axis=1)
            target_ankle_indices = [
                A3_BODY_NAMES.index("left_ankle_roll_Link"),
                A3_BODY_NAMES.index("right_ankle_roll_Link"),
            ]
            target_ankle_z = body_pos_w[:, target_ankle_indices, 2].mean(axis=1)
            z_offset = source_ankle_z - target_ankle_z
            root_pos[:, 2] += z_offset
            body_pos_w[:, :, 2] += z_offset[:, None]

        body_quat_wxyz = np.stack([_matrix_to_quat_wxyz(rotations[name]) for name in A3_BODY_NAMES], axis=1)
        body_lin_vel_w = _differentiate(body_pos_w, dt)
        body_ang_vel_w = _angular_velocity(body_quat_wxyz, dt)

        ordered_joint_pos = np.stack([joint_positions[name] for name in A3_DOF_NAMES], axis=1)
        ordered_joint_vel = _differentiate(ordered_joint_pos, dt)
        root_lin_vel = _differentiate(root_pos, dt)
        root_ang_vel = body_ang_vel_w[:, A3_BODY_NAMES.index(root_link)]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            fps=np.asarray([round(fps)], dtype=np.int64),
            joint_pos=np.concatenate([root_pos, root_quat_wxyz, ordered_joint_pos], axis=1).astype(np.float32),
            joint_vel=np.concatenate([root_lin_vel, root_ang_vel, ordered_joint_vel], axis=1).astype(np.float32),
            body_pos_w=body_pos_w.astype(np.float32),
            body_quat_w=body_quat_wxyz.astype(np.float32),
            body_lin_vel_w=body_lin_vel_w.astype(np.float32),
            body_ang_vel_w=body_ang_vel_w.astype(np.float32),
            joint_names=np.asarray(A3_DOF_NAMES),
            body_names=np.asarray(A3_BODY_NAMES),
        )

    print(f"Wrote {output_path}")
    print(
        f"frames={body_pos_w.shape[0]}, fps={fps:g}, "
        f"controlled_dofs={len(A3_DOF_NAMES)}, clipped_values={clipped_count}"
    )
    print(f"root_z=[{root_pos[:, 2].min():.3f}, {root_pos[:, 2].max():.3f}] m")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Source Holosoma 29-DoF NPZ")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output A3-compatible NPZ")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF, help="A3 29-DoF training URDF")
    parser.add_argument(
        "--no-align-feet",
        action="store_true",
        help="Keep source root height instead of matching source ankle height",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing output file")
    args = parser.parse_args()
    convert(args.input.resolve(), args.output.resolve(), args.urdf.resolve(), not args.no_align_feet, args.force)


if __name__ == "__main__":
    main()
