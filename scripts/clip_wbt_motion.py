#!/usr/bin/env python3
"""Extract an exact time window from a Holosoma WBT NPZ motion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from prepare_a3_t3d0_wbt_motion import _angular_velocity, _differentiate


def _sample_indexes(source_frames: int, source_fps: float, target_frames: int, target_fps: float):
    source_positions = np.arange(target_frames, dtype=np.float64) * source_fps / target_fps
    lower = np.floor(source_positions).astype(np.int64)
    lower = np.clip(lower, 0, source_frames - 1)
    upper = np.minimum(lower + 1, source_frames - 1)
    alpha = source_positions - lower
    return lower, upper, alpha


def _interpolate(values: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    weight_shape = (len(alpha),) + (1,) * (values.ndim - 1)
    weight = alpha.reshape(weight_shape)
    return values[lower] + weight * (values[upper] - values[lower])


def _slerp(values: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    q0 = np.asarray(values[lower], dtype=np.float64)
    q1 = np.asarray(values[upper], dtype=np.float64)
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)

    alpha_shape = (len(alpha),) + (1,) * (values.ndim - 1)
    weight = alpha.reshape(alpha_shape)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    linear = (1.0 - weight) * q0 + weight * q1
    spherical = np.divide(
        np.sin((1.0 - weight) * theta) * q0 + np.sin(weight * theta) * q1,
        sin_theta,
        out=linear.copy(),
        where=sin_theta > 1e-8,
    )
    result = np.where(dot > 0.9995, linear, spherical)
    return result / np.linalg.norm(result, axis=-1, keepdims=True)


def _resample_motion(arrays: dict[str, np.ndarray], source_fps: float, target_fps: float) -> dict[str, np.ndarray]:
    source_frames = len(arrays["joint_pos"])
    target_frames = round(source_frames * target_fps / source_fps)
    lower, upper, alpha = _sample_indexes(source_frames, source_fps, target_frames, target_fps)

    output: dict[str, np.ndarray] = {}
    for name, values in arrays.items():
        if values.ndim == 0 or len(values) != source_frames:
            output[name] = values
        elif name == "body_quat_w":
            output[name] = _slerp(values, lower, upper, alpha).astype(values.dtype)
        elif name == "joint_pos":
            sampled = _interpolate(values, lower, upper, alpha)
            sampled[:, 3:7] = _slerp(values[:, 3:7], lower, upper, alpha)
            output[name] = sampled.astype(values.dtype)
        else:
            output[name] = _interpolate(values, lower, upper, alpha).astype(values.dtype)

    dt = 1.0 / target_fps
    joint_pos = np.asarray(output["joint_pos"], dtype=np.float64)
    body_pos = np.asarray(output["body_pos_w"], dtype=np.float64)
    body_quat = np.asarray(output["body_quat_w"], dtype=np.float64)
    body_lin_vel = _differentiate(body_pos, dt)
    body_ang_vel = _angular_velocity(body_quat, dt)
    output["joint_vel"] = np.concatenate(
        [
            _differentiate(joint_pos[:, :3], dt),
            body_ang_vel[:, 0],
            _differentiate(joint_pos[:, 7:], dt),
        ],
        axis=1,
    ).astype(np.float32)
    output["body_lin_vel_w"] = body_lin_vel.astype(np.float32)
    output["body_ang_vel_w"] = body_ang_vel.astype(np.float32)
    output["fps"] = np.asarray([round(target_fps)], dtype=np.int64)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--duration-seconds", type=float, required=True)
    parser.add_argument(
        "--output-fps",
        type=float,
        help="Resample the extracted clip to this frame rate, including quaternion SLERP and recomputed velocities.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source = args.input.resolve()
    destination = args.output.resolve()
    if destination.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {destination}. Pass --force to replace it.")
    if args.start_seconds < 0.0 or args.duration_seconds <= 0.0:
        parser.error("start must be non-negative and duration must be positive")

    with np.load(source) as motion:
        fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
        total_frames = len(motion["joint_pos"])
        start_frame = round(args.start_seconds * fps)
        requested_frames = round(args.duration_seconds * fps)
        stop_frame = start_frame + requested_frames
        if start_frame < 0 or stop_frame > total_frames:
            raise ValueError(
                f"Requested [{start_frame}, {stop_frame}) but source contains {total_frames} frames at {fps:g} FPS"
            )

        output_arrays: dict[str, np.ndarray] = {}
        for name in motion.files:
            array = np.asarray(motion[name])
            output_arrays[name] = (
                array[start_frame:stop_frame] if array.ndim > 0 and len(array) == total_frames else array
            )

    output_fps = fps if args.output_fps is None else args.output_fps
    if output_fps <= 0.0:
        parser.error("output FPS must be positive")
    if not np.isclose(output_fps, round(output_fps)):
        parser.error("output FPS must be an integer because the WBT motion schema stores FPS as an integer")
    if not np.isclose(output_fps, fps):
        output_arrays = _resample_motion(output_arrays, fps, output_fps)

    output_frames = len(output_arrays["joint_pos"])

    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **output_arrays)
    metadata = {
        "source": source.name,
        "output": destination.name,
        "source_fps": fps,
        "fps": output_fps,
        "source_frames": total_frames,
        "frame_range": [start_frame, stop_frame],
        "source_clip_frames": requested_frames,
        "frames": output_frames,
        "start_seconds": start_frame / fps,
        "duration_seconds": output_frames / output_fps,
    }
    metadata_path = destination.with_suffix(".clip.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {destination}")
    print(f"Wrote {metadata_path}")
    print(f"frames={output_frames}, fps={output_fps:g}, duration={output_frames / output_fps:.2f}s")


if __name__ == "__main__":
    main()
