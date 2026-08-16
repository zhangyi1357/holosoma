#!/usr/bin/env python3
"""Validate a completed LAFAN1-to-A3 motion and preview-video batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from prepare_a3_t3d0_wbt_motion import A3_BODY_NAMES, A3_DOF_NAMES


def _read_bvh_frames(path: Path) -> int:
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.startswith("Frames:"):
                return int(line.split(":", maxsplit=1)[1].strip())
    raise ValueError(f"Missing Frames header: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bvh-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    bvh_dir = args.bvh_dir.resolve()
    output_dir = args.output_dir.resolve()
    sources = {path.stem: path for path in bvh_dir.rglob("*.bvh")}
    issues: list[str] = []
    clip_summaries = []
    total_frames = 0
    total_bytes = 0
    task_error_means = []
    task_error_maxes = []
    velocity_violation_samples = 0
    solver_preclip_joint_limit_violations = 0
    solver_preclip_max_joint_limit_violation_rad = 0.0

    if len(sources) != 77:
        issues.append(f"expected 77 source BVH clips, found {len(sources)}")

    for stem, bvh_path in sorted(sources.items()):
        expected_frames = _read_bvh_frames(bvh_path)
        motion_path = output_dir / f"{stem}.npz"
        report_path = output_dir / f"{stem}.report.json"
        video_path = output_dir / f"{stem}.mp4"
        paths = (motion_path, report_path, video_path)
        issues.extend(
            f"{stem}: missing or empty {path.name}" for path in paths if not path.is_file() or path.stat().st_size == 0
        )
        if any(not path.is_file() or path.stat().st_size == 0 for path in paths):
            continue

        total_bytes += motion_path.stat().st_size + report_path.stat().st_size + video_path.stat().st_size
        with np.load(motion_path) as motion:
            required = {
                "fps",
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
                "joint_names",
                "body_names",
            }
            missing = required - set(motion.files)
            if missing:
                issues.append(f"{stem}: NPZ missing keys {sorted(missing)}")
                continue
            fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
            joint_pos = np.asarray(motion["joint_pos"])
            joint_vel = np.asarray(motion["joint_vel"])
            body_pos = np.asarray(motion["body_pos_w"])
            body_quat = np.asarray(motion["body_quat_w"])
            body_lin_vel = np.asarray(motion["body_lin_vel_w"])
            body_ang_vel = np.asarray(motion["body_ang_vel_w"])
            joint_names = motion["joint_names"].tolist()
            body_names = motion["body_names"].tolist()
            expected_shapes = {
                "joint_pos": (expected_frames, 7 + len(A3_DOF_NAMES)),
                "joint_vel": (expected_frames, 6 + len(A3_DOF_NAMES)),
                "body_pos_w": (expected_frames, len(A3_BODY_NAMES), 3),
                "body_quat_w": (expected_frames, len(A3_BODY_NAMES), 4),
                "body_lin_vel_w": (expected_frames, len(A3_BODY_NAMES), 3),
                "body_ang_vel_w": (expected_frames, len(A3_BODY_NAMES), 3),
            }
            arrays = {
                "joint_pos": joint_pos,
                "joint_vel": joint_vel,
                "body_pos_w": body_pos,
                "body_quat_w": body_quat,
                "body_lin_vel_w": body_lin_vel,
                "body_ang_vel_w": body_ang_vel,
            }
            for name, array in arrays.items():
                if array.shape != expected_shapes[name]:
                    issues.append(f"{stem}: {name} shape {array.shape}, expected {expected_shapes[name]}")
                if not np.all(np.isfinite(array)):
                    issues.append(f"{stem}: {name} contains NaN/Inf")
            if joint_names != A3_DOF_NAMES:
                issues.append(f"{stem}: joint names/order do not match A3 29-DoF training order")
            if body_names != A3_BODY_NAMES:
                issues.append(f"{stem}: body names/order do not match the A3 training asset")
            quat_norm_error = float(np.max(np.abs(np.linalg.norm(joint_pos[:, 3:7], axis=1) - 1.0)))
            if quat_norm_error > 1e-4:
                issues.append(f"{stem}: maximum root quaternion norm error is {quat_norm_error}")

        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("frames") != expected_frames:
            issues.append(f"{stem}: report frames={report.get('frames')}, expected {expected_frames}")
        if report.get("controlled_dofs") != 29 or report.get("neck_dofs") != 0:
            issues.append(f"{stem}: report is not fixed-neck 29-DoF A3")
        if not report.get("velocity_limits_enforced_in_ik"):
            issues.append(f"{stem}: velocity limits were not enforced during IK")
        task_error_means.append(float(report["task_error"]["mean"]))
        task_error_maxes.append(float(report["task_error"]["max"]))
        velocity_violation_samples += int(report["joint_velocity_limits"]["violation_samples"])
        solver_preclip_joint_limit_violations += int(report["joint_limits"]["violations"])
        solver_preclip_max_joint_limit_violation_rad = max(
            solver_preclip_max_joint_limit_violation_rad,
            float(report["joint_limits"]["max_violation_rad"]),
        )

        reader = imageio.get_reader(video_path)
        try:
            video_meta = reader.get_meta_data()
        finally:
            reader.close()
        expected_duration = expected_frames / fps
        if video_meta.get("codec") != "h264":
            issues.append(f"{stem}: video codec={video_meta.get('codec')}, expected h264")
        if tuple(video_meta.get("size", ())) != (960, 720):
            issues.append(f"{stem}: video size={video_meta.get('size')}, expected 960x720")
        if abs(float(video_meta.get("fps", 0.0)) - fps) > 1e-3:
            issues.append(f"{stem}: video fps={video_meta.get('fps')}, expected {fps}")
        if abs(float(video_meta.get("duration", 0.0)) - expected_duration) > max(0.05, 1.5 / fps):
            issues.append(f"{stem}: video duration={video_meta.get('duration')}, expected {expected_duration:.4f}")

        total_frames += expected_frames
        clip_summaries.append(
            {
                "clip": stem,
                "frames": expected_frames,
                "fps": fps,
                "duration_seconds": expected_duration,
                "motion_bytes": motion_path.stat().st_size,
                "video_bytes": video_path.stat().st_size,
            }
        )

    extra_npz = sorted(path.stem for path in output_dir.glob("*.npz") if path.stem not in sources)
    extra_mp4 = sorted(path.stem for path in output_dir.glob("*.mp4") if path.stem not in sources)
    if extra_npz:
        issues.append(f"unexpected NPZ files: {extra_npz}")
    if extra_mp4:
        issues.append(f"unexpected MP4 files: {extra_mp4}")

    summary = {
        "passed": not issues,
        "clips": len(clip_summaries),
        "frames": total_frames,
        "duration_hours": sum(item["duration_seconds"] for item in clip_summaries) / 3600.0,
        "total_bytes": total_bytes,
        "controlled_dofs": 29,
        "neck_dofs": 0,
        "task_error_mean_range": [min(task_error_means), max(task_error_means)] if task_error_means else None,
        "task_error_max": max(task_error_maxes) if task_error_maxes else None,
        "solver_preclip_joint_limit_violations": solver_preclip_joint_limit_violations,
        "solver_preclip_max_joint_limit_violation_rad": solver_preclip_max_joint_limit_violation_rad,
        "joint_velocity_violation_samples": velocity_violation_samples,
        "issues": issues,
        "clip_summaries": clip_summaries,
    }
    validation_path = output_dir / "validation_report.json"
    validation_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "clip_summaries"}, indent=2))
    print(f"Wrote {validation_path}")
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
