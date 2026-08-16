#!/usr/bin/env python3
"""Batch-retarget every LAFAN1 BVH to A3 and render side-by-side MP4 files."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "src" / "holosoma" / "holosoma" / "data" / "motions" / "a3_t3d0" / "whole_body_tracking" / "lafan1"
)
RETARGET_SCRIPT = REPO_ROOT / "scripts" / "retarget_lafan1_to_a3_gmr.py"
RENDER_SCRIPT = REPO_ROOT / "scripts" / "render_a3_wbt_motion.py"


@dataclass(frozen=True)
class Clip:
    bvh: Path
    frames: int
    motion: Path
    report: Path
    video: Path


def _read_frame_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.startswith("Frames:"):
                return int(line.split(":", maxsplit=1)[1].strip())
    raise ValueError(f"Missing Frames header: {path}")


def _discover(bvh_dir: Path, output_dir: Path) -> list[Clip]:
    files = sorted(bvh_dir.rglob("*.bvh"), key=lambda path: path.name.lower())
    stems: dict[str, Path] = {}
    clips = []
    for path in files:
        key = path.stem.casefold()
        if key in stems:
            raise ValueError(f"Duplicate BVH stem: {stems[key]} and {path}")
        stems[key] = path
        motion = output_dir / f"{path.stem}.npz"
        clips.append(
            Clip(
                bvh=path,
                frames=_read_frame_count(path),
                motion=motion,
                report=motion.with_suffix(".report.json"),
                video=motion.with_suffix(".mp4"),
            )
        )
    if not clips:
        raise FileNotFoundError(f"No BVH files found under {bvh_dir}")
    return clips


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    return env


def _run(command: list[str]) -> tuple[float, str]:
    start = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=_subprocess_env(),
        check=False,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - start
    combined = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    if result.returncode != 0:
        raise RuntimeError(f"exit={result.returncode}\n{combined}")
    return elapsed, combined


def _retarget(clip: Clip, gmr_root: Path, force: bool) -> tuple[float, str]:
    if clip.motion.is_file() and clip.report.is_file() and not force:
        return 0.0, "skipped existing motion/report"
    command = [
        sys.executable,
        str(RETARGET_SCRIPT),
        "--bvh",
        str(clip.bvh),
        "--output",
        str(clip.motion),
        "--gmr-root",
        str(gmr_root),
    ]
    if force:
        command.append("--force")
    return _run(command)


def _render(clip: Clip, width: int, height: int, force: bool) -> tuple[float, str]:
    if clip.video.is_file() and not force:
        return 0.0, "skipped existing video"
    if not clip.motion.is_file():
        raise FileNotFoundError(f"Cannot render missing motion: {clip.motion}")
    command = [
        sys.executable,
        str(RENDER_SCRIPT),
        "--motion",
        str(clip.motion),
        "--output",
        str(clip.video),
        "--start-frame",
        "0",
        "--full-motion",
        "--width",
        str(width),
        "--height",
        str(height),
    ]
    if force:
        command.append("--force")
    return _run(command)


def _parallel_stage(name: str, clips: list[Clip], workers: int, function) -> tuple[list[dict], list[dict]]:
    # Longest-first scheduling keeps the final workers from being stuck with all long clips.
    ordered = sorted(clips, key=lambda clip: clip.frames, reverse=True)
    successes: list[dict] = []
    failures: list[dict] = []
    print(f"{name}: {len(ordered)} clips, workers={workers}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_clip = {executor.submit(function, clip): clip for clip in ordered}
        for completed, future in enumerate(as_completed(future_to_clip), start=1):
            clip = future_to_clip[future]
            try:
                elapsed, _ = future.result()
                state = "SKIP" if elapsed == 0.0 else "OK"
                print(
                    f"[{name} {completed:02d}/{len(ordered):02d}] {state} {clip.bvh.stem} "
                    f"frames={clip.frames} seconds={elapsed:.1f}",
                    flush=True,
                )
                successes.append({"clip": clip.bvh.stem, "frames": clip.frames, "seconds": elapsed})
            except Exception as error:  # Continue other independent clips and report every failure.
                print(f"[{name} {completed:02d}/{len(ordered):02d}] FAIL {clip.bvh.stem}: {error}", flush=True)
                failures.append({"clip": clip.bvh.stem, "frames": clip.frames, "error": str(error)})
    return successes, failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bvh-dir", type=Path, required=True)
    parser.add_argument("--gmr-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stage", choices=("all", "retarget", "render"), default="all")
    parser.add_argument("--retarget-workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--render-workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.retarget_workers < 1 or args.render_workers < 1:
        parser.error("worker counts must be positive")
    bvh_dir = args.bvh_dir.resolve()
    gmr_root = args.gmr_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    clips = _discover(bvh_dir, output_dir)
    started_at = time.time()
    manifest: dict[str, object] = {
        "source_dir": bvh_dir.name,
        "gmr_root": gmr_root.name,
        "output_dir": ".",
        "clips": len(clips),
        "frames": sum(clip.frames for clip in clips),
        "retarget": {"successes": [], "failures": []},
        "render": {"successes": [], "failures": []},
    }

    if args.stage in ("all", "retarget"):
        successes, failures = _parallel_stage(
            "retarget",
            clips,
            args.retarget_workers,
            lambda clip: _retarget(clip, gmr_root, args.force),
        )
        manifest["retarget"] = {"successes": successes, "failures": failures}

    render_clips = [clip for clip in clips if clip.motion.is_file()]
    if args.stage in ("all", "render"):
        successes, failures = _parallel_stage(
            "render",
            render_clips,
            args.render_workers,
            lambda clip: _render(clip, args.width, args.height, args.force),
        )
        manifest["render"] = {"successes": successes, "failures": failures}

    manifest["elapsed_seconds"] = time.time() - started_at
    manifest_path = output_dir / "batch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}", flush=True)

    failed = len(manifest["retarget"]["failures"]) + len(manifest["render"]["failures"])
    if failed:
        raise SystemExit(f"Batch completed with {failed} failed jobs; see {manifest_path}")


if __name__ == "__main__":
    main()
