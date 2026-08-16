#!/usr/bin/env python3
"""Extract an exact time window from a Holosoma WBT NPZ motion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--duration-seconds", type=float, required=True)
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

        output_arrays = {}
        for name in motion.files:
            array = np.asarray(motion[name])
            output_arrays[name] = (
                array[start_frame:stop_frame] if array.ndim > 0 and len(array) == total_frames else array
            )

    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **output_arrays)
    metadata = {
        "source": source.name,
        "output": destination.name,
        "fps": fps,
        "source_frames": total_frames,
        "frame_range": [start_frame, stop_frame],
        "frames": requested_frames,
        "start_seconds": start_frame / fps,
        "duration_seconds": requested_frames / fps,
    }
    metadata_path = destination.with_suffix(".clip.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {destination}")
    print(f"Wrote {metadata_path}")
    print(f"frames={requested_frames}, fps={fps:g}, duration={requested_frames / fps:.2f}s")


if __name__ == "__main__":
    main()
