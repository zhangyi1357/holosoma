"""Evaluate a HoloSoma checkpoint in Isaac Sim and record its rollout.

This small wrapper configures video and trajectory recording programmatically.
It avoids Tyro having to rebuild the full saved experiment schema when only
evaluation-output settings need to change.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path

import imageio_ffmpeg

from holosoma.config_types.eval_callback import (
    EvalCallbacksConfig,
    RecordingCallbackConfig,
    RecordingConfig,
)
from holosoma.eval_agent import run_eval_with_tyro
from holosoma.utils.eval_utils import (
    CheckpointConfig,
    init_eval_logging,
    load_saved_experiment_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument(
        "--native-video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record Isaac Sim camera clips in addition to the trajectory NPZ.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    video_dir = output_dir / "isaacsim_videos"
    eval_log_root = output_dir / "eval_logs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # HoloSoma's shared recorder invokes ``ffmpeg`` by name.  The Isaac Sim
    # environment already ships imageio-ffmpeg, so expose that binary without
    # requiring a separate system-wide FFmpeg installation.
    ffmpeg_exe = Path(imageio_ffmpeg.get_ffmpeg_exe())
    ffmpeg_alias_dir = output_dir / "ffmpeg_bin"
    ffmpeg_alias_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_alias = ffmpeg_alias_dir / "ffmpeg.exe"
    if not ffmpeg_alias.exists():
        try:
            ffmpeg_alias.hardlink_to(ffmpeg_exe)
        except OSError:
            import shutil

            shutil.copy2(ffmpeg_exe, ffmpeg_alias)
    os.environ["PATH"] = str(ffmpeg_alias_dir) + os.pathsep + os.environ.get("PATH", "")

    init_eval_logging()
    checkpoint_cfg = CheckpointConfig(checkpoint=str(checkpoint))
    saved_cfg, saved_wandb_path = load_saved_experiment_config(checkpoint_cfg)
    eval_cfg = saved_cfg.get_eval_config()

    camera_cfg = replace(eval_cfg.logger.video.camera, tracking_body_name="pelvis_link")
    video_cfg = replace(
        eval_cfg.logger.video,
        enabled=args.native_video,
        interval=1,
        width=args.width,
        height=args.height,
        playback_rate=1.0,
        output_format="h264",
        save_dir=str(video_dir),
        upload_to_wandb=False,
        show_command_overlay=False,
        use_recording_thread=False,
        camera=camera_cfg,
    )
    logger_cfg = replace(
        eval_cfg.logger,
        video=video_cfg,
        headless_recording=args.native_video,
        base_dir=str(eval_log_root),
    )
    training_cfg = replace(
        eval_cfg.training,
        headless=True,
        num_envs=1,
        max_eval_steps=args.steps,
        export_onnx=False,
    )
    eval_cfg = replace(eval_cfg, logger=logger_cfg, training=training_cfg)

    recording_cfg = EvalCallbacksConfig(
        recording=RecordingCallbackConfig(
            config=RecordingConfig(
                enabled=True,
                output_path=str(output_dir / "eval_recording.npz"),
                env_id=0,
            )
        )
    )

    run_eval_with_tyro(
        eval_cfg,
        checkpoint_cfg,
        saved_cfg,
        saved_wandb_path,
        eval_cbs_cfg=recording_cfg,
    )


if __name__ == "__main__":
    main()
