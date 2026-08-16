#!/usr/bin/env python3
"""Render exact A3 states recorded during an Isaac Sim evaluation to MP4."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import imageio.v2 as imageio
import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from prepare_a3_t3d0_wbt_motion import A3_DOF_NAMES, DEFAULT_URDF
from render_a3_wbt_motion import _add_scene
from retarget_lafan1_to_a3_gmr import _prepare_floating_urdf


def render_recording(
    recording_path: Path,
    output_path: Path,
    urdf_path: Path,
    width: int,
    height: int,
    camera_distance: float,
    camera_azimuth: float,
    camera_elevation: float,
    snapshot_path: Path | None,
    force: bool,
) -> None:
    if output_path.exists() and not force:
        raise FileExistsError(f"Output already exists: {output_path}. Pass --force to replace it.")

    with np.load(recording_path) as recording:
        metadata = json.loads(str(recording["_metadata_json"]))
        fps = float(metadata["fps"])
        dof_names = list(metadata["dof_names"])
        dof_pos = np.asarray(recording["dof_pos"], dtype=np.float64)
        root_pos = np.asarray(recording["root_pos"], dtype=np.float64)
        root_quat_xyzw = np.asarray(recording["root_quat_xyzw"], dtype=np.float64)

    if dof_names != A3_DOF_NAMES:
        raise ValueError("Recording joint order does not match the A3 T3D0 29-DoF policy order")
    if not (len(dof_pos) == len(root_pos) == len(root_quat_xyzw)):
        raise ValueError("Recording channels have inconsistent frame counts")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if snapshot_path is not None:
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="a3-eval-render-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        floating_urdf = temp_dir / "a3_t3d0_floating.urdf"
        compiled_mjcf = temp_dir / "a3_t3d0_compiled.xml"
        scene_mjcf = temp_dir / "a3_t3d0_scene.xml"
        _prepare_floating_urdf(urdf_path, floating_urdf)
        source_model = mj.MjModel.from_xml_path(str(floating_urdf))
        mj.mj_saveLastXML(str(compiled_mjcf), source_model)
        _add_scene(compiled_mjcf, scene_mjcf, width, height)

        model = mj.MjModel.from_xml_path(str(scene_mjcf))
        data = mj.MjData(model)
        qpos_addresses: list[int] = []
        for name in dof_names:
            joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"Rendered model is missing joint {name}")
            qpos_addresses.append(int(model.jnt_qposadr[joint_id]))

        renderer = mj.Renderer(model, height=height, width=width)
        camera = mj.MjvCamera()
        mj.mjv_defaultCamera(camera)
        camera.type = mj.mjtCamera.mjCAMERA_FREE
        camera.distance = camera_distance
        camera.azimuth = camera_azimuth
        camera.elevation = camera_elevation

        scene_option = mj.MjvOption()
        mj.mjv_defaultOption(scene_option)
        scene_option.geomgroup[:] = 0
        scene_option.geomgroup[1] = 1
        scene_option.geomgroup[2] = 1

        smoothed_lookat = root_pos[0] + np.asarray([0.0, 0.0, 0.05])
        writer = imageio.get_writer(
            output_path,
            fps=fps,
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
            macro_block_size=None,
        )
        try:
            label_font = ImageFont.truetype("arial.ttf", 18)
        except OSError:
            label_font = ImageFont.load_default()
        try:
            for frame_index in range(len(dof_pos)):
                data.qpos[:3] = root_pos[frame_index]
                quat_xyzw = root_quat_xyzw[frame_index]
                data.qpos[3:7] = quat_xyzw[[3, 0, 1, 2]]
                for motion_index, qpos_address in enumerate(qpos_addresses):
                    data.qpos[qpos_address] = dof_pos[frame_index, motion_index]
                mj.mj_forward(model, data)

                target_lookat = root_pos[frame_index] + np.asarray([0.0, 0.0, 0.05])
                # Snap the camera across evaluation resets, then smooth ordinary motion.
                if np.linalg.norm(target_lookat - smoothed_lookat) > 0.75:
                    smoothed_lookat = target_lookat
                else:
                    smoothed_lookat = 0.9 * smoothed_lookat + 0.1 * target_lookat
                camera.lookat[:] = smoothed_lookat
                renderer.update_scene(data, camera=camera, scene_option=scene_option)
                pixels = renderer.render()

                label = f"A3 FastSAC checkpoint 300 | Isaac Sim eval replay | t={frame_index / fps:05.2f}s"
                labeled_image = Image.fromarray(pixels)
                draw = ImageDraw.Draw(labeled_image)
                draw.rectangle((0, 0, width, 38), fill=(12, 15, 20))
                draw.text((16, 9), label, fill=(235, 235, 235), font=label_font)
                pixels = np.asarray(labeled_image)

                if frame_index == 0 and snapshot_path is not None:
                    imageio.imwrite(snapshot_path, pixels)
                writer.append_data(pixels)
                if (frame_index + 1) % 300 == 0:
                    print(f"Rendered {frame_index + 1}/{len(dof_pos)} frames")
        finally:
            writer.close()
            renderer.close()

    print(f"Wrote {output_path}")
    if snapshot_path is not None:
        print(f"Wrote {snapshot_path}")
    print(f"frames={len(dof_pos)}, fps={fps:g}, duration={len(dof_pos) / fps:.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera-distance", type=float, default=3.0)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-12.0)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    render_recording(
        recording_path=args.recording.resolve(),
        output_path=args.output.resolve(),
        urdf_path=args.urdf.resolve(),
        width=args.width,
        height=args.height,
        camera_distance=args.camera_distance,
        camera_azimuth=args.camera_azimuth,
        camera_elevation=args.camera_elevation,
        snapshot_path=args.snapshot.resolve() if args.snapshot else None,
        force=args.force,
    )


if __name__ == "__main__":
    main()
