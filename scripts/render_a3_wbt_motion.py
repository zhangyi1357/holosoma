#!/usr/bin/env python3
"""Render an A3 Holosoma whole-body-tracking NPZ to MP4 with MuJoCo."""

from __future__ import annotations

import argparse
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import imageio.v2 as imageio
import mujoco as mj
import numpy as np
from prepare_a3_t3d0_wbt_motion import A3_DOF_NAMES, DEFAULT_URDF
from retarget_lafan1_to_a3_gmr import _prepare_floating_urdf

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MOTION = (
    REPO_ROOT
    / "src"
    / "holosoma"
    / "holosoma"
    / "data"
    / "motions"
    / "a3_t3d0"
    / "whole_body_tracking"
    / "dance2_subject1_gmr_velocity_limited.npz"
)
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "gmr_lafan1" / "dance2_subject1_a3_preview.mp4"


def _add_scene(source_mjcf: Path, destination_mjcf: Path, width: int, height: int) -> None:
    tree = ET.parse(source_mjcf)  # noqa: S314 - generated local MuJoCo model.
    root = tree.getroot()
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    global_visual = visual.find("global")
    if global_visual is None:
        global_visual = ET.SubElement(visual, "global")
    global_visual.attrib.update({"offwidth": str(width), "offheight": str(height)})

    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        {
            "name": "preview_sky",
            "type": "skybox",
            "builtin": "gradient",
            "rgb1": "0.22 0.27 0.34",
            "rgb2": "0.035 0.045 0.065",
            "width": "512",
            "height": "3072",
        },
    )
    ET.SubElement(
        asset,
        "texture",
        {
            "name": "preview_floor_tex",
            "type": "2d",
            "builtin": "checker",
            "rgb1": "0.16 0.18 0.21",
            "rgb2": "0.08 0.09 0.11",
            "width": "512",
            "height": "512",
        },
    )
    ET.SubElement(
        asset,
        "material",
        {
            "name": "preview_floor_mat",
            "texture": "preview_floor_tex",
            "texrepeat": "12 12",
            "reflectance": "0.12",
        },
    )

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Generated MJCF has no worldbody")
    worldbody.insert(
        0,
        ET.Element(
            "geom",
            {
                "name": "preview_floor",
                "type": "plane",
                "size": "20 20 0.1",
                "material": "preview_floor_mat",
                "group": "2",
                "contype": "0",
                "conaffinity": "0",
            },
        ),
    )
    worldbody.insert(
        1,
        ET.Element(
            "light",
            {
                "name": "preview_key",
                "directional": "true",
                "pos": "-2 -3 6",
                "dir": "0.3 0.4 -1",
                "diffuse": "0.9 0.9 0.9",
                "specular": "0.25 0.25 0.25",
            },
        ),
    )
    worldbody.insert(
        2,
        ET.Element(
            "light",
            {
                "name": "preview_fill",
                "directional": "true",
                "pos": "3 2 4",
                "dir": "-0.4 -0.2 -1",
                "diffuse": "0.45 0.5 0.6",
                "specular": "0.1 0.1 0.1",
            },
        ),
    )
    tree.write(destination_mjcf, encoding="utf-8", xml_declaration=True)


def render(
    motion_path: Path,
    urdf_path: Path,
    output_path: Path,
    start_frame: int,
    duration_seconds: float | None,
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

    with np.load(motion_path) as motion:
        fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
        joint_pos = np.asarray(motion["joint_pos"], dtype=np.float64)
        joint_names = motion["joint_names"].tolist()

    if joint_pos.shape[1] != 7 + len(joint_names):
        raise ValueError("joint_pos must contain xyz+wxyz root columns followed by named joints")
    if joint_names != A3_DOF_NAMES:
        raise ValueError("Motion joint order does not match the A3 T3D0 training order")
    if fps <= 0.0:
        raise ValueError(f"Invalid motion fps: {fps}")

    stop_frame = len(joint_pos)
    if duration_seconds is not None:
        stop_frame = min(stop_frame, start_frame + round(duration_seconds * fps))
    if start_frame < 0 or stop_frame <= start_frame:
        raise ValueError(f"Invalid frame range [{start_frame}, {stop_frame})")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if snapshot_path is not None:
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="a3-render-") as temp_dir_name:
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
        qpos_addresses = []
        for name in joint_names:
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
        scene_option.geomgroup[1] = 1  # URDF visual meshes.
        scene_option.geomgroup[2] = 1  # Preview floor.

        smoothed_lookat = joint_pos[start_frame, :3] + np.asarray([0.0, 0.0, 0.05])
        writer = imageio.get_writer(
            output_path,
            fps=fps,
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
            macro_block_size=None,
        )
        try:
            for output_index, source_index in enumerate(range(start_frame, stop_frame)):
                frame = joint_pos[source_index]
                data.qpos[:7] = frame[:7]
                for motion_index, qpos_address in enumerate(qpos_addresses):
                    data.qpos[qpos_address] = frame[7 + motion_index]
                mj.mj_forward(model, data)

                target_lookat = frame[:3] + np.asarray([0.0, 0.0, 0.05])
                smoothed_lookat = 0.9 * smoothed_lookat + 0.1 * target_lookat
                camera.lookat[:] = smoothed_lookat
                renderer.update_scene(data, camera=camera, scene_option=scene_option)
                pixels = renderer.render()
                if output_index == 0 and snapshot_path is not None:
                    imageio.imwrite(snapshot_path, pixels)
                writer.append_data(pixels)

                if (output_index + 1) % 300 == 0:
                    print(f"Rendered {output_index + 1}/{stop_frame - start_frame} frames")
        finally:
            writer.close()
            renderer.close()

    print(f"Wrote {output_path}")
    if snapshot_path is not None:
        print(f"Wrote {snapshot_path}")
    print(f"frames={stop_frame - start_frame}, fps={fps:g}, duration={(stop_frame - start_frame) / fps:.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-frame", type=int, default=900)
    parser.add_argument("--duration-seconds", type=float, default=30.0)
    parser.add_argument(
        "--full-motion",
        action="store_true",
        help="Render from --start-frame through the final motion frame",
    )
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera-distance", type=float, default=3.0)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-12.0)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    render(
        motion_path=args.motion.resolve(),
        urdf_path=args.urdf.resolve(),
        output_path=args.output.resolve(),
        start_frame=args.start_frame,
        duration_seconds=None if args.full_motion else args.duration_seconds,
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
