#!/usr/bin/env python3
"""Render a bundled G1 Holosoma WBT motion, including its optional object."""

from __future__ import annotations

import argparse
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import imageio.v2 as imageio
import mujoco as mj
import numpy as np
from render_a3_wbt_motion import _add_scene

REPO_ROOT = Path(__file__).resolve().parents[1]
MOTION_DIR = (
    REPO_ROOT / "src" / "holosoma" / "holosoma" / "data" / "motions" / "g1_29dof" / "whole_body_tracking"
)
DEFAULT_MOTION = MOTION_DIR / "sub3_largebox_003_mj_w_obj.npz"
DEFAULT_OUTPUT = MOTION_DIR / "sub3_largebox_003_mj_w_obj.mp4"
DEFAULT_G1_MODEL = (
    REPO_ROOT
    / "src"
    / "holosoma_retargeting"
    / "holosoma_retargeting"
    / "models"
    / "g1"
    / "g1_29dof_w_largebox.xml"
)


def _resolve_compiled_assets(xml_path: Path, source_dir: Path) -> None:
    """Keep assets loadable after moving MuJoCo's compiled XML to a temp dir."""
    tree = ET.parse(xml_path)  # noqa: S314 - trusted bundled MuJoCo model
    root = tree.getroot()
    compiler = root.find("compiler")
    mesh_dir = source_dir / (compiler.get("meshdir", "") if compiler is not None else "")
    texture_dir = source_dir / (compiler.get("texturedir", "") if compiler is not None else "")
    for asset in root.findall("./asset/*"):
        asset_file = asset.get("file")
        if not asset_file:
            continue
        asset_dir = mesh_dir if asset.tag == "mesh" else texture_dir if asset.tag == "texture" else source_dir
        resolved = (asset_dir / asset_file).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Compiled MuJoCo asset does not exist: {resolved}")
        asset.set("file", resolved.as_posix())
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)


def render(
    motion_path: Path,
    model_path: Path,
    output_path: Path,
    width: int,
    height: int,
    camera_distance: float,
    camera_azimuth: float,
    camera_elevation: float,
    force: bool,
) -> None:
    if output_path.exists() and not force:
        raise FileExistsError(f"Output already exists: {output_path}. Pass --force to replace it.")

    with np.load(motion_path) as motion:
        required = {"fps", "joint_pos", "joint_names"}
        missing = required - set(motion.files)
        if missing:
            raise ValueError(f"Motion is missing {sorted(missing)}")
        fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
        joint_pos = np.asarray(motion["joint_pos"], dtype=np.float64)
        joint_names = motion["joint_names"].tolist()
        object_pos = np.asarray(motion["object_pos_w"], dtype=np.float64) if "object_pos_w" in motion else None
        object_quat = np.asarray(motion["object_quat_w"], dtype=np.float64) if "object_quat_w" in motion else None

    if joint_pos.shape[1] != 7 + len(joint_names):
        raise ValueError("joint_pos must contain xyz+wxyz root columns followed by named joints")
    if object_pos is not None and (object_quat is None or len(object_pos) != len(joint_pos)):
        raise ValueError("Object pose arrays do not match the robot motion")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="g1-render-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        compiled_mjcf = temp_dir / "g1_compiled.xml"
        scene_mjcf = temp_dir / "g1_scene.xml"
        source_model = mj.MjModel.from_xml_path(str(model_path))
        mj.mj_saveLastXML(str(compiled_mjcf), source_model)
        _resolve_compiled_assets(compiled_mjcf, model_path.parent)
        _add_scene(compiled_mjcf, scene_mjcf, width, height)

        model = mj.MjModel.from_xml_path(str(scene_mjcf))
        data = mj.MjData(model)
        qpos_addresses = []
        for name in joint_names:
            joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"G1 model is missing joint {name}")
            qpos_addresses.append(int(model.jnt_qposadr[joint_id]))

        object_qpos_address = None
        if object_pos is not None:
            object_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "largebox_link")
            if object_body_id < 0:
                raise ValueError("Object motion requires largebox_link in the MuJoCo model")
            object_joint_id = int(model.body_jntadr[object_body_id])
            object_qpos_address = int(model.jnt_qposadr[object_joint_id])
            model.geom_group[model.geom_bodyid == object_body_id] = 1

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

        smoothed_lookat = joint_pos[0, :3] + np.asarray([0.0, 0.0, 0.05])
        writer = imageio.get_writer(
            output_path,
            fps=fps,
            codec="libx264",
            quality=7,
            pixelformat="yuv420p",
            macro_block_size=None,
        )
        try:
            for index, frame in enumerate(joint_pos):
                data.qpos[:7] = frame[:7]
                for motion_index, qpos_address in enumerate(qpos_addresses):
                    data.qpos[qpos_address] = frame[7 + motion_index]
                if object_qpos_address is not None:
                    data.qpos[object_qpos_address : object_qpos_address + 3] = object_pos[index]
                    data.qpos[object_qpos_address + 3 : object_qpos_address + 7] = object_quat[index]
                mj.mj_forward(model, data)

                target_lookat = frame[:3] + np.asarray([0.0, 0.0, 0.05])
                smoothed_lookat = 0.9 * smoothed_lookat + 0.1 * target_lookat
                camera.lookat[:] = smoothed_lookat
                renderer.update_scene(data, camera=camera, scene_option=scene_option)
                writer.append_data(renderer.render())
        finally:
            writer.close()
            renderer.close()

    print(f"Wrote {output_path}")
    print(
        f"frames={len(joint_pos)}, fps={fps:g}, duration={len(joint_pos) / fps:.2f}s, "
        f"object={object_pos is not None}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--model", type=Path, default=DEFAULT_G1_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera-distance", type=float, default=3.0)
    parser.add_argument("--camera-azimuth", type=float, default=225.0)
    parser.add_argument("--camera-elevation", type=float, default=-12.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    render(
        motion_path=args.motion.resolve(),
        model_path=args.model.resolve(),
        output_path=args.output.resolve(),
        width=args.width,
        height=args.height,
        camera_distance=args.camera_distance,
        camera_azimuth=args.camera_azimuth,
        camera_elevation=args.camera_elevation,
        force=args.force,
    )


if __name__ == "__main__":
    main()
