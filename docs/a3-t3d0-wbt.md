# A3 T3D0 FastSAC Whole-Body Tracking

This integration provides a focused training path for the Agibot A3 T3D0:

- Isaac Sim / Isaac Lab only
- FastSAC only
- robot-only whole-body tracking (no object interaction)
- 29 policy-controlled joints
- fixed `head_yaw_joint` and `head_pitch_joint`

The robot meshes and base URDF come from
[`AgibotTech/A3-A3U-robot-model`](https://github.com/AgibotTech/A3-A3U-robot-model) and remain under
Mulan PSL v2. The license copy is bundled at
`src/holosoma/holosoma/data/robots/a3_t3d0/LICENSE.txt`.

## Setup

On Ubuntu 22.04 or newer, use an NVIDIA GPU supported by Isaac Sim 5.1:

```bash
bash scripts/setup_isaacsim.sh
source scripts/source_isaacsim_setup.sh
```

The first launch converts the A3 URDF to a rank-local USD cache under the robot asset directory.

The verified Windows environment for this checkout is:

- Python: `%USERPROFILE%\.holosoma_deps\miniconda3\envs\hssim\python.exe`
- Isaac Sim: 5.1.0
- Isaac Lab: 2.3.0 (`%USERPROFILE%\.holosoma_deps\IsaacLab`)
- PyTorch: 2.7.0 + CUDA 12.8

On Windows, launch a small run from PowerShell with:

```powershell
.\demo_scripts\demo_a3_t3d0_wbt_fast_sac.ps1 -NumEnvs 1024 -Seed 1
```

The Windows launcher accepts any remaining HoloSoma arguments. For example, append `logger:wandb` to enable online
logging. FastSAC automatically uses eager CUDA mode on Windows because the official PyTorch CUDA wheel does not include
a usable Triton compiler; Linux retains `torch.compile` acceleration.

## Check the motion before training

The repository includes a 30-second `walk1 subject1` clip retargeted from LAFAN1
to A3 with GMR. It is also the task's default motion. Inspect it in Isaac Sim:

```bash
python src/holosoma/holosoma/replay.py \
  exp:a3-t3d0-wbt-fast-sac \
  --training.headless=False \
  --training.num-envs=1
```

## Start FastSAC training

Direct command:

```bash
python src/holosoma/holosoma/train_agent.py \
  exp:a3-t3d0-wbt-fast-sac \
  logger:disabled \
  --training.seed=1
```

Or use the launcher:

```bash
NUM_ENVS=4096 SEED=1 bash demo_scripts/demo_a3_t3d0_wbt_fast_sac.sh logger:disabled
```

For a smaller GPU, begin with `NUM_ENVS=1024` or `--training.num-envs=1024`. FastSAC keeps one environment step per
collection iteration and performs four updates, matching HoloSoma's fast WBT baseline.

TensorBoard event files are always written to the run directory. W&B is
optional; `logger:disabled` disables W&B without disabling TensorBoard. Start a
local dashboard with:

```bash
tensorboard --logdir logs/A3WholeBodyTracking --port 6006
```

## Prepare another motion

An input clip must use Holosoma's NPZ schema and contain the same 29 named leg, waist, and arm joints. Convert it with:

```bash
python scripts/prepare_a3_t3d0_wbt_motion.py \
  --input=/absolute/path/source_motion.npz \
  --output=/absolute/path/a3_motion.npz
```

The converter:

1. reorders the input by joint name;
2. clips joint positions to the official A3 limits;
3. fixes both neck joints at zero;
4. recomputes all 30 simulated A3 body transforms and velocities from the URDF;
5. shifts the root vertically so the two ankle heights match the source clip.

This is a fast kinematic compatibility conversion, not an optimization-based
retargeter. For human-source motions, use the GMR path below instead. Use this
tool only when the input already has matching A3 joint semantics and needs
final schema/kinematics preparation.

Select the converted file at launch:

```bash
python src/holosoma/holosoma/train_agent.py \
  exp:a3-t3d0-wbt-fast-sac \
  --command.setup-terms.motion-command.params.motion-config.motion-file=/absolute/path/a3_motion.npz
```

## Retarget LAFAN1 BVH with GMR

The A3 integration also includes a headless GMR path for LAFAN1 BVH input:

- IK config: `configs/retargeting/gmr/bvh_lafan1_to_a3_t3d0.json`
- converter: `scripts/retarget_lafan1_to_a3_gmr.py`
- bundled example: the 30-second `walk1_subject1_0s_30s.npz` training clip

Use a checkout of [GMR](https://github.com/YanjieZe/GMR) and install its offline IK dependencies in a separate Python
environment. The converter imports only the headless GMR modules, so Torch, SMPL-X, and the MuJoCo viewer are not
required for BVH conversion.

```powershell
$env:GMR_ROOT = "C:\path\to\GMR"
python scripts\retarget_lafan1_to_a3_gmr.py `
  --bvh C:\path\to\lafan1\walk1_subject1.bvh
```

The safe defaults remove the `Head -> head_yaw_Link` IK task because both neck joints are fixed, run 30 warm-up IK
updates on the first frame, and enforce every A3 URDF joint velocity limit. Use
`--no-drop-fixed-head-target` or `--no-enforce-velocity-limits` only for comparison experiments. Each conversion writes
a sibling `.report.json` containing task residuals, limit violations, root/ankle height ranges, the GMR revision, and the
exact source frame range.

The supplied numeric parameters can produce an unconstrained trajectory, but
the complete validation run found joint-velocity spikes when limits were not
enforced. The validated velocity-limited `walk1_subject1` output has no
position or velocity limit violations. Only its 30-second training clip is
bundled:

```powershell
python src\holosoma\holosoma\train_agent.py `
  exp:a3-t3d0-wbt-fast-sac `
  --command.setup-terms.motion-command.params.motion-config.motion-file=C:\absolute\path\walk1_subject1_0s_30s.npz
```

## Training choices and limitations

- The policy action, joint position, and joint velocity dimensions are all 29. The head remains attached to the torso as
  fixed geometry and never appears in observations or actions.
- A3's official limits, effort limits, velocity limits, and MJCF joint friction values are used. The PD baseline follows
  [HOPE's A3 configuration](https://github.com/hitchopen/HOPE/blob/main/hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/robots/agibot_a3.py):
  Kp uses its deploy values, while training Kd is the deploy/message Kd plus MJCF passive viscous damping. HoloSoma's
  explicit position controller has zero desired velocity and no separate passive-viscous term, making that summed Kd
  the equivalent training model. These gains still require validation before any real-robot deployment.
- Self-collision is disabled. The official model uses detailed collision meshes that overlap around adjacent joints;
  disabling self-collision improves stability and throughput for thousands of parallel environments.
- This preset does not add MuJoCo/MJWarp, object WBT, sim-to-real deployment, or an A3 hardware bridge.
