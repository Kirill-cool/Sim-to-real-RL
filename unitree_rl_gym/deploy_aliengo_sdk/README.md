# Aliengo Read-Only State Debug (Python)

This stage adds a **read-only** telemetry tool for Unitree Aliengo.

- Script: `deploy_aliengo_sdk/scripts/state_debug.py`
- Architecture: `Python -> robot_interface -> libunitree_legged_sdk.so -> UDP -> Aliengo`
- Important: this script **does not send commands** (`udp.SetSend` / `udp.Send` are not used).

## Environment

```bash
conda activate unitree-rl
```

Check import:

```bash
python -c "import robot_interface; print(robot_interface)"
```

## Run

From `isaacgym/unitree_rl_gym`:

```bash
python deploy_aliengo_sdk/scripts/state_debug.py --state-level low
```

Low-level mode (`LowState`: motorState / IMU / footForce):

```bash
python deploy_aliengo_sdk/scripts/state_debug.py \
  --state-level low \
  --robot-ip 192.168.123.10 \
  --local-port 8082 \
  --target-port 8007 \
  --max-steps 1000 \
  --print-every 50 \
  --sleep 0.002
```

High-level mode (`HighState`: velocity / position / yawSpeed):

```bash
python deploy_aliengo_sdk/scripts/state_debug.py \
  --state-level high \
  --robot-ip 192.168.123.220 \
  --local-port 8081 \
  --target-port 8082 \
  --max-steps 1000 \
  --print-every 50 \
  --sleep 0.002
```

Notes:
- `low` is for joint telemetry (`motorState`) + IMU + contact-like foot forces.
- `high` is for body-level telemetry (`HighState.velocity`, `position`, `yawSpeed`).
- `HighState.velocity` must not be assumed to be a direct `base_lin_vel` replacement until frame/sign are validated on the real robot.

## Safety note

The script is read-only, but run it only in a correctly configured Aliengo network environment.

## Next stage

After successful telemetry readout, the next recommended step is `stand_test.py` (not included here).

## Policy Deploy (Python)

Script:

```bash
python deploy_aliengo_sdk/scripts/deploy_policy.py --policy-path <path_to_policy.pt>
```

Read-only preview (no policy command send):

```bash
python deploy_aliengo_sdk/scripts/deploy_policy.py --policy-path <path_to_policy.pt>
```

Stand-only before SPACE (policy still not sent):

```bash
python deploy_aliengo_sdk/scripts/deploy_policy.py --policy-path <path_to_policy.pt> --send-stand
```

Allow policy send only after SPACE:

```bash
python deploy_aliengo_sdk/scripts/deploy_policy.py \
  --policy-path <path_to_policy.pt> \
  --send-stand \
  --send-policy \
  --action-scale-real 0.05
```

Safety notes:
- Policy commands start only after pressing `SPACE`.
- Without `--send-policy`, policy commands are never sent (preview only).
- If `--base-lin-vel-source highstate` is used, validate velocity frame/sign on robot before walking.

## Aliengo Combined / UPESI Inference

Script:

```bash
python deploy_aliengo_sdk/scripts/deploy_combined_policy.py --policy-path <combined_policy.pt> --alpha-source zero
```

Read-only preview with zero alpha (safe fallback):

```bash
python deploy_aliengo_sdk/scripts/deploy_combined_policy.py \
  --policy-path <combined_policy.pt> \
  --alpha-source zero
```

Stand before SPACE, no policy send:

```bash
python deploy_aliengo_sdk/scripts/deploy_combined_policy.py \
  --policy-path <combined_policy.pt> \
  --alpha-source zero \
  --send-stand
```

Policy send after SPACE with fixed alpha:

```bash
python deploy_aliengo_sdk/scripts/deploy_combined_policy.py \
  --policy-path <combined_policy.pt> \
  --alpha-source fixed \
  --alpha-path <alpha.pt> \
  --send-stand \
  --send-policy \
  --action-scale-real 0.05
```

Notes and warnings:
- `zero` alpha is only a fallback and may not match trained adaptation.
- Prefer `fixed` or `online-file` alpha when `online_alpha.pt` (or equivalent adapted package alpha) is available.
- `--base-lin-vel-source highstate` requires frame/sign validation before using for walking.
- Do not enable `--send-policy` until joint mapping, stand pose, and safe test conditions are validated.
- Commands never switch to policy control before SPACE; without `--send-policy` the script stays in preview-only policy mode.
