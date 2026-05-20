#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import select
import signal
import sys
import termios
import time
import tty
from typing import Any, Dict, List, Optional, Tuple

LOW_CMD_LENGTH = 610
LOW_STATE_LENGTH = 771
LOWLEVEL = 0xFF
MOTOR_COUNT = 12
OBS_DIM = 48
ACT_DIM = 12

# HighState defaults from official Aliengo sport examples.
HIGHSTATE_IP = "192.168.123.220"
HIGHSTATE_LOCAL_PORT = 8081
HIGHSTATE_TARGET_PORT = 8082
HIGH_CMD_LENGTH = 113
HIGH_STATE_LENGTH = 244

OBS_SCALE_LIN_VEL = 2.0
OBS_SCALE_ANG_VEL = 0.25
OBS_SCALE_DOF_POS = 1.0
OBS_SCALE_DOF_VEL = 0.05
COMMANDS_SCALE = [OBS_SCALE_LIN_VEL, OBS_SCALE_LIN_VEL, OBS_SCALE_ANG_VEL]

# Expected sim order (from URDF traversal in this project).
SIM_DOF_ORDER = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]

# SDK motor order from Unitree SDK constants FR_0..RL_2.
SDK_MOTOR_ORDER = [
    "FR_0", "FR_1", "FR_2",
    "FL_0", "FL_1", "FL_2",
    "RR_0", "RR_1", "RR_2",
    "RL_0", "RL_1", "RL_2",
]

# Mapping from SDK motor slot -> semantic joint name.
SDK_MOTOR_TO_JOINT = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]


class TerminalKeyReader:
    def __init__(self) -> None:
        self.fd = None
        self.old_attrs = None
        self.enabled = False

    def start(self) -> None:
        try:
            self.fd = sys.stdin.fileno()
            if sys.stdin.isatty():
                self.old_attrs = termios.tcgetattr(self.fd)
                tty.setcbreak(self.fd)
                self.enabled = True
        except Exception:
            self.enabled = False

    def stop(self) -> None:
        if self.enabled and self.fd is not None and self.old_attrs is not None:
            try:
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_attrs)
            except Exception:
                pass
        self.enabled = False

    def poll_key(self) -> str:
        if not self.enabled:
            return ""
        try:
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        except Exception:
            return ""
        if not ready:
            return ""
        try:
            return sys.stdin.read(1)
        except Exception:
            return ""


def safe_get(obj: Any, attr: str, default: Any = None) -> Any:
    if obj is None:
        return default
    try:
        return getattr(obj, attr)
    except Exception:
        return default


def safe_index(seq: Any, idx: int, default: Any = None) -> Any:
    if seq is None:
        return default
    try:
        return seq[idx]
    except Exception:
        return default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy Aliengo TorchScript policy via robot_interface")
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--robot-ip", default="192.168.123.10")
    parser.add_argument("--local-port", type=int, default=8082)
    parser.add_argument("--target-port", type=int, default=8007)
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--kp", type=float, default=20.0)
    parser.add_argument("--kd", type=float, default=0.5)
    parser.add_argument("--action-scale-real", type=float, default=0.05)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--stand-duration", type=float, default=3.0)
    parser.add_argument("--send-stand", action="store_true")
    parser.add_argument("--send-policy", action="store_true")
    parser.add_argument("--vx", type=float, default=0.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw-rate", type=float, default=0.0)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--base-lin-vel-source", choices=["zero", "highstate"], default="zero")
    parser.add_argument("--dry-run", action="store_true", help="Alias: do not send policy commands")
    parser.add_argument("--roll-pitch-limit", type=float, default=0.6)
    parser.add_argument("--max-target-delta", type=float, default=0.25)
    return parser.parse_args()


def load_default_joint_angles() -> Dict[str, float]:
    try:
        from legged_gym.envs.aliengo.aliengo_config import AliengoRoughCfg

        return dict(AliengoRoughCfg.init_state.default_joint_angles)
    except Exception:
        # Fallback values from AliengoRoughCfg.
        return {
            "FL_hip_joint": -0.0,
            "FL_thigh_joint": 0.8,
            "FL_calf_joint": -1.5,
            "FR_hip_joint": 0.0,
            "FR_thigh_joint": 0.8,
            "FR_calf_joint": -1.5,
            "RL_hip_joint": -0.0,
            "RL_thigh_joint": 1.0,
            "RL_calf_joint": -1.5,
            "RR_hip_joint": 0.0,
            "RR_thigh_joint": 1.0,
            "RR_calf_joint": -1.5,
        }


def build_joint_mappings() -> Tuple[List[int], List[int], List[float]]:
    sim_name_to_idx = {name: idx for idx, name in enumerate(SIM_DOF_ORDER)}
    sim_to_sdk = [-1] * ACT_DIM
    sdk_to_sim = [-1] * ACT_DIM
    for sdk_idx, joint_name in enumerate(SDK_MOTOR_TO_JOINT):
        sim_idx = sim_name_to_idx.get(joint_name, -1)
        sdk_to_sim[sdk_idx] = sim_idx
        if sim_idx >= 0:
            sim_to_sdk[sim_idx] = sdk_idx

    unresolved = [idx for idx, v in enumerate(sim_to_sdk) if v < 0]
    if unresolved:
        print("WARNING: unresolved SIM->SDK mapping indices:", unresolved)
    else:
        print("SIM->SDK mapping built successfully.")

    print("SIM_DOF_ORDER:", SIM_DOF_ORDER)
    print("SDK_MOTOR_ORDER:", SDK_MOTOR_ORDER)
    print("SIM_TO_SDK:", sim_to_sdk)
    print("SDK_TO_SIM:", sdk_to_sim)
    print("WARNING: verify mapping on robot before aggressive motion.")

    default_by_name = load_default_joint_angles()
    default_sim = [float(default_by_name.get(name, 0.0)) for name in SIM_DOF_ORDER]
    return sim_to_sdk, sdk_to_sim, default_sim


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return torch.tensor([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=torch.float32)


def rotate_inverse_by_quat(quat_wxyz: List[float], vec_xyz: List[float]) -> torch.Tensor:
    q = torch.tensor(quat_wxyz, dtype=torch.float32)
    norm = torch.norm(q)
    if float(norm) < 1e-8:
        return torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
    q = q / norm
    q_conj = torch.tensor([q[0], -q[1], -q[2], -q[3]], dtype=torch.float32)
    v = torch.tensor([0.0, vec_xyz[0], vec_xyz[1], vec_xyz[2]], dtype=torch.float32)
    rotated = quat_mul(quat_mul(q_conj, v), q)
    return rotated[1:]


def lowstate_to_sim_dofs(low_state: Any, sdk_to_sim: List[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pos_sim = torch.zeros((ACT_DIM,), dtype=torch.float32)
    vel_sim = torch.zeros((ACT_DIM,), dtype=torch.float32)
    pos_sdk = torch.zeros((ACT_DIM,), dtype=torch.float32)

    motor_state = safe_get(low_state, "motorState")
    for sdk_idx in range(ACT_DIM):
        m = safe_index(motor_state, sdk_idx)
        q = float(safe_get(m, "q", 0.0) or 0.0)
        dq = float(safe_get(m, "dq", 0.0) or 0.0)
        pos_sdk[sdk_idx] = q
        sim_idx = sdk_to_sim[sdk_idx]
        if sim_idx >= 0:
            pos_sim[sim_idx] = q
            vel_sim[sim_idx] = dq
    return pos_sim, vel_sim, pos_sdk


def build_observation(
    low_state: Any,
    high_state: Optional[Any],
    base_lin_vel_source: str,
    command_vec: torch.Tensor,
    default_dof_pos_sim: torch.Tensor,
    last_action: torch.Tensor,
    sdk_to_sim: List[int],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    imu = safe_get(low_state, "imu")

    gyro = safe_get(imu, "gyroscope", [0.0, 0.0, 0.0])
    base_ang_vel = torch.tensor([
        float(safe_index(gyro, 0, 0.0) or 0.0),
        float(safe_index(gyro, 1, 0.0) or 0.0),
        float(safe_index(gyro, 2, 0.0) or 0.0),
    ], dtype=torch.float32)

    quat = safe_get(imu, "quaternion", [1.0, 0.0, 0.0, 0.0])
    quat_wxyz = [
        float(safe_index(quat, 0, 1.0) or 1.0),
        float(safe_index(quat, 1, 0.0) or 0.0),
        float(safe_index(quat, 2, 0.0) or 0.0),
        float(safe_index(quat, 3, 0.0) or 0.0),
    ]
    projected_gravity = rotate_inverse_by_quat(quat_wxyz, [0.0, 0.0, -1.0])

    if base_lin_vel_source == "highstate" and high_state is not None:
        v = safe_get(high_state, "velocity", [0.0, 0.0, 0.0])
        base_lin_vel = torch.tensor([
            float(safe_index(v, 0, 0.0) or 0.0),
            float(safe_index(v, 1, 0.0) or 0.0),
            float(safe_index(v, 2, 0.0) or 0.0),
        ], dtype=torch.float32)
    else:
        base_lin_vel = torch.zeros((3,), dtype=torch.float32)

    dof_pos_sim, dof_vel_sim, pos_sdk = lowstate_to_sim_dofs(low_state, sdk_to_sim)

    obs = torch.cat([
        base_lin_vel * OBS_SCALE_LIN_VEL,
        base_ang_vel * OBS_SCALE_ANG_VEL,
        projected_gravity,
        command_vec * torch.tensor(COMMANDS_SCALE, dtype=torch.float32),
        (dof_pos_sim - default_dof_pos_sim) * OBS_SCALE_DOF_POS,
        dof_vel_sim * OBS_SCALE_DOF_VEL,
        last_action,
    ], dim=0)
    obs = obs.view(1, -1)
    if obs.shape[1] != OBS_DIM:
        raise RuntimeError(f"Observation dim mismatch: got {obs.shape[1]}, expected {OBS_DIM}")

    diag = {
        "base_lin_vel": base_lin_vel,
        "base_ang_vel": base_ang_vel,
        "projected_gravity": projected_gravity,
        "dof_pos_sim": dof_pos_sim,
        "dof_vel_sim": dof_vel_sim,
        "pos_sdk": pos_sdk,
    }
    return obs, diag


def extract_action(policy_output: Any) -> torch.Tensor:
    if isinstance(policy_output, (list, tuple)):
        if len(policy_output) == 0:
            raise RuntimeError("Empty policy output")
        policy_output = policy_output[0]

    if not isinstance(policy_output, torch.Tensor):
        policy_output = torch.as_tensor(policy_output, dtype=torch.float32)

    action = policy_output.detach().to(dtype=torch.float32, device="cpu")
    if action.ndim == 2:
        action = action[0]
    if action.ndim != 1:
        action = action.view(-1)
    if action.numel() != ACT_DIM:
        raise RuntimeError(f"Policy action dim mismatch: got {action.numel()}, expected {ACT_DIM}")
    action = torch.clamp(action, -1.0, 1.0)
    return action


def sim_targets_to_sdk(target_q_sim: torch.Tensor, sdk_to_sim: List[int]) -> torch.Tensor:
    target_q_sdk = torch.zeros((ACT_DIM,), dtype=torch.float32)
    for sdk_idx in range(ACT_DIM):
        sim_idx = sdk_to_sim[sdk_idx]
        if sim_idx >= 0:
            target_q_sdk[sdk_idx] = target_q_sim[sim_idx]
    return target_q_sdk


def fill_lowcmd(
    cmd: Any,
    current_q_sdk: torch.Tensor,
    target_q_sdk: torch.Tensor,
    kp: float,
    kd: float,
) -> None:
    for sdk_idx in range(ACT_DIM):
        m = cmd.motorCmd[sdk_idx]
        m.q = float(current_q_sdk[sdk_idx])
        m.dq = 0.0
        m.Kp = 0.0
        m.Kd = 0.0
        m.tau = 0.0

    for sdk_idx in range(ACT_DIM):
        m = cmd.motorCmd[sdk_idx]
        m.q = float(target_q_sdk[sdk_idx])
        m.dq = 0.0
        m.Kp = float(kp)
        m.Kd = float(kd)
        m.tau = 0.0


def main() -> int:
    args = parse_args()

    try:
        global torch  # imported lazily so --help works without runtime deps.
        import torch  # type: ignore
    except Exception as import_err:
        print(f"ERROR: failed to import torch: {import_err}")
        return 1

    if args.rate <= 0:
        print("ERROR: --rate must be > 0")
        return 2
    if args.duration <= 0:
        print("ERROR: --duration must be > 0")
        return 2
    if args.print_every <= 0:
        print("ERROR: --print-every must be > 0")
        return 2
    if not os.path.isfile(args.policy_path):
        print(f"ERROR: policy file not found: {args.policy_path}")
        return 2

    send_policy_enabled = bool(args.send_policy) and (not bool(args.dry_run))

    print("=" * 80)
    print("Aliengo deploy_policy.py")
    print(f"policy_path={args.policy_path}")
    print(
        f"UDP low-level: ip={args.robot_ip} local_port={args.local_port} target_port={args.target_port}"
    )
    print(
        f"flags: send_stand={bool(args.send_stand)} send_policy={bool(args.send_policy)} dry_run={bool(args.dry_run)}"
    )
    print(f"effective send_policy_enabled={send_policy_enabled}")
    print("Safety: policy commands start only after SPACE.")
    if args.base_lin_vel_source == "highstate":
        print("WARNING: base_lin_vel_source=highstate requires frame/sign validation before walking.")
    print("=" * 80)

    try:
        import robot_interface as sdk
    except Exception as import_err:
        print(f"ERROR: failed to import robot_interface: {import_err}")
        return 1

    policy = torch.jit.load(args.policy_path, map_location="cpu")
    policy.eval()

    sim_to_sdk, sdk_to_sim, default_sim_list = build_joint_mappings()
    default_dof_pos_sim = torch.tensor(default_sim_list, dtype=torch.float32)

    command_vec = torch.tensor([args.vx, args.vy, args.yaw_rate], dtype=torch.float32)
    last_action = torch.zeros((ACT_DIM,), dtype=torch.float32)

    udp_low = sdk.UDP(args.local_port, args.robot_ip, args.target_port, LOW_CMD_LENGTH, LOW_STATE_LENGTH, -1)
    low_cmd = sdk.LowCmd()
    low_state = sdk.LowState()
    udp_low.InitCmdData(low_cmd)
    low_cmd.levelFlag = LOWLEVEL

    udp_high = None
    high_state = None
    if args.base_lin_vel_source == "highstate":
        try:
            udp_high = sdk.UDP(
                HIGHSTATE_LOCAL_PORT,
                HIGHSTATE_IP,
                HIGHSTATE_TARGET_PORT,
                HIGH_CMD_LENGTH,
                HIGH_STATE_LENGTH,
                -1,
            )
            high_state = sdk.HighState()
            print(
                "HighState UDP initialized "
                f"(ip={HIGHSTATE_IP}, local={HIGHSTATE_LOCAL_PORT}, target={HIGHSTATE_TARGET_PORT})."
            )
        except Exception as e:
            print(f"WARNING: failed to init HighState UDP ({e}); base_lin_vel will fallback to zero.")
            udp_high = None
            high_state = None

    running = True

    def _stop_handler(_signum: int, _frame: Any) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _stop_handler)
    signal.signal(signal.SIGTERM, _stop_handler)

    key_reader = TerminalKeyReader()
    key_reader.start()

    prepare_start = time.time()
    policy_start = None
    mode = "PREPARE"
    step_idx = 0
    dt = 1.0 / float(args.rate)
    stand_phase_expired_printed = False

    print("[PREPARE] Press SPACE to switch to policy mode. Press Ctrl+C to exit.")
    if not args.send_stand:
        print("[PREPARE] send-stand is OFF: no stand commands before SPACE.")
    if not send_policy_enabled:
        print("[POLICY_PREVIEW] send-policy is OFF (or --dry-run ON): policy commands will NOT be sent.")

    try:
        while running:
            loop_start = time.time()
            step_idx += 1

            udp_low.Recv()
            udp_low.GetRecv(low_state)
            if udp_high is not None and high_state is not None:
                udp_high.Recv()
                udp_high.GetRecv(high_state)

            obs, diag = build_observation(
                low_state=low_state,
                high_state=high_state,
                base_lin_vel_source=args.base_lin_vel_source,
                command_vec=command_vec,
                default_dof_pos_sim=default_dof_pos_sim,
                last_action=last_action,
                sdk_to_sim=sdk_to_sim,
            )

            with torch.no_grad():
                policy_out = policy(obs)
            action = extract_action(policy_out)

            # Safety: orientation from IMU rpy.
            imu = safe_get(low_state, "imu")
            rpy = safe_get(imu, "rpy", [0.0, 0.0, 0.0])
            roll = float(safe_index(rpy, 0, 0.0) or 0.0)
            pitch = float(safe_index(rpy, 1, 0.0) or 0.0)
            if abs(roll) > float(args.roll_pitch_limit) or abs(pitch) > float(args.roll_pitch_limit):
                print(
                    "SAFETY STOP: roll/pitch limit exceeded "
                    f"(roll={roll:.3f}, pitch={pitch:.3f}, limit={args.roll_pitch_limit:.3f})."
                )
                break

            current_q_sdk = diag["pos_sdk"]

            # Stand target for PREPARE, policy target for POLICY.
            stand_target_q_sim = default_dof_pos_sim.clone()
            policy_target_q_sim = default_dof_pos_sim + action * float(args.action_scale_real)

            if torch.max(torch.abs(policy_target_q_sim - default_dof_pos_sim)).item() > float(args.max_target_delta):
                print(
                    "SAFETY STOP: target delta exceeded max-target-delta "
                    f"({float(args.max_target_delta):.3f} rad)."
                )
                break

            if mode == "PREPARE":
                pressed = key_reader.poll_key()
                if pressed == " ":
                    policy_start = time.time()
                    mode = "POLICY" if send_policy_enabled else "POLICY_PREVIEW"
                    print(f"SPACE detected: switching to {mode} mode.")

                prepare_elapsed = time.time() - prepare_start
                stand_active = bool(args.send_stand) and (prepare_elapsed <= float(args.stand_duration))
                if bool(args.send_stand) and (not stand_active) and (not stand_phase_expired_printed):
                    print(
                        f"[PREPARE] stand-duration elapsed ({args.stand_duration}s): "
                        "stand commands are now paused while waiting for SPACE."
                    )
                    stand_phase_expired_printed = True

                if stand_active:
                    stand_target_q_sdk = sim_targets_to_sdk(stand_target_q_sim, sdk_to_sim=sdk_to_sim)
                    fill_lowcmd(low_cmd, current_q_sdk=current_q_sdk, target_q_sdk=stand_target_q_sdk, kp=args.kp, kd=args.kd)
                    udp_low.SetSend(low_cmd)
                    udp_low.Send()

            else:
                if policy_start is None:
                    policy_start = time.time()
                elapsed_policy = time.time() - policy_start
                if elapsed_policy > float(args.duration):
                    print(f"{mode}: duration elapsed ({args.duration}s). Exiting.")
                    break

                if send_policy_enabled:
                    target_q_sdk = sim_targets_to_sdk(policy_target_q_sim, sdk_to_sim=sdk_to_sim)
                    fill_lowcmd(low_cmd, current_q_sdk=current_q_sdk, target_q_sdk=target_q_sdk, kp=args.kp, kd=args.kd)
                    udp_low.SetSend(low_cmd)
                    udp_low.Send()

            last_action = action.clone()

            if step_idx % int(args.print_every) == 0:
                current_sim_q = diag["dof_pos_sim"]
                preview_err = (policy_target_q_sim - current_sim_q)
                send_now = False
                if mode == "PREPARE":
                    send_now = bool(args.send_stand) and ((time.time() - prepare_start) <= float(args.stand_duration))
                elif mode == "POLICY":
                    send_now = send_policy_enabled
                action_min = float(torch.min(action).item())
                action_max = float(torch.max(action).item())
                blv = diag["base_lin_vel"]
                print(
                    f"[{mode}] step={step_idx} cmd=({args.vx:.2f},{args.vy:.2f},{args.yaw_rate:.2f}) "
                    f"a[min,max]=({action_min:.3f},{action_max:.3f}) "
                    f"base_lin_vel=({float(blv[0]):.3f},{float(blv[1]):.3f},{float(blv[2]):.3f}) "
                    f"roll={roll:.3f} pitch={pitch:.3f} send={send_now} "
                    f"q0={float(current_sim_q[0]):.3f} err0={float(preview_err[0]):.3f}"
                )

            loop_elapsed = time.time() - loop_start
            sleep_time = dt - loop_elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass
    finally:
        key_reader.stop()

    print("deploy_policy finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
