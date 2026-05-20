#!/usr/bin/python

import sys
import time
import math
import select
import tty
import termios

sys.path.append('../lib/python/amd64')
import robot_interface as sdk

# low cmd
TARGET_PORT = 8007
LOCAL_PORT = 8082
TARGET_IP = "192.168.123.10"   # target IP address

LOW_CMD_LENGTH = 610
LOW_STATE_LENGTH = 771

READ_DT = 0.002 # 500Hz
PRINT_EVERY_N_STEPS = 50

# Joint control setup
# Set target per joint (rad).
# Use None to ignore a joint, or float value to command it.
TARGET_POSITIONS_RAD = {
    "FR_0": -0.1,
    "FR_1": 1.,
    "FR_2": -1.5,
    "FL_0": 0.1,
    "FL_1": 1.,
    "FL_2": -1.5,
    "RR_0": -0.1,
    "RR_1": 1.,
    "RR_2": -1.5,
    "RL_0": 0.1,
    "RL_1": 1.,
    "RL_2": -1.5,
}
# Per-joint command parameters (applied in Stage 2 for enabled joints)
TARGET_DQ = {
    "FR_0": 0.0, "FR_1": 0.0, "FR_2": 0.0,
    "FL_0": 0.0, "FL_1": 0.0, "FL_2": 0.0,
    "RR_0": 0.0, "RR_1": 0.0, "RR_2": 0.0,
    "RL_0": 0.0, "RL_1": 0.0, "RL_2": 0.0,
}
TARGET_KP = {
    "FR_0": 40.0, "FR_1": 40.0, "FR_2": 40.0,
    "FL_0": 40.0, "FL_1": 40.0, "FL_2": 40.0,
    "RR_0": 40.0, "RR_1": 40.0, "RR_2": 40.0,
    "RL_0": 40.0, "RL_1": 40.0, "RL_2": 40.0,
}
TARGET_KD = {
    "FR_0": 2.0, "FR_1": 2.0, "FR_2": 2.0,
    "FL_0": 2.0, "FL_1": 2.0, "FL_2": 2.0,
    "RR_0": 2.0, "RR_1": 2.0, "RR_2": 2.0,
    "RL_0": 2.0, "RL_1": 2.0, "RL_2": 2.0,
}
TARGET_TAU = {
    "FR_0": 0.0, "FR_1": 0.0, "FR_2": 0.0,
    "FL_0": 0.0, "FL_1": 0.0, "FL_2": 0.0,
    "RR_0": 0.0, "RR_1": 0.0, "RR_2": 0.0,
    "RL_0": 0.0, "RL_1": 0.0, "RL_2": 0.0,
}
MOVE_DURATION_SEC = 2.0


def jointLinearInterpolation(initPos, targetPos, rate):
    if rate > 1.0:
        rate = 1.0
    elif rate < 0.0:
        rate = 0.0
    return initPos * (1 - rate) + targetPos * rate


def setup_cbreak_keyboard():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    return fd, old_settings


def restore_keyboard(fd, old_settings):
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def read_key_nonblocking(fd):
    ready, _, _ = select.select([fd], [], [], 0.0)
    if ready:
        return sys.stdin.read(1)
    return None


def print_stage_status(state, step, stage_name, joint_targets, q_cmd_map=None):
    parts = []
    for joint_name, joint_idx in joint_targets:
        q = state.motorState[joint_idx].q
        dq = state.motorState[joint_idx].dq
        part = f"{joint_name}({joint_idx}) q={q:.4f} dq={dq:.4f}"
        if q_cmd_map is not None:
            part += f" q_cmd={q_cmd_map[joint_name]:.4f}"
        parts.append(part)

    print(
        f"[{stage_name}] step={step} | "
        + " | ".join(parts)
        + f" | imu_rpy=({state.imu.rpy[0]:.4f}, {state.imu.rpy[1]:.4f}, {state.imu.rpy[2]:.4f})"
    )


if __name__ == '__main__':
    d = {
        'FR_0': 0, 'FR_1': 1, 'FR_2': 2,
        'FL_0': 3, 'FL_1': 4, 'FL_2': 5,
        'RR_0': 6, 'RR_1': 7, 'RR_2': 8,
        'RL_0': 9, 'RL_1': 10, 'RL_2': 11,
    }

    PosStopF = math.pow(10, 9)
    VelStopF = 16000.0
    LOWLEVEL = 0xff

    for joint_name in TARGET_POSITIONS_RAD:
        if joint_name not in d:
            raise ValueError(
                f"Unknown joint '{joint_name}' in TARGET_POSITIONS_RAD. "
                f"Available joints: {sorted(d.keys())}"
            )
    for joint_name in d:
        if joint_name not in TARGET_DQ:
            raise ValueError(f"Missing '{joint_name}' in TARGET_DQ.")
        if joint_name not in TARGET_KP:
            raise ValueError(f"Missing '{joint_name}' in TARGET_KP.")
        if joint_name not in TARGET_KD:
            raise ValueError(f"Missing '{joint_name}' in TARGET_KD.")
        if joint_name not in TARGET_TAU:
            raise ValueError(f"Missing '{joint_name}' in TARGET_TAU.")
    target_joints = [
        (joint_name, d[joint_name])
        for joint_name, target in TARGET_POSITIONS_RAD.items()
        if target is not None
    ]
    if not target_joints:
        raise ValueError("No joints selected for control. Set at least one non-None value in TARGET_POSITIONS_RAD.")

    udp = sdk.UDP(LOCAL_PORT, TARGET_IP, TARGET_PORT, LOW_CMD_LENGTH, LOW_STATE_LENGTH, -1)
    cmd = sdk.LowCmd()
    state = sdk.LowState()

    udp.InitCmdData(cmd)
    cmd.levelFlag = LOWLEVEL

    motiontime = 0
    stage = 1
    stage2_start_step = None
    stage2_start_time = None
    stage2_start_q = {}

    print("Stage 1: reading robot state only.")
    print("Target joints/positions (rad):")
    for joint_name, _ in target_joints:
        print(
            f"  {joint_name}: q={TARGET_POSITIONS_RAD[joint_name]:.4f}, "
            f"dq={TARGET_DQ[joint_name]:.4f}, "
            f"Kp={TARGET_KP[joint_name]:.4f}, "
            f"Kd={TARGET_KD[joint_name]:.4f}, "
            f"tau={TARGET_TAU[joint_name]:.4f}"
        )
    print("Press SPACE to start Stage 2 (target command). Press 'q' to exit.")

    fd, old_settings = setup_cbreak_keyboard()
    try:
        while True:
            time.sleep(READ_DT)
            motiontime += 1

            # Keep state acquisition unchanged
            udp.Recv()
            udp.GetRecv(state)

            key = read_key_nonblocking(fd)
            if key in ('q', 'Q'):
                print("Exit requested by user.")
                break

            if stage == 1:
                if motiontime % PRINT_EVERY_N_STEPS == 0:
                    print_stage_status(state, motiontime, "STAGE1", target_joints)

                if key == ' ':
                    stage = 2
                    stage2_start_step = motiontime
                    stage2_start_time = time.monotonic()
                    for joint_name, joint_idx in target_joints:
                        stage2_start_q[joint_name] = state.motorState[joint_idx].q
                    print(f"Stage 2 started at step={motiontime}")
                    for joint_name, _ in target_joints:
                        print(
                            f"  {joint_name}: q_start={stage2_start_q[joint_name]:.4f} "
                            f"-> q_target={TARGET_POSITIONS_RAD[joint_name]:.4f}"
                        )

            if stage == 2:
                elapsed_sec = time.monotonic() - stage2_start_time
                rate = elapsed_sec / MOVE_DURATION_SEC
                q_cmd_map = {}
                for joint_name, _ in target_joints:
                    q_cmd_map[joint_name] = jointLinearInterpolation(
                        stage2_start_q[joint_name],
                        TARGET_POSITIONS_RAD[joint_name],
                        rate,
                    )

                for i in range(20):
                    cmd.motorCmd[i].q = PosStopF
                    cmd.motorCmd[i].dq = VelStopF
                    cmd.motorCmd[i].Kp = 0.0
                    cmd.motorCmd[i].Kd = 0.0
                    cmd.motorCmd[i].tau = 0.0

                for joint_name, joint_idx in target_joints:
                    cmd.motorCmd[joint_idx].q = q_cmd_map[joint_name]
                    cmd.motorCmd[joint_idx].dq = TARGET_DQ[joint_name]
                    cmd.motorCmd[joint_idx].Kp = TARGET_KP[joint_name]
                    cmd.motorCmd[joint_idx].Kd = TARGET_KD[joint_name]
                    cmd.motorCmd[joint_idx].tau = TARGET_TAU[joint_name]

                udp.SetSend(cmd)
                udp.Send()

                if motiontime % PRINT_EVERY_N_STEPS == 0:
                    print_stage_status(state, motiontime, "STAGE2", target_joints, q_cmd_map=q_cmd_map)

    finally:
        restore_keyboard(fd, old_settings)
