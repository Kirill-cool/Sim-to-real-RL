#!/usr/bin/python

import sys
import time
import math
#import numpy as np

sys.path.append('../lib/python/amd64')
import robot_interface as sdk

# low cmd
TARGET_PORT = 8007
LOCAL_PORT = 8082
TARGET_IP = "192.168.123.10"   # target IP address

LOW_CMD_LENGTH = 610
LOW_STATE_LENGTH = 771

PRINT_EVERY_N_STEPS = 50
MOTOR_NAMES = [
    "FR_0", "FR_1", "FR_2",
    "FL_0", "FL_1", "FL_2",
    "RR_0", "RR_1", "RR_2",
    "RL_0", "RL_1", "RL_2",
]


def _fmt_float_seq(seq, precision=4):
    return "[" + ", ".join(f"{v:.{precision}f}" for v in seq) + "]"


def _fmt_int_seq(seq):
    return "[" + ", ".join(str(v) for v in seq) + "]"


def print_low_state_structured(state, step):
    print("\n" + "=" * 100)
    print(f"LOW STATE SNAPSHOT | step={step}")
    print("=" * 100)
    print(
        f"levelFlag={state.levelFlag}  commVersion={state.commVersion}  "
        f"robotID={state.robotID}  SN={state.SN}  bandWidth={state.bandWidth}"
    )
    print(
        "imu: "
        f"quaternion={_fmt_float_seq(state.imu.quaternion)}  "
        f"gyroscope={_fmt_float_seq(state.imu.gyroscope)}  "
        f"accelerometer={_fmt_float_seq(state.imu.accelerometer)}  "
        f"rpy={_fmt_float_seq(state.imu.rpy)}  "
        f"temperature={state.imu.temperature}"
    )

    print("-" * 100)
    print("motorState[0..19]:")
    for i in range(20):
        m = state.motorState[i]
        if i < len(MOTOR_NAMES):
            motor_label = MOTOR_NAMES[i]
        else:
            motor_label = f"M{i}"
        print(
            f"  [{i:02d}] {motor_label}: "
            f"mode={m.mode}  q={m.q:.4f}  dq={m.dq:.4f}  ddq={m.ddq:.4f}  "
            f"tauEst={m.tauEst:.4f}  q_raw={m.q_raw:.4f}  dq_raw={m.dq_raw:.4f}  "
            f"ddq_raw={m.ddq_raw:.4f}  temp={m.temperature}  reserve={_fmt_int_seq(m.reserve)}"
        )

    print("-" * 100)
    print(f"footForce={_fmt_int_seq(state.footForce)}")
    print(f"footForceEst={_fmt_int_seq(state.footForceEst)}")
    print(f"tick={state.tick}")
    print(f"wirelessRemote={_fmt_int_seq(state.wirelessRemote)}")
    print(f"reserve={state.reserve}  crc={state.crc}")
    print("=" * 100)


def jointLinearInterpolation(initPos, targetPos, rate):

    #rate = np.fmin(np.fmax(rate, 0.0), 1.0)
    if rate > 1.0:
        rate = 1.0
    elif rate < 0.0:
        rate = 0.0

    p = initPos*(1-rate) + targetPos*rate
    return p


if __name__ == '__main__':

    d = {'FR_0':0, 'FR_1':1, 'FR_2':2,
         'FL_0':3, 'FL_1':4, 'FL_2':5, 
         'RR_0':6, 'RR_1':7, 'RR_2':8, 
         'RL_0':9, 'RL_1':10, 'RL_2':11 }
    PosStopF  = math.pow(10,9)
    VelStopF  = 16000.0
    HIGHLEVEL = 0x00
    LOWLEVEL  = 0xff
    sin_mid_q = [0.0, 1.2, -2.0]
    dt = 0.002
    qInit = [0, 0, 0]
    qDes = [0, 0, 0]
    sin_count = 0
    rate_count = 0
    Kp = [0, 0, 0]
    Kd = [0, 0, 0]

    udp = sdk.UDP(LOCAL_PORT, TARGET_IP, TARGET_PORT, LOW_CMD_LENGTH, LOW_STATE_LENGTH, -1)
    #udp = sdk.UDP(8082, "192.168.123.10", 8007, 610, 771)
    safe = sdk.Safety(sdk.LeggedType.Aliengo)
    
    cmd = sdk.LowCmd()
    state = sdk.LowState()
    udp.InitCmdData(cmd)
    cmd.levelFlag = LOWLEVEL


    Tpi = 0
    motiontime = 0
    while True:
        time.sleep(0.002)
        motiontime += 1

        # print(motiontime)
        udp.Recv()
        udp.GetRecv(state)
        if motiontime % PRINT_EVERY_N_STEPS == 0:
            print_low_state_structured(state, motiontime)
        
        if( motiontime >= 0):

            # first, get record initial position
            if( motiontime >= 0 and motiontime < 10):
                qInit[0] = state.motorState[d['FR_0']].q
                qInit[1] = state.motorState[d['FR_1']].q
                qInit[2] = state.motorState[d['FR_2']].q
            
            # second, move to the origin point of a sine movement with Kp Kd
            if( motiontime >= 10 and motiontime < 400):
                rate_count += 1
                rate = rate_count/200.0                       # needs count to 200
                # Kp = [5, 5, 5]
                # Kd = [1, 1, 1]
                Kp = [20, 20, 20]
                Kd = [2, 2, 2]
                
                qDes[0] = jointLinearInterpolation(qInit[0], sin_mid_q[0], rate)
                qDes[1] = jointLinearInterpolation(qInit[1], sin_mid_q[1], rate)
                qDes[2] = jointLinearInterpolation(qInit[2], sin_mid_q[2], rate)
            
            # last, do sine wave
            freq_Hz = 1
            # freq_Hz = 5
            freq_rad = freq_Hz * 2* math.pi
            t = dt*sin_count
            if( motiontime >= 400):
                sin_count += 1
                # sin_joint1 = 0.6 * sin(3*M_PI*sin_count/1000.0)
                # sin_joint2 = -0.9 * sin(3*M_PI*sin_count/1000.0)
                sin_joint1 = 0.6 * math.sin(t*freq_rad)
                sin_joint2 = -0.9 * math.sin(t*freq_rad)
                qDes[0] = sin_mid_q[0]
                qDes[1] = sin_mid_q[1] + sin_joint1
                qDes[2] = sin_mid_q[2] + sin_joint2
                # qDes[2] = sin_mid_q[2]
            

            cmd.motorCmd[d['FR_0']].q = 0
            cmd.motorCmd[d['FR_0']].dq = 0
            cmd.motorCmd[d['FR_0']].Kp = Kp[0]
            cmd.motorCmd[d['FR_0']].Kd = Kd[0]
            cmd.motorCmd[d['FR_0']].tau = -1.6

            cmd.motorCmd[d['FR_1']].q = 0
            cmd.motorCmd[d['FR_1']].dq = 0
            cmd.motorCmd[d['FR_1']].Kp = Kp[1]
            cmd.motorCmd[d['FR_1']].Kd = Kd[1]
            cmd.motorCmd[d['FR_1']].tau = 0.0

            cmd.motorCmd[d['FR_2']].q =  0
            cmd.motorCmd[d['FR_2']].dq = 0
            cmd.motorCmd[d['FR_2']].Kp = Kp[2]
            cmd.motorCmd[d['FR_2']].Kd = Kd[2]
            cmd.motorCmd[d['FR_2']].tau = 0.0
            # cmd.motorCmd[d['FR_2']].tau = 2 * sin(t*freq_rad)


        # if(motiontime > 10):
        #     safe.PowerProtect(cmd, state, 1)




        # udp.SetSend(cmd)
        # udp.Send()
