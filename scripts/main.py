#!/usr/bin/env python3
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import quaternion
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
import os
import psutil
import sys
import time

from dmp.dmp_position import PositionDMP
from dmp.dmp_rotation import RotationDMP
from controller.admittance_controller import AdaCon
from utils.data_processing import extractData
from utils.utils import *


if __name__ == '__main__':
    
    # Connection with the robot
    rtde_frequency = 500.0
    dt = 1.0 / rtde_frequency  # 2ms
    flags = RTDEControl.FLAG_VERBOSE | RTDEControl.FLAG_UPLOAD_SCRIPT
    ur_cap_port = 50002
    robot_ip = "192.168.1.131"
    lookahead_time = 0.1
    gain = 100
    rt_receive_priority = 90
    rt_control_priority = 85

    rtde_r = RTDEReceive(robot_ip, rtde_frequency, [], True, False, rt_receive_priority)
    rtde_c = RTDEControl(robot_ip, rtde_frequency, flags, ur_cap_port, rt_control_priority)

    # Set application real-time priority
    os_used = sys.platform
    process = psutil.Process(os.getpid())
    if os_used == "win32":  # Windows (either 32-bit or 64-bit)
        process.nice(psutil.REALTIME_PRIORITY_CLASS)
    elif os_used == "linux":  # linux
        rt_app_priority = 80
        param = os.sched_param(rt_app_priority)
        try:
            os.sched_setscheduler(0, os.SCHED_FIFO, param)
        except OSError:
            print("Failed to set real-time process scheduler to %u, priority %u" % (os.SCHED_FIFO, rt_app_priority))
        else:
            print("Process real-time priority set to: %u" % rt_app_priority)

    # Load a demonstration file containing robot positions.
    # demo = np.loadtxt("demo.dat", delimiter=" ", skiprows=1)
    str = "data.csv"
    demo = extractData(str, 'actual_TCP_pose_0')
    force = extractData(str, 'actual_TCP_force_0')
    actual_velocity = extractData(str, 'actual_TCP_speed_0')
    print(actual_velocity)
    actual_velocity_p = actual_velocity[:, 0:3]
    force_p = force[:, 0:3]

    # demo = np.array(demo)

    tau = 0.002 * len(demo)
    t = np.arange(0, tau, 0.002)
    # Split for position and rotation
    demo_p = demo[:, 0:3]
    demo_r = demo[:, 3:6]

    # 3D circle trajectory for simulation:
    #demo_p = createCircle(demo_p=demo_p)

    N = 180  # TODO: Try changing the number of basis functions to see how it affects the output.

    # DMP model for position:
    dmpP = PositionDMP(n_bfs=N, alpha=48.0)
    dmpP.train(demo_p, t, tau)


    # Change the initial and goal positions for the next floor
    dmpP.p0 = [dmpP.p0[0], dmpP.p0[1], dmpP.p0[2]-0.2]
    dmpP.gp = [dmpP.gp[0], dmpP.gp[1], dmpP.gp[2]-0.2]

    # TODO: ...or a different time constant:
    # tau = 12

    # Generate an output trajectory from the trained DMP
    # Position, velocity and acceleration:
    dmp_p, dmp_dp, dmp_ddp = dmpP.rollout(t, tau)

    alpha = 48
    n_bfs = 3  # No. of basis functions
    dof = 3  # Degrees of freedom

    # Learning parameters
    qs = 150
    qd = 100
    gamma = 15

    g = radialBasis(alpha=alpha, n_bfs=n_bfs, size=len(demo_p))

    ks, kd, v, tau1 = learnParameters(dof=dof, qs=qs, qd=qd, gamma=gamma, t=t, g=g, demo_p=dmp_p, actual_vel=dmp_dp)
    trials = 10 # Trials

    #Controller:
    ac = AdaCon(dof=dof, tr=trials, t=t)

    init_pose = [dmp_p[0, 0], dmp_p[0, 1], dmp_p[0, 2], demo_r[0, 0], demo_r[0, 1], demo_r[0, 2]]
    
    for trial in range(trials):
        
        # Move to initial position
        rtde_c.moveL(init_pose, 0.5, 0.5)
        
        # In the recording is needed, uncomment :
        # rtde_r.startFileRecording("data2.csv")
        # print("Data recording started, press [Ctrl-C] to end recording.")

        # Initialize :
        i = 0
        ac.pos_old = np.zeros(dof)
        ac.vel_old = np.zeros(dof)
        time.sleep(3)
        ac.pos = dmp_p[0, :]
        ac.vel = dmp_dp[0, :]
        
        try:
            while i < len(demo):
                print(i)
                t_start = rtde_c.initPeriod()

                # Get force from the sensor
                actual_force = rtde_r.getActualTCPForce()[0:3]
                ac.iter_learn(pos=dmp_p, rot=demo_r, i=trial, j=i, vel_p=dmp_dp, vel_r=dmp_p, ks=ks, kd=kd, actual_force = actual_force, desired_force=force_p,
                              actual_acceleration=dmp_ddp, tau1=tau1)
                
                # Motion using DMP trajectory:
                #rtde_c.speedL([dmp_dp[i, 0], dmp_dp[i, 1], dmp_dp[i, 2], 0, 0, 0], 0.5, 0.002)
                
                # Motion using controller trajectory:
                rtde_c.speedL([ac.vel[0], ac.vel[1], ac.vel[2], 0, 0, 0], 0.5, 0.002)
                
                # For recording:
                # sys.stdout.write("\r")
                # sys.stdout.write("{:3d} samples.".format(i))
                # sys.stdout.flush()

                rtde_c.waitPeriod(t_start)
                i += 1

        except KeyboardInterrupt:
            print("Control Interrupted!")
            rtde_c.speedStop()
            rtde_c.servoStop()
            rtde_c.stopScript()
            rtde_r.stopFileRecording()
            print("\nData recording stopped.")

        # Update parameters
        ks = ac.ks_collect[trial, :, :]
        kd = ac.kd_collect[trial, :, :]
        rtde_c.speedStop()

    rtde_c.stopScript()
    
    # Save data:
    np.save('tau.npy', ac.tau_collect)
    np.save('force.npy', ac.force_collect)
    np.save('kd.npy', ac.ks_collect)
    np.save('ks.npy', ac.kd_collect)
    np.save('pos_error.npy', ac.pos_err_collect)
    np.save('vel_error.npy', ac.vel_err_collect)
    
    
    # Plot the data:
    fig1, axs = plt.subplots(3, 1, sharex=True)
    axs[0].plot(t, ac.tau_collect[0, 0, :], 'blue', label='First trial')
    axs[0].plot(t, ac.tau_collect[3, 0, :], 'gray', label='Middle')
    axs[0].plot(t, ac.tau_collect[7, 0, :], 'gray')
    axs[0].plot(t, ac.tau_collect[trial - 1, 0, :], 'red', label='Last trial')
    axs[0].set_xlabel('t (s)')
    axs[0].set_ylabel('Tau[0]')

    axs[1].plot(t, ac.tau_collect[0, 1, :], 'blue', label='First trial')
    axs[1].plot(t, ac.tau_collect[3, 1, :], 'gray', label='Middle')
    axs[1].plot(t, ac.tau_collect[7, 1, :], 'gray')
    axs[1].plot(t, ac.tau_collect[trial - 1, 1, :], 'red', label='Last trial')
    axs[1].set_xlabel('t (s)')
    axs[1].set_ylabel('Tau[1]')

    axs[2].plot(t, ac.tau_collect[0, 2, :], 'blue', label='First trial')
    axs[2].plot(t, ac.tau_collect[3, 2, :], 'gray', label='Middle')
    axs[2].plot(t, ac.tau_collect[7, 2, :], 'gray')
    axs[2].plot(t, ac.tau_collect[trial - 1, 2, :], 'red', label='Last trial')
    axs[2].set_xlabel('t (s)')
    axs[2].set_ylabel('Tau[2]')
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()

    fig1, axs = plt.subplots(3, 1, sharex=True)
    axs[0].plot(t, ac.force_collect[0, 0, :], 'blue', label='First trial')
    axs[0].plot(t, ac.force_collect[3, 0, :], 'gray', label='Middle')
    axs[0].plot(t, ac.force_collect[7, 0, :], 'gray')
    axs[0].plot(t, ac.force_collect[trial - 1, 0, :], 'red', label='Last trial')
    axs[0].set_xlabel('t (s)')
    axs[0].set_ylabel('Ext Force[0]')

    axs[1].plot(t, ac.force_collect[0, 1, :], 'blue', label='First trial')
    axs[1].plot(t, ac.force_collect[3, 1, :], 'gray', label='Middle')
    axs[1].plot(t, ac.force_collect[7, 1, :], 'gray')
    axs[1].plot(t, ac.force_collect[trial - 1, 1, :], 'red', label='Last trial')
    axs[1].set_xlabel('t (s)')
    axs[1].set_ylabel('Ext Force[1]')

    axs[2].plot(t, ac.force_collect[0, 2, :], 'blue', label='First trial')
    axs[2].plot(t, ac.force_collect[3, 2, :], 'gray', label='Middle')
    axs[2].plot(t, ac.force_collect[7, 2, :], 'gray')
    axs[2].plot(t, ac.force_collect[trial - 1, 2, :], 'red', label='Last trial')
    axs[2].set_xlabel('t (s)')
    axs[2].set_ylabel('Ext Force[2]')
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()

    fig1, axs = plt.subplots(3, 1, sharex=True)
    axs[0].plot(t, ac.tau_collect[trial-1, 0, :], 'blue', label='Tau')
    axs[0].plot(t, ac.force_collect[trial - 1, 0, :], 'red', label='External Force')

    axs[0].set_xlabel('t (s)')
    axs[0].set_ylabel('Tau[0] vs Ext Force[0]')

    axs[1].plot(t, ac.tau_collect[trial-1, 1, :], 'blue', label='Tau')
    axs[1].plot(t, ac.force_collect[trial - 1, 1, :], 'red', label='External Force')
    axs[1].set_xlabel('t (s)')
    axs[1].set_ylabel('Tau[1] vs Ext Force[1]')

    axs[2].plot(t, ac.tau_collect[trial - 1, 2, :], 'blue', label='Tau')
    axs[2].plot(t, ac.force_collect[trial - 1, 2, :], 'red', label='External Force')
    axs[2].set_xlabel('t (s)')
    axs[2].set_ylabel('Tau[2] vs Ext Force[2]')
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()

    fig1, axs = plt.subplots(4, 1, sharex=True)
    axs[0].plot(t, ac.ks_collect[0,0, :], 'blue', label='First trial')
    axs[0].plot(t, ac.ks_collect[3,0, :], 'gray',label='Middle')
    axs[0].plot(t, ac.ks_collect[7,0, :], 'gray')
    axs[0].plot(t, ac.ks_collect[trial-1,0, :], 'red', label='Last trial')

    axs[0].set_xlabel('t (s)')
    axs[0].set_ylabel('Ks')

    axs[1].plot(t, ac.kd_collect[0,0, :], 'blue', label='First trial')
    axs[1].plot(t, ac.kd_collect[3,0, :], 'gray', label='Middle')
    axs[1].plot(t, ac.kd_collect[7,0, :], 'gray')
    axs[1].plot(t, ac.kd_collect[trial-1,0, :], 'red', label='Last trial')
    axs[1].set_xlabel('t (s)')
    axs[1].set_ylabel('Kd')
    
    axs[2].plot(t, ac.pos_err_collect[0,0, :], 'blue', label='First trial')
    axs[2].plot(t, ac.pos_err_collect[3,0, :], 'gray', label='Middle')
    axs[2].plot(t, ac.pos_err_collect[7,0, :], 'gray')
    axs[2].plot(t, ac.pos_err_collect[trial-1,0, :], 'red', label='Last trial')
    axs[2].set_xlabel('t (s)')
    axs[2].set_ylabel('Position error')
    
    axs[3].plot(t, ac.vel_err_collect[0,0, :], 'blue', label='First trial')
    axs[3].plot(t, ac.vel_err_collect[3,0, :], 'gray', label='Middle')
    axs[3].plot(t, ac.vel_err_collect[7,0, :], 'gray')
    axs[3].plot(t, ac.vel_err_collect[trial-1,0, :], 'red', label='Last trial')
    axs[3].set_xlabel('t (s)')
    axs[3].set_ylabel('Velocity error')
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    axs[3].legend()
    plt.show()

    # Rotation:

    # DMP model for rotation:
    '''
    dmpR = RotationDMP(n_bfs=N, alpha=48.0)
    dmpR.train(data_q, t, tau)

    demo_quat_array = np.empty((len(data_q), 4))
    for n, d in enumerate(data_q):
        demo_quat_array[n] = [d[0], d[1], d[2], d[3]]

    # dmpR.q0=[0.015457728325892973, -0.9997245418426188, -0.01703957032809377, -0.004642425614064722]
    # dmpR.gq=[0.012457728325892973, 0.9397245418426188, 0.01703957032809377, 0.004642425614064722]
    dmp_q, dmp_dq, dmp_ddq = dmpR.rollout(t, tau)

    dmp_q_list = np.empty((len(dmp_q), 4))
    for n, d in enumerate(dmp_q):
        # print(quaternion.as_float_array(d)[2])
        arr = quaternion.as_float_array(d)
        dmp_q_list[n] = [arr[0], arr[1], arr[2], arr[3]]


    # 2D plot the DMP against the original demonstration
    fig3, axsq = plt.subplots(4, 1, sharex=True)
    axsq[0].plot(t, data_q[:, 0], label='Demonstration')
    axsq[0].plot(t, dmp_q_list[:, 0], label='DMProt')
    axsq[0].set_xlabel('t (s)')
    axsq[0].set_ylabel('X (m)')

    axsq[1].plot(t, data_q[:, 1], label='Demonstration')
    axsq[1].plot(t, dmp_q_list[:, 1], label='DMProt')
    axsq[1].set_xlabel('t (s)')
    axsq[1].set_ylabel('Y (m)')

    axsq[2].plot(t, data_q[:, 2], label='Demonstration')
    axsq[2].plot(t, dmp_q_list[:, 2], label='DMProt')
    axsq[2].set_xlabel('t (s)')
    axsq[2].set_ylabel('Z (m)')
    axsq[2].legend()

    axsq[3].plot(t, data_q[:, 3], label='Demonstration')
    axsq[3].plot(t, dmp_q_list[:, 3], label='DMProt')
    axsq[3].set_xlabel('t (s)')
    axsq[3].set_ylabel('Z (m)')
    axsq[3].legend()
    '''
