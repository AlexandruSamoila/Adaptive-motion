from __future__ import division, print_function
from dmp_position import PositionDMP
#from dmp_rot import RotationDMP
from rotation import RotationDMP
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyquaternion import Quaternion
import math
import quaternion
from AdaptiveAdmittanceController import Ada_con

import numpy as np # Scientific computing library for Python
 
def euler2quat(data):

  #Convert an Euler angle to a quaternion.
   
  data_q = np.zeros((len(data),4))
  i=0
  for o in data:
    roll = o[0]
    pitch = o[1]
    yaw = o[2]
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    
    data_q[i]=[qw,qx,qy,qz]
    i+=1
 
  return data_q

def axis2quat(data):
      
      # Convert angle axis to quaternion
      
      data_q=np.empty((len(data),4))
      
      for i, d in enumerate(data):
        angle = np.linalg.norm(d)
        axis_normed = d/angle
        s = math.sin(angle/2)
        data_q[i]=[math.cos(angle/2), s*axis_normed[0], s*axis_normed[1], s*axis_normed[2]]
        
      return data_q

if __name__ == '__main__':
    # Load a demonstration file containing robot positions.
    demo = np.loadtxt("demo.dat", delimiter=" ", skiprows=1)

    tau = 0.002 * len(demo)
    t = np.arange(0, tau, 0.002)
    #Split for position and rotation
    demo_p = demo[:, 0:3]
    demo_r = demo[:, 3:6]

    # for i in range(len(demo_r)-1):
    #     if demo_r[i].dot(demo_r[i+1]) < 0:
    #         demo_r[i+1] *= -1
    
    data_q = axis2quat(demo_r)
 
  
    N = 67 # TODO: Try changing the number of basis functions to see how it affects the output.
    
    #DMP model for position:
    dmpP = PositionDMP(n_bfs=N, alpha=48.0)
    dmpP.train(demo_p, t, tau)
    
    #DMP model for rotation:
    
    # dmpR = RotationDMP(n_bfs=N, alpha=48.0)
    # dmpR.train(data_q,t,tau)
    
    # demo_quat_array = np.empty((len(data_q),4))
    # for n, d in enumerate(data_q):
    #   demo_quat_array[n] = [d[0],d[1],d[2],d[3]]
    
    
    

    # TODO: Try setting a different starting point for the dmp:
    # dmp.p0 = [x, y, z]
    
    #dmpP.p0=[0.341529, 0.121866, 0.3000988]
    #dmpR.q0=[]
    
    # TODO: ...or a different goal point:
    # dmp.g0 = [x, y, z]\
        
    #dmpP.gp=[0.1, 0.2, 0.469289]
    #dmp.g0=[0.36, 0.22, 0.469289]
    # TODO: ...or a different time constant:
    #tau=12

    # Generate an output trajectory from the trained DMP
    # Position, velocity and acceleration:
    dmp_p, dmp_dp, dmp_ddp = dmpP.rollout(t, tau)
    
    tr=22 #trials
    
    ac = Ada_con(dof = 3,tr=tr,t=t)
    ac.iter_learn(pos=dmp_p,vel_p=dmp_dp)
    fig2 = plt.figure(2)
    ax = plt.axes(projection='3d')
    ax.plot3D(ac.pos_collect[tr-1, 0,:], ac.pos_collect[tr-1, 1,:], ac.pos_collect[tr-1, 2,:], label='System')
    ax.plot3D(dmp_p[:, 0], dmp_p[:, 1], dmp_p[:, 2], label='DMP')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()
    exit()
    
    fig, axs = plt.subplots(2)
    fig.suptitle('Simulation')
    axs[0].plot(t, dmp_p[:,2],label='DMP')
    axs[1].plot(t, ac.pos_collect[tr - 1][0],label='System')
    axs[0].legend()
    axs[1].legend()
    plt.show()
   
    
    
    
    
    #dmpR.q0=[0.015457728325892973, -0.9997245418426188, -0.01703957032809377, -0.004642425614064722]
    #dmpR.gq=[0.012457728325892973, 0.9397245418426188, 0.01703957032809377, 0.004642425614064722]
    dmp_q, dmp_dq, dmp_ddq = dmpR.rollout(t, tau)
  
    dmp_q_list = np.empty((len(dmp_q),4))
    for n, d in enumerate(dmp_q):
      #print(quaternion.as_float_array(d)[2]) 
      arr = quaternion.as_float_array(d)  
      dmp_q_list[n] = [arr[0],arr[1],arr[2],arr[3]]
    
    # dmp_q_list=np.zeros((len(dmp_q)-1,4))
    # for d,dm in enumerate(dmp_q):
    #        dmp_q_list[d]=[dmp_q[d].w,dmp_q[d].x,dmp_q[d].y,dmp_q[d].z]
    
    
    
    # # Plot values for the simulation!
    # fig1, axs = plt.subplots(6, 1, sharex=True)
    # axs[0].plot(t, ac.pos_collect[0, 0,:], label='Demonstration')
    # #axs[0].plot(t, dmp_p[:, 0], label='DMP')
    # axs[0].set_xlabel('t (s)')
    # axs[0].set_ylabel('q1')

    # axs[1].plot(t, ac.pos_collect[0, 1,:], label='Demonstration')
    # #axs[1].plot(t, dmp_p[:, 1], label='DMP')
    # axs[1].set_xlabel('t (s)')
    # axs[1].set_ylabel('q2')

    # axs[2].plot(t, ac.pos_collect[0, 2,:], label='Demonstration')
    # #axs[2].plot(t, dmp_p[:, 2], label='DMP')
    # axs[2].set_xlabel('t (s)')
    # axs[2].set_ylabel('q3')
    # axs[3].plot(t, ac.pos_collect[0, 3,:], label='Demonstration')
    # #axs[2].plot(t, dmp_p[:, 2], label='DMP')
    # axs[3].set_xlabel('t (s)')
    # axs[3].set_ylabel('q4')
    # axs[4].plot(t, ac.pos_collect[0, 4,:], label='Demonstration')
    # #axs[2].plot(t, dmp_p[:, 2], label='DMP')
    # axs[4].set_xlabel('t (s)')
    # axs[4].set_ylabel('q5')
    # axs[5].plot(t, ac.pos_collect[0, 5,:], label='Demonstration')
    # #axs[2].plot(t, dmp_p[:, 2], label='DMP')
    # axs[5].set_xlabel('t (s)')
    # axs[5].set_ylabel('q6')
    # axs[5].legend()
    # plt.show()
    #2D plot the DMP against the original demonstration
    fig1, axs = plt.subplots(3, 1, sharex=True)
    axs[0].plot(t, demo_p[:, 0], label='Demonstration')
    axs[0].plot(t, dmp_p[:, 0], label='DMP')
    axs[0].set_xlabel('t (s)')
    axs[0].set_ylabel('X (m)')

    axs[1].plot(t, demo_p[:, 1], label='Demonstration')
    axs[1].plot(t, dmp_p[:, 1], label='DMP')
    axs[1].set_xlabel('t (s)')
    axs[1].set_ylabel('Y (m)')

    axs[2].plot(t, demo_p[:, 2], label='Demonstration')
    axs[2].plot(t, dmp_p[:, 2], label='DMP')
    axs[2].set_xlabel('t (s)')
    axs[2].set_ylabel('Z (m)')
    axs[2].legend()

    # 3D plot the DMP against the original demonstration
    fig2 = plt.figure(2)
    ax = plt.axes(projection='3d')
    ax.plot3D(demo_p[:, 0], demo_p[:, 1], demo_p[:, 2], label='Demonstration')
    ax.plot3D(dmp_p[:, 0], dmp_p[:, 1], dmp_p[:, 2], label='DMP')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    #plt.show()
    
    # 2D plot the DMP against the original demonstration
    fig3, axsq = plt.subplots(4, 1, sharex=True)
    axsq[0].plot(t, data_q[:,0], label='Demonstration')
    axsq[0].plot(t, dmp_q_list[:,0], label='DMProt')
    axsq[0].set_xlabel('t (s)')
    axsq[0].set_ylabel('X (m)')

    axsq[1].plot(t, data_q[:,1], label='Demonstration')
    axsq[1].plot(t, dmp_q_list[:,1], label='DMProt')
    axsq[1].set_xlabel('t (s)')
    axsq[1].set_ylabel('Y (m)')

    axsq[2].plot(t, data_q[:,2], label='Demonstration')
    axsq[2].plot(t, dmp_q_list[:,2], label='DMProt')
    axsq[2].set_xlabel('t (s)')
    axsq[2].set_ylabel('Z (m)')
    axsq[2].legend()
    
    axsq[3].plot(t, data_q[:,3], label='Demonstration')
    axsq[3].plot(t, dmp_q_list[:,3], label='DMProt')
    axsq[3].set_xlabel('t (s)')
    axsq[3].set_ylabel('Z (m)')
    axsq[3].legend()

    # # 3D plot the DMP against the original demonstration
    # fig4 = plt.figure(2)
    # axq = plt.axes(projection='3d')
    # axq.plot3D(data_q[:][0], data_q[:][1], data_q[:][2], label='Demonstration')
    # axq.plot3D(dmp_q_list[:][0], dmp_q_list[:][1], dmp_q_list[:][2], label='DMP')
    # axq.set_xlabel('X')
    # axq.set_ylabel('Y')
    # axq.set_zlabel('Z')
    # axq.legend()
    # plt.show()