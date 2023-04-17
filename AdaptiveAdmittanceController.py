class Ada_con():
    
    def __init__(self, dof, tr, t):
        import numpy as np
        #from pDMP_functions import pDMP

        # Number of learning trials
        self.nt = tr # Value chosen arbitrarily

        # Number of samples of trajectory per trial
        self.time = t # Total experiment time - chosen arbitrarily
        self.dt = 0.05 # Time step - chosen arbitrarily
        self.samples = int(1/self.dt) * self.time + 1

        # Number of degrees of freedom of robot to be controlled
        self.dof = dof

        # Translational and rotational part of desired position and velocity
        self.tran_des = np.zeros(3)
        self.rot_des = np.zeros(3)
        self.dtran_des = np.zeros(3)
        self.drot_des = np.zeros(3)
        self.tran = np.zeros(3)
        self.rot = np.zeros(3)
        self.dtran = np.zeros(3)
        self.drot = np.zeros(3)

        # Desired position and velocity
        self.pos_des = np.zeros(self.dof)
        self.vel_des = np.zeros(self.dof)

        # Actual position and velocity
        self.pos = np.zeros(self.dof)
        self.vel = np.zeros(self.dof)

        # Position and velocity difference / error
        self.pos_diff = np.zeros(self.dof)
        self.vel_diff = np.zeros(self.dof)

        # Tracking error
        self.tra_diff = np.zeros(self.dof)

        # Keep track of current position and velocity, position error and ...
        # velocity error, and tracking error during learning trials for plotting
        # Matrix of dimension:
        # (number of learning trials) x (DOF of robot) x (samples of one learning trial)
        self.pos_collect = np.zeros((self.nt, self.dof, self.samples))
        self.vel_collect = np.zeros((self.nt, self.dof, self.samples))
        self.pos_err_collect = np.zeros((self.nt, self.dof, self.samples))
        self.vel_err_collect = np.zeros((self.nt, self.dof, self.samples))
        self.tra_err_collect = np.zeros((self.nt, self.dof, self.samples))
        self.tau_collect = np.zeros((self.nt, self.dof, self.samples))
        self.ks_collect = np.zeros((self.nt, self.dof, self.samples))
        self.kd_collect = np.zeros((self.nt, self.dof, self.samples))
        self.pos_des_collect = np.zeros((self.nt, self.dof, self.samples))
        self.vel_des_collect = np.zeros((self.nt, self.dof, self.samples))
        self.v_collect = np.zeros((self.nt, self.dof, self.samples))


        # Controller output
        self.tau = np.zeros(self.dof)

        # Adaptive stiffness and damping
        self.ks = np.zeros(self.dof)
        self.kd = np.zeros(self.dof)

        # Feedforward term
        self.v = np.zeros(self.dof)

        # Adaptation rates for stiffness and damping
        self.qs = 3 # Value chosen arbitrarily
        self.qd = 2.6 # Value chosen arbitrarily

        # Adaptation rate for feedforward term
        self.qv = 1.5 # Value chosen arbitrarily

        # Tracking error coefficient
        self.gamma = 2 # Value chosen arbitrarily

        # Forgetting factor
        self.lambd = 0.995 # Chosen arbitrarily

        self.pos_old = np.zeros(self.dof) # Used for calculating desired velocity
        self.vel_old = np.zeros(self.dof) # Used for calculating desired acceleration

        # Temporary value for simulation
        self.rad = np.zeros(self.dof)

    def get_pos_diff(self):
        '''
            Generate position error
        '''
        import numpy as np

        self.pos_diff = np.subtract(self.pos, self.pos_des)
    
    def get_vel_diff(self):
        '''
            Generate velocity error
        '''
        import numpy as np

        self.vel_diff = np.subtract(self.vel, self.vel_des)
        #print("Velocity error: ", self.vel_diff)
        #print("Desired vel: ", self.vel_des)
        #print("Actual vel: ", self.vel)

    def get_tra_diff(self):
        '''
            Generate tracking error
        '''
        self.tra_diff = self.gamma * self.pos_diff + self.vel_diff
        #print("Velocity error: ", self.vel_diff)
    
    def mass_spring_damper(self):
        '''
            Simple mass-spring-damper system for simulation of controller
        '''
        import numpy as np
        
        self.spring_force = self.spring * self.pos
        self.damper_force = self.damper * self.vel
        self.noise = np.random.normal(0, 0.2, self.dof) # Simulate sensor noise
        self.acceleration = -(self.spring_force + self.damper_force) / self.mass + self.noise + self.tau
        #self.acceleration = -(self.spring_force + self.damper_force) / self.mass + self.noise
        # + or - self.tau in the equation above?
        self.vel = self.vel + self.acceleration * self.dt
        self.pos = self.pos + self.vel * self.dt
        
        #print("Mass of system: ", self.mass)
        #print("Spring force:", self.spring_force)
        #print("Damper force: ", self.damper_force)
        #print("Sensor noise: ", self.noise)
        #print("Iteration counter: ", self.counter)
        #print("Acceleration of system: ", self.acceleration)
        #print("Control output: ", self.tau) 
        #print("Velocity of system: ", self.vel)
        #print("Position of system: ", self.pos)
        
    def q_to_eul(self, q, dq, q_des, dq_des):
        '''
            Convert a quaternion into euler angles
            and quaternion rate to euler angle rate (unsure about the rates)

            q: Current orientation
            dq: Current orientation rate
            q_des: Desired orientation
            dq_des: Desired orientation rate
        '''
        from scipy.spatial.transform import Rotation as R
        import numpy as np

        self.rot = R.from_quat(q)
        self.drot = R.from_quat(dq)

        self.rot_des = R.from_quat(q_des)
        self.drot_des = R.from_quat(dq_des)

        # Check if orientation 'zyx' can be used or if 'zyz' is more appropriate
        self.rot = self.rot.as_euler('zyx', degrees=True)
        self.drot = self.drot.as_euler('zyx', degrees=True)
        self.rot_des = self.rot_des.as_euler('zyx', degrees=True)
        self.drot_des = self.drot_des.as_euler('zyx', degrees=True)

        for i in range(3):
            self.pos_des[i + 3] = self.rot_des[i]
            self.vel_des[i + 3] = self.drot_des[i]
            self.pos[i + 3] = self.rot[i]
            self.vel[i + 3] = self.drot[i]

    def iter_learn(self):
        '''
            Update gains iteratively through trials
        '''
        import numpy as np

        self.counter = 0
        
        for i in range(self.nt):
            
            # Reset velocity and position for desired values for simulation
            self.pos_old = np.array([0, 0, 0, 0, 0, 0])
            self.vel_old = np.array([0, 0, 0, 0, 0, 0])

            # Step for simulation - one full cycle for cosine / sine per trial
            self.rad = 2 * np.pi / self.samples

            # Set initial values for simple simulation of mass-spring-damper system
            self.pos = np.array([1, 1, 1, 1, 1, 1])
            #self.pos = np.zeros(self.dof)
            self.vel = np.zeros(self.dof)
            #self.vel = np.array([1, 1, 1, 1, 1, 1])
            self.mass = 1
            self.spring = 2.5
            self.damper = 0.3

            for j in range(self.samples):

                self.counter = self.counter + 1

                # Get position and velocity of system for simulation
                self.mass_spring_damper()

                # Generate some motion for self.pos_des and self.vel_des!
                self.pos_des = np.cos(self.rad * j)
                '''
                self.pos_des[0] = np.cos(self.rad * j)
                self.pos_des[1] = np.cos(self.rad * j)
                self.pos_des[2] = np.cos(self.rad * j)
                self.pos_des[3] = np.cos(self.rad * j)
                self.pos_des[4] = np.cos(self.rad * j)
                self.pos_des[5] = np.cos(self.rad * j)
                '''
                self.vel_des = (self.pos_des - self.pos_old) / self.dt

                #self.pos_des[0],self.pos_des[1],self.pos_des[2] = pos[j, 0],pos[j, 1],pos[j, 2] 
                #self.vel_des[0],self.vel_des[1],self.vel_des[2] = vel_p[j, 0],vel_p[j, 1],vel_p[j, 2] 
                
                # Apparently a huge desired velocity for the first sample
                # --> very huge velocity error --> program terminates
                # Setting the desired velocity at sample 0 to 0 to avoid this
                if j == 0:
                    self.vel_des = 0 #But WHY GODDAMNIT! Ask supervisor

                g = 0.75 # Need from DMP!

                # Get new actual position and velocity!
                # Need DMP output

                # Get position, velocity, and tracking errors
                self.get_pos_diff()
                self.get_vel_diff()
                self.get_tra_diff()

                # Update gains
                if i == 0:
                    self.ks = self.qs * self.tra_diff * self.pos_diff * g
                    self.kd = self.qd * self.tra_diff * self.vel_diff * g
                    self.v = self.qv * self.tra_diff * g
                    #print("Spring: ", self.ks)
                    #print("Damper: ", self.kd)
                    #print("Debugging: ", self.vel_diff)
                else:
                    self.ks = self.ks_collect[i - 1][:, j] + self.qs * self.tra_err_collect[i - 1][:, j] * self.pos_err_collect[i - 1][:, j] * g
                    self.kd = self.kd_collect[i - 1][:, j] + self.qd * self.tra_err_collect[i - 1][:, j] * self.vel_err_collect[i - 1][:, j] * g
                    self.v = self.v_collect[i - 1][:, j] + self.qv * self.tra_err_collect[i - 1][:, j] * g
                    #print("New spring: ", self.ks)
                    #print("New damper: ", self.kd)
                
                # Combine gains into torque - check paper for correct formula!
                self.tau = -(self.ks * self.pos_diff + self.kd * self.vel_diff) - self.v
                #self.tau = -(self.ks * self.pos_diff + self.kd * self.vel_diff)
                # Tau works "fine" with only spring gain
                #self.tau = -(self.ks * self.pos_diff)
                # Code does not run with only damper gain...
                # Code runs now after setting vel_des at sample 0 to 0
                #self.tau = -(self.kd * self.vel_diff)
                #print("Output: ", self.tau)

                # Collect data for plotting and gain update after trial 0
                for k in range(self.dof):
                    self.pos_collect[i][k][j] = self.pos[k]
                    self.vel_collect[i][k][j] = self.vel[k]
                    self.pos_err_collect[i][k][j] = self.pos_diff[k]
                    self.vel_err_collect[i][k][j] = self.vel_diff[k]
                    self.tra_err_collect[i][k][j] = self.tra_diff[k]
                    self.tau_collect[i][k][j] = self.tau[k]
                    self.ks_collect[i][k][j] = self.ks[k]
                    self.kd_collect[i][k][j] = self.kd[k]
                    self.v_collect[i][k][j] = self.v[k]
                    #self.pos_des_collect[i][k][j] = self.pos_des[k]
                    #self.vel_des_collect[i][k][j] = self.vel_des[k]
                
                self.pos_old = self.pos_des

if __name__ == '__main__':    
    import matplotlib.pyplot as plt
    import numpy as np
    
    tr = 30 # Number of trials
    t = 10 # Total time of each trial
    t_interval = np.arange(0, t + 0.05, 0.05)

    ac = Ada_con(dof=6, tr=tr, t=t)
    ac.iter_learn()
    
    #print(ac.v_collect)
    #print(np.shape(ac.pos_err_collect))
    #print(ac.ks_collect[0])
    #print(ac.ks_collect[0][:,0])
    '''
    A = np.zeros((2, 6, 5))
    A[0][0][0] = 1
    A[0][1][0] = 2
    A[0][2][0] = 3
    A[0][3][0] = 4
    A[0][4][0] = 5
    A[0][5][0] = 6
    print(A)
    print("Please work!: ", A[0][:,0])
    B = A[0]
    print("Try: ", B[:,0])
    #print(ac.pos_collect)
    #print(ac.pos_collect[0][0])
    #print(np.shape(ac.pos_collect))
    #print("Gamma: ", ac.gamma)
    '''

    #print(np.shape(ac.pos_collect))
    '''
    # Plot position error of first and last trial
    fig, axs = plt.subplots(2)
    fig.suptitle('Simulation')
    axs[0].plot(t_interval, ac.pos_err_collect[0][0])
    axs[1].plot(t_interval, ac.pos_err_collect[tr - 1][0])
    plt.show()
    '''
    '''
    # Plot velocity error of first and last trial
    fig, axs = plt.subplots(2)
    fig.suptitle('Simulation')
    axs[0].plot(t_interval, ac.vel_err_collect[0][0])
    axs[1].plot(t_interval, ac.vel_err_collect[tr - 1][0])
    plt.show()
    '''
    '''
    # Plot controller output of first and last trial
    fig, axs = plt.subplots(2)
    fig.suptitle('Simulation')
    axs[0].plot(t_interval, ac.tau_collect[0][0])
    axs[1].plot(t_interval, ac.tau_collect[tr - 1][0])
    plt.show()
    '''
    # Plot position of system of first and last trial

    #print(ac.pos_collect)
    
    fig, axs = plt.subplots(2)
    fig.suptitle('Simulation')
    axs[0].plot(t_interval, ac.pos_collect[0][0])
    axs[1].plot(t_interval, ac.pos_collect[tr - 1][0])
    plt.show()
    

    #plt.plot(t_interval, ac.pos_collect[0][0])
    #plt.show()


    # Plot values for the simulation!






