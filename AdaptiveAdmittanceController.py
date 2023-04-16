class Ada_con():
    
    def __init__(self, dof):

        import time
        from datetime import datetime
        import itertools
        import numpy.random as npr
        import numpy as np
        import matplotlib.pyplot as plt
        #from pDMP_functions import pDMP

        # Number of learning trials
        self.nt = 2 # Value chosen arbitrarily

        # Number of samples of trajectory per trial
        self.time = 1 # Total experiment time - chosen arbitrarily
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

        # Controller output
        self.tau = np.zeros(self.dof)

        # Adaptive stiffness and damping
        self.ks = np.zeros(self.dof)
        self.kd = np.zeros(self.dof)

        # Adaptation rates for stiffness and damping
        self.qs = 5 # Value chosen arbitrarily
        self.qd = 5 # Value chosen arbitrarily

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
        import time
        from datetime import datetime
        import itertools
        import numpy.random as npr
        import numpy as np
        import matplotlib.pyplot as plt

        self.pos_diff = np.subtract(self.pos, self.pos_des)
    
    def get_vel_diff(self):
        '''
            Generate velocity error
        '''
        import time
        from datetime import datetime
        import itertools
        import numpy.random as npr
        import numpy as np
        import matplotlib.pyplot as plt

        self.vel_diff = np.subtract(self.vel, self.vel_des)

    def get_tra_diff(self):
        '''
            Generate tracking error
        '''
        self.tra_diff = self.gamma * self.pos_diff + self.vel_diff
    
    def mass_spring_damper(self):
        '''
            Simple mass-spring-damper system for simulation of controller
        '''
        import time
        from datetime import datetime
        import itertools
        import numpy.random as npr
        import numpy as np
        import matplotlib.pyplot as plt

        self.spring_force = self.spring * self.pos
        self.damper_force = self.damper * self.vel
        self.noise = np.random.normal(0, 0.2, self.dof) # Simulate sensor noise
        self.acceleration = - (self.spring_force + self.damper_force) / self.mass + self.noise + self.tau
        # + or - self.tau in the equation above?
        self.vel = self.vel + self.acceleration * self.dt
        self.pos = self.pos + self.vel * self.dt
        
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
        import time
        from datetime import datetime
        import itertools
        import numpy.random as npr
        import numpy as np
        import matplotlib.pyplot as plt
        
        for i in range(self.nt):
            
            # Reset velocity and position for desired values for simulation
            self.pos_old = np.array([0, 0, 0, 0, 0, 0])
            self.vel_old = np.array([0, 0, 0, 0, 0, 0])

            # Step for simulation - one full cycle for cosine / sine per trial
            self.rad = 2 * np.pi / self.samples

            # Set initial values for simple simulation of mass-spring-damper system
            self.pos = np.array([1, 2, 3, 4, 5, 6])
            self.vel = np.zeros(self.dof)
            self.mass = 1
            self.spring = 2.5
            self.damper = 0.3

            for j in range(self.samples):

                # Get position and velocity of system for simulation
                self.mass_spring_damper()

                # Generate some motion for self.pos_des and self.vel_des!
                self.pos_des = np.cos(self.rad * j)
                self.vel_des = (self.pos_des - self.pos_old) / self.dt

                g = 0.75 # Need from DMP!

                # Get new actual position and velocity!
                # Need DMP output

                # Get position, velocity, and tracking errors
                self.get_pos_diff()
                self.get_vel_diff()
                self.get_tra_diff

                # Update gains
                self.ks = self.ks + self.qs * self.tra_diff * self.pos_diff * g
                self.kd = self.kd + self.qd * self.tra_diff * self.vel_diff * g

                # Combine gains into torque - check paper for correct formula!
                self.tau = -(self.ks * self.pos_diff + self.kd * self.vel_diff)

                # Collect data for plotting
                for k in range(self.dof):
                    self.pos_collect[i][k][j] = self.pos[k]
                    self.vel_collect[i][k][j] = self.vel[k]
                    self.pos_err_collect[i][k][j] = self.pos_diff[k]
                    self.vel_err_collect[i][k][j] = self.vel_diff[k]
                    self.tra_err_collect[i][k][j] = self.tra_diff[k]
                
                self.pos_old = self.pos_des

if __name__ == '__main__':    
    ac = Ada_con(dof = 6)
    ac.iter_learn()
    print(ac.pos_collect)
    # Plot values for the simulation!






