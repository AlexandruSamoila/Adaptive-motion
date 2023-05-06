import math
import numpy as np
from Phase_variable import PhaseVariable


class Ada_con():

    def __init__(self, dof, tr, t):
        import numpy as np
        # from pDMP_functions import pDMP

        # Number of learning trials
        self.nt = tr  # Value chosen arbitrarily
        # Number of samples of trajectory per trial
        self.time = len(t)  # Total experiment time - chosen arbitrarily
        self.dt = 0.002  # Time step - chosen arbitrarily
        self.samples = len(t)  # int(1/self.dt) * self.time + 1

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
        self.qs = 1.5  # Value chosen arbitrarily
        self.qd = 1.5  # Value chosen arbitrarily

        # Adaptation rate for feedforward term
        self.qv = 15.3  # Value chosen arbitrarily

        # Tracking error coefficient
        self.gamma = 4  # Value chosen arbitrarily

        # Forgetting factor
        self.lambd = 0.1  # Chosen arbitrarily

        self.pos_old = np.zeros(self.dof)  # Used for calculating desired velocity
        self.vel_old = np.zeros(self.dof)  # Used for calculating desired acceleration

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
        # print("Velocity error: ", self.vel_diff)
        # print("Desired vel: ", self.vel_des)
        # print("Actual vel: ", self.vel)

    def get_tra_diff(self):
        '''
            Generate tracking error
        '''
        self.tra_diff = self.gamma * self.pos_diff + self.vel_diff
        # print("Velocity error: ", self.vel_diff)

    def mass_spring_damper(self):
        '''
            Simple mass-spring-damper system for simulation of controller
        '''
        import numpy as np

        self.spring_force = self.ks * self.pos_diff
        self.damper_force = self.kd * self.vel_diff
        self.noise = np.random.normal(0, 0, self.dof)  # Simulate sensor noise
        self.acceleration = (self.force - self.spring_force - self.damper_force) / self.mass + self.actual_acceleration
        # self.acceleration = -(self.spring_force + self.damper_force) / self.mass + self.noise
        # + or - self.tau in the equation above?

        self.vel = self.vel + self.acceleration * self.dt
        self.pos = self.pos + self.vel * self.dt

        # print("Mass of system: ", self.mass)
        # print("Spring force:", self.spring_force)
        # print("Damper force: ", self.damper_force)
        # print("Sensor noise: ", self.noise)
        # print("Iteration counter: ", self.counter)
        # print("Acceleration of system: ", self.acceleration)
        # print("Control output: ", self.tau)
        # print("Velocity of system: ", self.vel)
        # print("Position of system: ", self.pos)

    def q_to_eul(self, q, dq, q_des, dq_des):
        '''

        TODO:

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

    def radialBasis(self, alpha, n_bfs):

        self.alpha = alpha

        # No. of basis functions
        self.n_bfs = n_bfs

        self.pv = PhaseVariable()

        # Centres of the Gaussian basis functions
        self.c = np.exp(self.alpha * np.linspace(0, 1, self.n_bfs))

        # Variance of the Gaussian basis functions
        self.h = 1.0 / np.gradient(self.c) ** 2

        tau = 0.002 * self.samples
        ts = np.arange(0, tau, 0.002)

        s = self.pv.rollout(ts)  # Integrate canonical system

        def features(xj):
            w = np.exp(-0.5 * self.h * (xj - self.c) ** 2)
            return w / w.sum()

        g = np.stack([features(xj) for xj in s])

        return g

    def iter_learn(self, pos, vel_p, ks, kd, v, force, actual_acceleration):
        '''
            Update gains iteratively through trials
        '''

        self.counter = 0

        alpha = 44
        n_bfs = 8

        g = self.radialBasis(alpha=alpha, n_bfs=n_bfs)

        for i in range(self.nt):

            # Reset velocity and position for desired values for simulation
            self.pos_old[0], self.pos_old[1], self.pos_old[2] = 0, 0, 0
            self.vel_old[0], self.vel_old[1], self.vel_old[2] = 0, 0, 0

            # Step for simulation - one full cycle for cosine / sine per trial
            self.rad = 2 * np.pi / self.samples

            # Set initial values for simple simulation of mass-spring-damper system
            self.pos[0], self.pos[1], self.pos[2] = pos[0, 0], pos[0, 1], pos[0, 2]
            # self.pos = np.zeros(self.dof)
            self.vel[0], self.vel[1], self.vel[2] = vel_p[0, 0], vel_p[0, 1], vel_p[0, 2]
            # self.vel = np.array([1, 1, 1, 1, 1, 1])

            """
                TODO: Try modify this parameters to simulate divergent force.
            """
            self.mass = 1
            self.spring = 5
            self.damper = 2 * math.sqrt(self.mass * (self.spring+1))

            for j in range(self.samples):

                self.actual_acceleration=actual_acceleration[j,:]
                self.counter = self.counter + 1

                # Get position and velocity of system for simulation
                #self.tau = force[i, :]
                self.force = force[j, :]

                self.pos_des[0], self.pos_des[1], self.pos_des[2] = pos[j, 0], pos[j, 1], pos[j, 2]
                self.vel_des[0], self.vel_des[1], self.vel_des[2] = vel_p[j, 0], vel_p[j, 1], vel_p[j, 2]

                # Get position, velocity, and tracking errors
                self.get_pos_diff()
                self.get_vel_diff()
                self.get_tra_diff()
                self.mass_spring_damper()
                # Update gains
                if i == 0:
                    self.ks = ks[j, :]
                    self.kd = kd[j, :]
                    self.v = v[j, :]
                    # print("Spring: ", self.ks)
                    # print("Damper: ", self.kd)
                    # print("Debugging: ", self.vel_diff)
                    self.tau = force[j, :]
                else:
                    self.ks = (self.ks_collect[i - 1][:, j] + self.qs * self.tra_err_collect[i - 1][:, j] * \
                              self.pos_err_collect[i - 1][:, j] * g[j, self.n_bfs - self.dof:self.n_bfs])
                    self.kd = (self.kd_collect[i - 1][:, j] + self.qd * self.tra_err_collect[i - 1][:, j] * \
                              self.vel_err_collect[i - 1][:, j] * g[j, self.n_bfs - self.dof:self.n_bfs])
                    self.v = (self.v_collect[i - 1][:, j] + self.qv * self.tra_err_collect[i - 1][:, j] * g[j,
                                                                                                         self.n_bfs - self.dof:self.n_bfs])
                    # print("New spring: ", self.ks)
                    # print("New damper: ", self.kd)

                    # Combine gains into torque - check paper for correct formula!


                    self.tau = -(self.ks * self.pos_diff + self.kd * self.vel) - self.v

                # self.tau = -(self.ks * self.pos_diff + self.kd * self.vel_diff)
                # Tau works "fine" with only spring gain
                # self.tau = -(self.ks * self.pos_diff)
                # Code does not run with only damper gain...
                # Code runs now after setting vel_des at sample 0 to 0
                # self.tau = -(self.kd * self.vel_diff)
                # print("Output: ", self.tau)

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
                    # self.pos_des_collect[i][k][j] = self.pos_des[k]
                    # self.vel_des_collect[i][k][j] = self.vel_des[k]

                self.pos_old = self.pos_des