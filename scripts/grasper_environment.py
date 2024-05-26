"""
Addapted from: https://github.com/tum-i6/VTPRL/blob/main/agent/iiwa_sample_joint_vel_env.py

A sample Env class inheriting from basic gym.Env  for the Kuka LBR iiwa manipulator with 7 links and a Gripper.
Unity-based simulator is used as the main simulator for physics/rendering computations. The Unity interface
receives joint velocities as commands and returns joint positions and velocities.
"""

import math
import numpy as np
from gym import spaces, core
from gym.utils import seeding

# the max velocity allowed for a joint in radians
MAX_VEL = 1.1  # 1/8 * np.pi

# angle limits in radians
MAX_ANGLE_1 = 3
MAX_ANGLE_2 = 2.1

# max distance between end-effector and target
MAX_DISTANCE = 2

# timestep duration in seconds
DELTA = 0.05

# max allowed latency in seconds when latency randomization is enabled
MAX_LATENCY = 1

# max noise range when noise randomization is enabled
MAX_NOISE = 0.1


class GrasperEnvironment(core.Env):
    
    def __init__(self, id, max_ts, config):
        self.max_ts = max_ts
        self.ENV_KEY = "manipulator_environment"
        self.state_type = config["state"]

        self.random_noise = config["random_noise"]
        self.random_latency = config["random_latency"]
        self.mode = config["mode"]
        self.working_dir = config["working_dir"]
        self.simulator_observation = None

        self.num_joints = config["number_joints"]

        # position and orientaton deltas placeholders
        state_array_limits = [1., 1., 1., 1., 1., 1., 1., 1., 1.]
        # joint angle placeholders
        state_array_limits.extend([1] * self.num_joints)

        self.observation_size = len(state_array_limits)

        high = np.array(state_array_limits)
        low = -high

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        high = np.array([1] * self.num_joints)
        low = -high
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.previous_latency = 0
        self.state_history = [None for _ in range(self.max_ts + 1)]
        self.id = id
        self.reset_counter = 0
        self.reset_state = None
        self.ts = 0
        self.collided = 0
        self.seed()

        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def update(self, observation):
        """
        a replacement of the standard step() method used in OpenAI gym. Unlike typical gym environment where the step()
        function is called at each timestep, in this case the simulator_vec_env.py step() function is called during
        execution, which in turn calls the update() function here for a single env to pass its simulator observation.
        Reasons are related to the logic of how to communicate with the simulator and how to reset individual
        environments running in the simulator independently.
        """
        self._convert_observation(observation['Observation'])
        self.ts += 1

        info = {}
        terminal = self._get_terminal()
        reward = self._get_reward()  # if not terminal else 10

        if self.ts > self.max_ts:
            terminal = True

        return self.state, reward, terminal, info


    def _get_reward(self):
        
        absolute_distance = self._get_distance()
        
        if self.simulator_observation[33] > 0.3:
            print("object lifted at env: " + str(self.id))
            self.object_grasped = 1

        return -absolute_distance + (10 if self.simulator_observation[33] > 0.3 else 0) - self.collided

    def _convert_observation(self, new_observation):
        #input()
        """
        method used for creating task-specific agent state from the generic simulator observation returned.
        The simulator observation has the following array of 34 values:
        [a1, a2, a3, a4, a5, a6, a7,
        v1, v2, v3, v4, v5, v6, v7,
        ee_x, ee_y, ee_z ee_fx, ee_fy, ee_fz, ee_ux, ee_uy, ee_uz,
        t_x, t_y, t_z, t_fx, t_fy, t_fz, t_ux, t_uy, t_uz,
        o_x, o_y, o_z, o_fx, o_fy, o_fz, o_ux, o_uy, o_uz,
        g_p,
        c]
        where a1..a7 are the angles of each joint of the robot in radians, v1..v7 are the velocities of each joint
        in rad/sec, x, y, and z for ee, t and o are the coordinates and and f/u_x, f/u_y, and f/u_z for ee, t and o are the
        unit forward and up direction vectors for x, y and z components of the rotation for the end-effector, the target and the object(box)
        respectively. g_p is the position (opening) of the gripper and c is a collision flag (0 if no collision and 1
        if a collision of any part of the robot with the floor happened)
        """
        self.simulator_observation = new_observation

        self.collided = 1 if (self.collided > 0 or self.simulator_observation[-1] > 0) else 0
        if self.simulator_observation[-1] > 0:
            self.collision_happened = 1
        self.gripper_position = self.simulator_observation[-2]

        self.target_x, self.target_y, self.target_z, \
            self.target_fx, self.target_fy, self.target_fz, \
            self.target_ux, self.target_uy, self.target_uz, = self._get_target_pose()
        self.object_x, self.object_y, self.object_z, \
            self.object_fx, self.object_fy, self.object_fz,\
            self.object_ux, self.object_uy, self.object_uz = self._get_object_pose()
        
        self.ee_x, self.ee_y, self.ee_z, \
            self.ee_fx, self.ee_fy, self.ee_fz,\
            self.ee_ux, self.ee_uy, self.ee_uz,\
            = self._get_end_effector_pose()
        self.joint_angles = self.simulator_observation[0 : self.num_joints]
        self.joint_speeds = self.simulator_observation[7 : 7 + self.num_joints]

        self.dx = (self.object_x - self.ee_x) / MAX_DISTANCE
        self.dy = (self.object_y - self.ee_y) / MAX_DISTANCE
        self.dz = (self.object_z - self.ee_z) / MAX_DISTANCE
        self.dfx =(self.object_fx - self.ee_fx) / MAX_DISTANCE
        self.dfy =(self.object_fy - self.ee_fy) / MAX_DISTANCE
        self.dfz =(self.object_fz - self.ee_fz) / MAX_DISTANCE
        self.dux = self.ee_ux / MAX_DISTANCE 
        self.duy = (1 - self.ee_uy) / MAX_DISTANCE 
        self.duz = self.ee_uz / MAX_DISTANCE

        
        if (abs(self.joint_angles[0]) > 2.61799 or 
            abs(self.joint_angles[1]) > 1.74533 or
            abs(self.joint_angles[2]) > 2.61799 or 
            abs(self.joint_angles[3]) > 1.74533 or
            abs(self.joint_angles[4]) > 2.61799 or 
            abs(self.joint_angles[5]) > 1.74533 or
            abs(self.joint_angles[6]) > 2.61799):
                #print("joint limits reached")
                self.collided = 1
                self.joint_limits_reached = 1

        if(self.ee_y) <= 0.05:
            self.collided = 1
            print("safety limits reached") 

        self.state = [self.dx, self.dy, self.dz, self.dfx, self.dfy, self.dfz, self.dux, self.duy, self.duz]
        for i in range(self.num_joints):
            # add the angles for the enabled joints
            if i % 2 == 0:
                self.state.append(self.simulator_observation[i] / MAX_ANGLE_1)
            else:
                self.state.append(self.simulator_observation[i] / MAX_ANGLE_2)

        if self.state_type == 'av':
            for i in range(self.num_joints):
                # the second 7 values are joint velocities, if velocities are part of the state include them
                self.state.append(self.simulator_observation[7 + i] / MAX_VEL)

        self.state_history[self.ts] = self.state

        if self.random_latency and self.ts > 0 and self.ts < 200:
            # only enter if ts > 0, because on reset it should see the actual state
            latency = self.random_latencies[self.ts][0] #self.np_random.uniform(0, MAX_LATENCY)
            # latency can't be greater than the elapsed time
            latency = min(self.ts * DELTA, latency)
            # latency can't be greater than the last latency and the delta time since then
            latency = min(self.previous_latency + DELTA, latency)
            # store the latest latency for the next step
            self.previous_latency = latency

            if latency == 0:
                lat_state = self.state
            else:
                # the latency is a linear interpolation between the two discrete states that correspond
                ratio = round(latency/DELTA, 6)
                earlier_state = self.state_history[self.ts-math.ceil(ratio)]
                earlier_state_portion = (ratio-math.floor(ratio))
                later_state = self.state_history[self.ts-math.floor(ratio)]
                try:
                    lat_state = [x * earlier_state_portion + y * (1 - earlier_state_portion) for x, y in zip(
                        earlier_state, later_state
                    )]
                    self.state = lat_state
                except:
                    print("Error happened")

        if self.random_noise and self.ts < 200:
            self.state = np.array(self.state) + self.random_noises[self.ts]
            self.state = self.state.clip(-1, 1).tolist()


    def _get_end_effector_pose(self):
         
        # 3F gripper
        u = np.cross([self.simulator_observation[17], self.simulator_observation[18], self.simulator_observation[19]], 
                    [self.simulator_observation[20], self.simulator_observation[21], self.simulator_observation[22]])
        f = [self.simulator_observation[20], self.simulator_observation[21], self.simulator_observation[22]]
            
        return \
            self.simulator_observation[14], self.simulator_observation[15], self.simulator_observation[16], \
            f[0], f[1], f[2], \
            u[0], u[1], u[2]
    

    def _get_object_pose(self):
        # fixed pose
        return 0.4, 0.225, 0.4, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0
            

    def _get_target_pose(self):
        return \
            self.simulator_observation[23], self.simulator_observation[24], self.simulator_observation[25], \
            self.simulator_observation[26], self.simulator_observation[27], self.simulator_observation[28], \
            self.simulator_observation[29], self.simulator_observation[30], self.simulator_observation[31]

    def _get_distance(self):

        dist_position = np.linalg.norm(np.array([self.object_x - self.ee_x, self.object_y - self.ee_y, self.object_z - self.ee_z]))
        dist_rotation_up = np.linalg.norm(np.array([self.ee_ux, 1 + self.ee_uy, self.ee_uz]))
        return dist_position + 0.5 * (dist_rotation_up)

    def step(self):
        """
        not used directly on level on single environment, see update() method instead
        """
        pass

    def reset(self):
        """
        reset method that can set the environment in the simulator in a specific initial state and enable/disable joints
        The method defines the self.reset_state that is used from simulator_vec_env to send the reset values to the
        Unity simulator. The simulator reset state has the following array of 32 values:
        [j1, j2, 23, j4, j5, j6, j7,
        a1, a2, a3, a4, a5, a6, a7,
        v1, v2, v3, v4, v5, v6, v7,
        t_x, t_y, t_z, t_rx, t_ry, t_rz,
        o_x, o_y, o_z, o_rx, o_ry, o_rz,
        g_p]
        where j1..j7 are flags indicating whether a joint should be enabled (1) or disabled (0),
        a1..a7 are the initial angles of each joint of the robot in radians (currently only 0 initial values supported
        due to unity limitations)
        v1..v7 are the initial velocities of each joint in rad/sec (currently only 0 initial values supported due to
        unity limitations)
        x, y, and z for t and o are the initial coordinates of the target and the object in meters (note that y is
        vertical axis in unity)
        rx, ry, and rz for t and o are the euler angles in degrees for the rotation of the object and the target
        g_p is the position (opening) of the gripper (0 is open, value up to 90 supported)
        """
        self.reset_counter += 1

        self.ts = 0
        self.collided = 0
        self.gripper_action = 0

        self.dx = 0
        self.dy = 0
        self.dz = 0
        self.dfx = 0
        self.dfy = 0
        self.dfz = 0
        self.dux = 0
        self.duy = 0
        self.duz = 0

        if self.mode == 0:
            # evaluation mode, reproducable randomizations
            self.random_noises = self.np_random.uniform(-MAX_NOISE, MAX_NOISE, (self.max_ts, self.observation_size))
            self.random_latencies = self.np_random.uniform(0, MAX_LATENCY, (self.max_ts, 1))
            self.random_torques = self.np_random.uniform(10, 10_000, (self.max_ts, self.num_joints))            
        else:
            self.random_noises = np.random.uniform(-MAX_NOISE, MAX_NOISE, (self.max_ts, self.observation_size))
            self.random_latencies = np.random.uniform(0, MAX_LATENCY, (self.max_ts, 1))
            self.random_torques = np.random.uniform(10, 10_000, (self.max_ts, self.num_joints))   
            
        
        self.object_x = 0.4
        self.object_y = 0.225
        self.object_z = 0.4
        # the target is used as a stand for the object during grasping, not used directly
        self.target_x = 0.4
        self.target_y = 0.2-0.001/2
        self.target_z = 0.4
        
        # initial robot configuration
        joints = np.array([0.0, 0.0, 0.0, -1.64, 0.0, 1.50, 0.0])
        jointSpeeds = np.zeros(shape=(7,))

        # all joints are enabled
        self.reset_state = [1, 1, 1, 1, 1, 1, 1]
        self.reset_state.extend(joints)
        self.reset_state.extend(jointSpeeds)
        # initial target and object position and orientation and gripper opening
        self.reset_state.extend([
            self.target_x, self.target_y, self.target_z, 0, 0, 0,
            self.object_x, self.object_y, self.object_z, 0, 0, 0,
            0])

    def _get_terminal(self):
        """
        a function to define terminal condition, it can be customized to define terminal conditions based on episode
        duration, whether the task was solved, whether collision or some illegal state happened
        """
        if self.ts > self.max_ts:
            return True
        return False


    def update_action(self, action):
        if self.collided:
            action [:]= 0
            self.gripper_action = 0
            return
        
        # manual grasp attempt at the end of the episode
        if 200 < self.ts < 250:
            # commands 0 for all joit veloctiies to stop robot movement
            action[:] = 0
            # closes the gripper
            self.gripper_action = (self.ts-200) * 2
        elif self.ts >= 250:
            # and rotates the joints of the robot backwards to attempt lifting the object
            action[:] = -np.array(self.joint_angles) * 2
            self.gripper_action = 100
        return action

    