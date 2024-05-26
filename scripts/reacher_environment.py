"""
Addapted from: https://github.com/tum-i6/VTPRL/blob/main/agent/iiwa_sample_joint_vel_env.py
"""
import math

import numpy as np
import pandas as pd

from gym import spaces, core
from gym.utils import seeding

# the max velocity allowed for a joint in radians
MAX_VEL = 1.1 

# angle limits in radians
MAX_ANGLE_1 = 3
MAX_ANGLE_2 = 2.1

# max distance between end-effector and target
MAX_DISTANCE = 2

# timestep duration in seconds
DELTA = 0.02

# max allowed latency in seconds when latency randomization is enabled
MAX_LATENCY = 1

# max noise range when noise randomization is enabled
MAX_NOISE = 0.1

class ReacherEnvironment(core.Env):

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

        # dx, dy, dz placeholders
        state_array_limits = [1., 1., 1.]
        # angle placeholders
        state_array_limits.extend([1] * self.num_joints)

        high = np.array(state_array_limits)
        low = -high

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.observation_size = len(state_array_limits)

        high = np.array([1] * self.num_joints)
        low = -high
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.previous_latency = 0
        self.state_history = [None for _ in range(self.max_ts + 1)]
        self.distance_history = np.zeros(self.max_ts + 1)
        self.action_history = [None for _ in range(self.max_ts + 1)]
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

        self._convert_observation(observation['Observation'])
        self.ts += 1

        info = {}
        terminal = self._get_terminal()
        reward = self._get_reward()

        if self.ts > self.max_ts:
            terminal = True

        return self.state, reward, terminal, info

    def _get_reward(self):

        absolute_distance = self._get_distance()
        reward = - 1 * absolute_distance
        if self.collided:
            reward = reward * (self.max_ts - self.ts + 1)
        return reward

    def _convert_observation(self, new_observation):

        # the last value from the simulator indicates whether the manipulator collided with the floor
        self.collided = 1 if new_observation[-1] > 0 else 0
        new_observation = new_observation[:-1]

        self.simulator_observation = new_observation

        self.target_x, self.target_y, self.target_z, self.target_d1, self.target_d2, self.target_d3 = \
            self._get_target_coordinates()
        self.object_x, self.object_y, self.object_z, self.object_d1, self.object_d2, self.object_d3 = \
            self._get_object_coordinates()
        self.ee_x, self.ee_y, self.ee_z, self.ee_d1, self.ee_d2, self.ee_d3 = self._get_end_effector_coordinates()
        self.joint_angles = self.simulator_observation[0:self.num_joints]
        self.joint_speeds = self.simulator_observation[7:7 + self.num_joints]

        dx = (self.target_x - self.ee_x) / MAX_DISTANCE
        dy = (self.target_y - self.ee_y) / MAX_DISTANCE
        dz = (self.target_z - self.ee_z) / MAX_DISTANCE

        self.state = [dx, dy, dz]
        for i in range(self.num_joints):
            # add the angles for the enabled joints
            if i % 2 == 0:
                self.state.append(self.simulator_observation[i] / MAX_ANGLE_1)
            else:
                self.state.append(self.simulator_observation[i] / MAX_ANGLE_2)

        self.state_history[self.ts] = self.state
        self.distance_history[self.ts] = self._get_distance()      

        if self.random_latency and self.ts > 0:
            # only enter if ts > 0, because on reset it should see the actual state
            latency = self.random_latencies[self.ts][0]
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


        if self.random_noise:
            self.state = np.array(self.state) + self.random_noises[self.ts]
            self.state = self.state.clip(-1, 1).tolist()


    def _get_end_effector_coordinates(self):
        return \
            self.simulator_observation[-18], self.simulator_observation[-17], self.simulator_observation[-16], \
            self.simulator_observation[-15], self.simulator_observation[-14], self.simulator_observation[-13]


    def _get_object_coordinates(self):
        return \
            self.simulator_observation[-6], self.simulator_observation[-5], self.simulator_observation[-4], \
            self.simulator_observation[-3], self.simulator_observation[-2], self.simulator_observation[-1]


    def _get_target_coordinates(self):
        return \
            self.simulator_observation[-12], self.simulator_observation[-11], self.simulator_observation[-10], \
            self.simulator_observation[-9], self.simulator_observation[-8], self.simulator_observation[-7]


    def _get_distance(self):

        return \
            np.linalg.norm(np.array([self.target_x - self.ee_x, self.target_y - self.ee_y, self.target_z - self.ee_z]))


    def step(self):
        pass


    def update_action(self, action):
        self.action_history[self.ts] = action


    def continuity_cost(self):
        # based on: Smooth Exploration for Robotic Reinforcement Learning paper
        deltas = [None for _ in range(self.max_ts-1)]
        for i in range(1, self.max_ts):
            deltas[i-1] = np.mean(np.power(self.action_history[i] - self.action_history[i+1], 2) / 4)
        deltas = np.array(deltas)
        return 100 * np.mean(deltas)


    def distance_mean_std(self):
        return np.mean(self.distance_history[250:]), np.std(self.distance_history[250:])
 

    def reset(self):

        self.ts = 0
        self.collided = 0

        self.target_x = 0.387046
        self.target_y = 0.588643
        self.target_z = -0.861801

        if self.mode == 1:
            self.random_noises = np.random.uniform(-MAX_NOISE, MAX_NOISE, (self.max_ts + 1, self.observation_size))
            self.random_latencies = np.random.uniform(0, MAX_LATENCY, (self.max_ts + 1, 1))
            self.random_torques = np.random.uniform(10, 10_000, (self.max_ts + 1, self.num_joints)) 

        else:
            self.random_noises = self.np_random.uniform(-MAX_NOISE, MAX_NOISE, (self.max_ts + 1, self.observation_size))
            self.random_latencies = self.np_random.uniform(0, MAX_LATENCY, (self.max_ts + 1, 1))
            self.random_torques = self.np_random.uniform(10, 10_000, (self.max_ts + 1, self.num_joints))

        # the object is not relevant for the task, spawn it under the floor
        self.object_x = 0
        self.object_y = -1
        self.object_z = 0

        # the joints to be enabled have value 1 and the others have value 0
        self.reset_state = [1] * self.num_joints + [0] * (7 - self.num_joints)

        self.reset_state.extend([
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
            self.target_x, self.target_y, self.target_z,
            0, 0, 0,
            self.object_x, self.object_y, self.object_z,
            0, 0, 0])

        self.reset_counter += 1

    def _get_terminal(self):
        if self.collided:
            return True
        if self.mode == 0 and (abs(self.joint_angles[0]) > 2.61799 or abs(self.joint_angles[1]) > 1.74533):
            print("joint limits reached during evaluation")
            print(self.id)
            self.collided = True
            return True
        return False

