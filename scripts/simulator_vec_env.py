import ast
import base64
import json
import time
import atexit

import numpy as np
import roslibpy
import sys
sys.path.append("external/stable-baselines3")
from stable_baselines3.common.vec_env import DummyVecEnv
import subprocess
from subprocess import PIPE
import zmq
import grpc
import service_pb2_grpc
from service_pb2 import StepRequest


class CommandModel:

    def __init__(self, id, command, env_key, value):
        self.id = id
        self.environment = env_key
        self.command = command
        self.value = value


class SimulatorVecEnv(DummyVecEnv):
    _client = None

    def __init__(self, env_fns, config, spaces=None ):
        """
        envs: list of environments to create
        """
        DummyVecEnv.__init__(self, env_fns)
        self.config = config
        if config["random_noise"]:
            print("randomizing noise")
        if config["random_latency"]:
            print("randomizing latency")
        if config["random_torque"]:
            print("randomizing torque")

        # comment the sub-process creation, opening and sleep below if the simulator is manually started outside of the docker container
        self.env_process = subprocess.Popen(
            [config['root_directory'] + "/simulator/VTPRL/environment/simulator/v0.92cdr/Linux/ManipulatorEnvironment/ManipulatorEnvironment.x86_64",
             "-pn", str(config["port_number"]),
             "-s", str(config["seed"]),
             "-rt", str(0 if config["random_torque"] is False else 1),
            "-logFile", config['working_dir'] + "/simulation.log",
            # uncomment the line below for headless running of the simulator, note that there might be discrepency of the results between headless and windowed mode
            #"-batchmode", "-nographics"
            ],
            stdout=PIPE, stderr=PIPE, stdin=PIPE,
            cwd=config['root_directory'] + ("/experiments/configurations/reaching" if config['experiment_type'] == 'reacher' else
            "/experiments/configurations/grasping"),
            shell=False)
        atexit.register(kill_proc, self.env_process)
        time.sleep(10)
        
        self.current_step = 0
        self.communication_type = config['communication_type']
        self.port_number = config['port_number']
        print(config["port_number"])
        self.ip_address = config['ip_address']
        self.start = 0
        self.nenvs = len(env_fns)
        self.train_envs = [env_fn(env_id=ID) for env_fn, ID in zip(env_fns, [x for x in range(self.nenvs)])]
        self.envs = self.train_envs
        
        if self.communication_type == 'ROS':
            # Connect to ROS server
            if SimulatorVecEnv._client is None:
                SimulatorVecEnv._client = roslibpy.Ros(host=self.ip_address, port=int(self.port_number))
                SimulatorVecEnv._client.run()
            self.service = roslibpy.Service(SimulatorVecEnv._client, '/step', 'rosapi/GetParam')
            self.request = roslibpy.ServiceRequest([['name', 'none'], ['default', 'none']])
        elif self.communication_type == 'ZMQ':
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.REQ)
            self.socket.connect("tcp://127.0.0.1:" + str(self.port_number))
        elif self.communication_type == 'GRPC':
            self.channel = grpc.insecure_channel(self.ip_address + ":" + str(self.port_number))
            self.stub = service_pb2_grpc.CommunicationServiceStub(self.channel)
        else:
            print("Please specify either ROS or ZMQ communication mode for this environment")


    def switch_to_training(self):
        self.envs = self.train_envs


    def switch_to_validation(self):
        self.envs = self.validation_envs


    def step(self, actions):
        self.current_step += 1
        for env, act in zip(self.envs, actions):
            env.update_action(act)
        if self.config['experiment_type'] == 'grasper': 
            while self.envs[0].max_ts - 100 < self.envs[0].ts < self.envs[0].max_ts:
                # perform close gripper and lift actions
                for env, act in zip(self.envs, actions):
                    env.ts += 1
                    env.update_action(act)
                    
                request = self._create_request("ACTION", self.envs, actions)
                # execute the simulation for all environments and get observations
                observations = self._send_request(request)
                
        #print("current step:" + str(self.current_step))
        # create request containing all environments with the actions to be executed
        request = self._create_request("ACTION", self.envs, actions)
        # execute the simulation for all environments and get observations
        observations = self._send_request(request)
        observations_converted = []
        terminated_environments = []
        rews = []
        dones = []
        infos = []
        for env, observation in zip(self.envs, observations):
            obs, rew, done, info = env.update(ast.literal_eval(observation))
            observations_converted.append(obs)
            rews.append(rew)
            dones.append(done)
            infos.append(info)
            if done:
                terminated_environments.append(env)
        # reset all the terminated environments
        [env.reset() for env in terminated_environments]
        if len(terminated_environments) > 0:
            request = self._create_request("RESET", terminated_environments)
            # currently, the simulator returns array of all the environments, not just the terminated ones
            observations = self._send_request(request)
            for env in terminated_environments:
                obs, _, _, _ = self.envs[env.id].update(ast.literal_eval(observations[env.id]))
                observations_converted[env.id] = obs
        return np.stack(observations_converted), np.stack(rews), np.stack(dones), infos


    def step_wait(self):
        # only because VecFrameStack uses step_async to provide the actions, then step_wait to execute a step
        return self.step(self.actions)


    def _create_request(self, command, environments, actions=None):
        content = ''
        if command == "ACTION":
            for act, env in zip(actions, environments):
                if self.config["experiment_type"] == "grasper":
                    act = np.append(act, [env.gripper_action], 0)
                act_json = json.dumps(CommandModel(env.id, "ACTION", env.ENV_KEY, str(act.tolist())), default=lambda x: x.__dict__)
                content += (act_json + ",")

        elif command == "RESET":
            self.start = time.time()
            for env in environments:
                reset_string = str(env.reset_state)
                act_json = json.dumps(CommandModel(env.id, "RESET", env.ENV_KEY, reset_string), default=lambda x: x.__dict__)
                content += (act_json + ",")

        return '[' + content + ']'


    def _send_request(self, content):
        if self.communication_type == 'ROS':
            self.request['name'] = content
            return self._parse_result(self.service.call(self.request))
        elif self.communication_type == 'ZMQ':
            self.socket.send_string(content)
            response = self.socket.recv()
            return self._parse_result(response)
        else:
            reply = self.stub.step(StepRequest(data=content))
            return self._parse_result(reply.data)


    def _parse_result(self, result):
        if self.communication_type == 'ROS':
            return ast.literal_eval(result['value'])
        elif self.communication_type == 'ZMQ':
            return ast.literal_eval(result.decode("utf-8"))
        else:
            return ast.literal_eval(result)


    def reset(self, should_reset=True):

        if should_reset:
            [env.reset() for env in self.envs]
        request = self._create_request("RESET", self.envs)
        observations = self._send_request(request)
        observations_converted = []
        for env, observation in zip(self.envs, observations):
            obs, _, _, _ = env.update(ast.literal_eval(observation))
            observations_converted.append(obs)
        return np.array(observations_converted)

    def reset_task(self):
        pass

    def close(self):
        SimulatorVecEnv._client.terminate()

    def __len__(self):
        return self.nenvs


    # Calling destructor
    def __del__(self):
        print("Destructor called")
        self.env_process.terminate()

def kill_proc(proc):
    print("env proces kill called")
    try:
        proc.terminate()
    except Exception:
        print("exception occured on env process kill")
        pass