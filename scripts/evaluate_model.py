import argparse

import sys
sys.path.append("external/stable-baselines3")
from stable_baselines3.ppo import PPO
from crl.ppo_ewc import PPO_EWC

from utils import log, get_env
from config import Config
from utils import calculate_mean_reward

import numpy as np

np.set_printoptions(precision=3)

def evaluate_model(model_path, env):

    print("Evaluating model: " + model_path)
    model = PPO_EWC.load(model_path)
    mean_reward = calculate_mean_reward(model=model, env=env, render=False, n_runs=1, seed=17)
    print("* evaluation finished; avg_rew: {0}".format(mean_reward))

    return mean_reward


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-mp", "--model_path", type=str, help="the path to the model")
    args = parser.parse_args()

    config = Config.get_config_dict()
    config["mode"] = 0
    config["n_envs"] = 50
    config['random_torque'] = False
    config['random_latency'] = False
    config['random_noise'] = False
    config['working_dir'] = config['root_directory'] + "/experiments/evaluation"
    #config["port_number"] = '9089'
    
    eval_env_ideal = get_env(config)
    
    mean_reward = evaluate_model(model_path=args.model_path, env=eval_env_ideal)

