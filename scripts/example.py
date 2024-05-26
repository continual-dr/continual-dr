import sys
sys.path.append("external/stable-baselines3")

from crl.ppo_ewc import PPO_EWC
from crl.ppo_online_ewc import PPO_ONLINE_EWC

from utils import get_env
from config import Config
from utils import calculate_mean_reward, get_new_model

import numpy as np
np.set_printoptions(precision=3)


if __name__ == "__main__":
    
    config = Config.get_config_dict()
    
    # training mode
    config["mode"] = 1
    
    # number of paralel training envs
    config["n_envs"] = 50
    
    config['working_dir'] = config['root_directory'] + "/experiments/test"
    
    # the training strategy, for PPO_ONLINE_EWC use "continual_online", or "finetuning" for no CL regularization
    config['strategy'] = "finetuning"

    if config['strategy'] not in ["continual", "continual_online"]:
        config["ewc_lambda"] = 0.0
        print("not a continual learning strategy, setting ewc-labda to 0")


    # ideal env
    config['random_torque'] = False
    config['random_latency'] = False
    config['random_noise'] = False
    env = get_env(config)

    # create new model and train it on task 0 (ideal sim)
    model = get_new_model(env, config)  
    model.learn(total_timesteps=1_000_000)
    # updating EWC params after training")
    model.update_ewc_params(task_id=0)
    model.save(config["working_dir"] + "/model_0.zip")

    for i in range (1, 4):
        # randomized torque env
        config['random_torque'] = i == 1
        config['random_latency'] = i == 2
        config['random_noise'] = i == 3 
        # change port number to avoid waiting on the previous simulator to shut down
        config["port_number"] = str(9092 - i)
        
        env = get_env(config)
        if config["strategy"] == "continual_online":
            model = PPO_ONLINE_EWC.load(config["working_dir"] + f"/model_{i-1}.zip", env=env)
        else:
            model = PPO_EWC.load(config["working_dir"] + f"/model_{i-1}.zip", env=env)
        model.learn(total_timesteps=1_000_000)
        # updating EWC params after training")
        model.update_ewc_params(task_id=i)
        model.save(config["working_dir"] + f"/model_{i}.zip")
        

