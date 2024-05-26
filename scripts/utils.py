import os
import time
import sys
sys.path.append("external/stable-baselines3")
from stable_baselines3.ppo import PPO
from crl.ppo_ewc import PPO_EWC
from crl.ppo_online_ewc import PPO_ONLINE_EWC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.ppo import PPO
from simulator_vec_env import SimulatorVecEnv
from reacher_environment import ReacherEnvironment
from grasper_environment import GrasperEnvironment
from gym.utils import seeding
import numpy as np
LOG_FILE = None


def log(string, log_file=None, global_log=False):
    """

    :param string: the string to be written in the log file
    :param log_file: the log file to be written into
    :param global_log: whether to write in the global log instead of a specific log file
    """
    display_string = time.ctime() + "\t" + string + "\n"

    print(display_string)

    if global_log:
        with open("globalLog.txt", "a") as f_handle:
            f_handle.write(display_string)
    else:
        with open(LOG_FILE, "a") as f_handle:
            f_handle.write(display_string)


def get_folder_name(config):
    """
    :param config: given a config dictionary, the function constructs a folder name using the parameters from the config
    :return: the folder name based on the specified parameters, and a unique int increment if the folder exists
    """

    log_dir = config["log_dir"].strip()
    base_name = 'model_NJ{}_ST{}_RT{}_RN{}_RL{}_S{}_PM{}' .format(
        config['number_joints'],
        config['state'],
        (1 if config['random_torque'] else 0),
        (1 if config['random_noise'] else 0),
        (1 if config['random_latency'] else 0),
        config['seed'],
        (1 if (config['load_model_file_name'] != "") else 0)
    )

    idx = 0
    while os.path.exists(log_dir + "/Model/" + base_name + "--%03d" % idx):
        idx += 1

    folder_name = log_dir + "/Model/" + base_name + "--%03d" % idx

    global LOG_FILE
    LOG_FILE = folder_name + "/log.txt"

    return folder_name

def get_env(conf):
    env_key = conf['env_key']

    def create_env(env_id=0):
        if conf['experiment_type'] == 'reacher':
            env = ReacherEnvironment(max_ts=500, id=env_id, config=conf)
        else:
            env = GrasperEnvironment(max_ts=300, id=env_id, config=conf)
        return env

    num_envs = conf['n_envs']
    env = [create_env for i in range(num_envs)]
    env = VecMonitor(SimulatorVecEnv(env, conf))
    return env


def get_new_model(env, conf):

    if not (conf['strategy'] == "continual" or conf['strategy'] == "continual_online"):
        ewc_lambda = 0
        print("non-continual strategy, ewc_lambda set to 0 in get_new_model")
    else:
        ewc_lambda = conf['ewc_lambda']
        print("ewc_lambda set to value from config file")
       
    if conf['strategy'] == "continual_online":
        model = PPO_ONLINE_EWC("MlpPolicy",
                        env,
                        n_steps=conf['n_steps'],
                        batch_size=conf['batch_size'],
                        gae_lambda=conf['lam'],
                        gamma=conf['gamma'],
                        n_epochs=conf['noptepochs'],
                        ent_coef=conf['ent_coef'],
                        use_sde=conf['use_sde'],
                        learning_rate=conf['learning_rate'],
                        clip_range=conf['cliprange'],
                        verbose=1,
                        tensorboard_log='./Log/Tensorboard/',
                        ewc_num_parallel_envs=conf['n_envs'],
                        ewc_buffer_limit=conf['ewc_buffer_limit'],
                        ewc_replay_batch_size=conf['ewc_replay_batch_size'],
                        ewc_replay_samples=conf['ewc_replay_samples'],
                        ewc_lambda=ewc_lambda,
                        ewc_online_gamma=conf['ewc_online_gamma'],
                        ewc_online_fisher_norm=conf['ewc_online_fisher_norm']
                    )
    else:
        model = PPO_EWC("MlpPolicy",
                        env,
                        n_steps=conf['n_steps'],
                        batch_size=conf['batch_size'],
                        gae_lambda=conf['lam'],
                        gamma=conf['gamma'],
                        n_epochs=conf['noptepochs'],
                        ent_coef=conf['ent_coef'],
                        use_sde=conf['use_sde'],
                        learning_rate=conf['learning_rate'],
                        clip_range=conf['cliprange'],
                        verbose=1,
                        tensorboard_log='./Log/Tensorboard/',
                        ewc_num_parallel_envs=conf['n_envs'],
                        ewc_buffer_limit=conf['ewc_buffer_limit'],
                        ewc_replay_batch_size=conf['ewc_replay_batch_size'],
                        ewc_replay_samples=conf['ewc_replay_samples'],
                        ewc_lambda=ewc_lambda,
                    )

    return model


def duration_string(duration):
    minutes = int(duration / 60)
    seconds = int(duration % 60)
    return "{0:2d}m:{1:2d}s".format(minutes, seconds)


def calculate_mean_reward(model=None, env=None, render=False, n_runs=1, seed=None):
    """
    A function to calculate a mean reward given a model and an environment to test on. It can
    handle environments that have variable number of timesteps (e.g. if terminal condition is reached before max steps)
    :param model: the model to use for prediction
    :param env: the environment to test on (it is expected that VecEnv environment type is provided)
    :param render: whether to visualise the environment during the evaluation
    :param n_runs: how many runs to perform (e.g. usually the VecEnv has X processes where X is number of CPUs), so for
    for more episodes n_runs is used such that n_runs*X episodes will be executed
    :param seed: For reproducibility a seed can be provided, the seed is set once per call of the function

    :return: a mean reward, calculated as the average of all episode rewards TODO: return array optionally
    """

    episode_rewards = []
    n_random, seed = seeding.np_random(seed)
    run_seeds = n_random.randint(1_000_000, size=n_runs)

    for i in range(n_runs):
        print("_____________________________________________________________________")
        print("run: " + str(i) + " seed: " + str(run_seeds[i]))
        env.seed(int(run_seeds[i]))
        obs = env.reset()
        cumulative_reward = 0
        # running_envs is a mask to make each environment in each process run only once in cases of different number
        # of possible timesteps per environment (usually due to early environment solving due to terminal condition
        # other than the maximum number of timesteps). once all environments have completed the run, each environment is
        # considered again
        running_envs = np.ones(env.num_envs, dtype=bool)
        while True:
            action, _states = model.predict(obs, deterministic=True)
            # set the actions to 0 for finished envs (0 usually interpreted as a "do nothing" action)
            action = action * running_envs[:, None]
            obs, rewards, dones, info = env.step(action)
            if render:
                env.render(mode='rgb_array')
            # use the reward per timestep only from the environments that are still running
            cumulative_reward += (rewards*running_envs)
            # update the running envs (sets to 0 the ones that had terminated in this timestep)
            running_envs = np.multiply(running_envs, np.bitwise_not(dones))
            if not np.any(running_envs):
                print(cumulative_reward)
                episode_rewards.append(cumulative_reward)
                break
    mean_reward = np.mean(episode_rewards)
    return mean_reward
