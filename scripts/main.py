import os
import argparse
import json
import sys
sys.path.append("external/stable-baselines3")
from stable_baselines3.common.utils import set_random_seed

from run_single_training import run_single
from utils import get_folder_name, log
from config import Config


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--environment", type=str, help="the environment to use")
    parser.add_argument("-l", "--log_dir", type=str, help="the directory to log info and store data to")
    parser.add_argument("-m", "--mode", type=int,
                        help="Whether to run testing(0), training(1), or hyperparameter-optimization(2)")
    parser.add_argument("-stg", "--strategy", type=str, help="training_strategy")
    parser.add_argument("-ind", "--index", type=int, help="the index of the task", default=0)
    parser.add_argument("-img", "--use_images", type=int, help="if True image instead of numeric observation is used")
    parser.add_argument("-iw", "--image_width", type=int, help="width of the image if image observation is used")
    parser.add_argument("-ih", "--image_height", type=int, help="height of the image if image observation is used")
    parser.add_argument("-rl", "--random_latency", type=int, help="whether to randomize latency")
    parser.add_argument("-rt", "--random_torque", type=int, help="whether to randomize torque for joint motors")
    parser.add_argument("-rn", "--random_noise", type=int, help="whether to add random noise to the observations")
    parser.add_argument("-ra", "--randomize_appearance", type=int, help="whether to randomize appearance")
    parser.add_argument("-rv", "--randomize_viewpoint", type=int, help="whether to randomize viewpoint")
    parser.add_argument("-mn", "--model_name", type=str, help="a pretrained model name to start from")
    parser.add_argument("-nt", "--number_timesteps", type=int, help="number of timesteps for training")
    parser.add_argument("-n_envs", "--number_envs", type=int, help="number of parallel envs to start")
    parser.add_argument("-s", "--seed", type=int, help="the seed to set for the pseudo-random generators")
    parser.add_argument("-nj", "--number_joints", type=int, help="the number of robot joints to be used")
    parser.add_argument("-jv", "--joint_velocity", type=int, help="the max possible velocity of a joint in deg/sec")
    parser.add_argument("-st", "--state", type=str, help="a for angles only or av for angles and velocities")
    parser.add_argument("-pn", "--port_number", type=int, help="the number of the port to connect to for Unity sim")
    parser.add_argument("-el", "--ewc_lambda", type=float, help="the ewc lambda parameter value")
    args = parser.parse_args()

    set_random_seed(args.seed)

    config = Config.get_config_dict()
    config['log_dir'] = args.log_dir
    config['mode'] = args.mode
    config['strategy'] = args.strategy
    config['index'] = args.index
    config['n_envs'] = args.number_envs
    config['n_ts'] = args.number_timesteps
    config['load_model_file_name'] = args.model_name
    config['random_latency'] = bool(args.random_latency)
    config['random_torque'] = bool(args.random_torque)
    config['random_noise'] = bool(args.random_noise)
    config['seed'] = args.seed
    config['port_number'] = args.port_number
    config['ewc_lambda'] = args.ewc_lambda
    config['working_dir'] = get_folder_name(config)
    config['command_line_params'] = ' '.join(sys.argv[1:-2])

    log("creating new working dir: " + config['working_dir'], global_log=True)
    os.makedirs(config['working_dir']) #, exist_ok=True)

    log("args: "+ str(args), global_log=True)
    with open(config['working_dir'] + '/config.json', 'w') as fp:
        json.dump(config, fp)
        log("saving config in: {}".format(config['working_dir'] + '/config.json'))
        print(config)

    if args.mode == 0:
        # Test
        run_single(config, False, False, True)

    elif args.mode == 1:
        # Train
        run_single(config, True, True, False)

