import time
import sys
sys.path.append("external/stable-baselines3")
from crl.ppo_ewc import PPO_EWC
from crl.ppo_online_ewc import PPO_ONLINE_EWC

from stable_baselines3.common.callbacks import \
    EvalCallback, CheckpointCallback, CallbackList, BaseCallback, EveryNTimesteps

from utils import log, get_env, duration_string, get_new_model, calculate_mean_reward


def run_single(conf, training=True, render=False, evaluate=True):

    file_name = None
    model = None
    if training:

        folder_name = conf['working_dir']
        
        # Save a checkpoint
        checkpoint_callback = CheckpointCallback(save_freq=1, save_path=folder_name,
                                                 name_prefix='rl_model')
        callback = CallbackList([checkpoint_callback])
        everyNtimestepscallback = EveryNTimesteps(conf['save_frequency'], callback=callback)

        start = time.time()

        log('***************************')
        log("* start calculation")

        if conf['load_model_file_name']:

            env = get_env(conf)
            log("Loading pretrained model for training: " + conf['load_model_file_name'])
            print("Loading model")

            if conf['strategy'] == 'continual_online':
                if int(conf["index"]) == 1:
                    model = PPO_ONLINE_EWC.load_from_ewc(conf['load_model_file_name'], 
                                  verbose=True, 
                                  env=env, 
                                  ewc_online_gamma=conf['ewc_online_gamma'],
                                  ewc_online_fisher_norm=conf['ewc_online_fisher_norm'],
                                  conf=conf,
                                  )
                    print("loading from non-online to online CL model")
                else:
                    model = PPO_ONLINE_EWC.load(conf['load_model_file_name'], 
                                  verbose=True, 
                                  env=env, 
                                  only_buffers=conf['ewc_load_only_buffers']
                                  )
            else:
                model = PPO_EWC.load(conf['load_model_file_name'], 
                                  verbose=True, 
                                  env=env, 
                                  only_buffers=conf['ewc_load_only_buffers']
                                  )       
     
            if conf['ewc_load_only_buffers']:
                print("loading only the buffers from previous tasks")
            
            
            if conf['strategy'] == 'continual' or conf['strategy'] == 'continual_online':
                    print("setting ewc_lambda to " + str(conf["ewc_lambda"]))
                    model.ewc_lambda = conf["ewc_lambda"]
            elif (model.ewc_lambda > 0):
                print("non-continual strategy with previous model ewc_lambda > 0, setting it to 0")
                model.ewc_lambda = 0

            model.set_env(env=env, force_reset=True)
            model.tensorboard_log = folder_name
        else:
            log("Creating new model for training")
            env = get_env(conf)
            model = get_new_model(env, conf)
            model.tensorboard_log = folder_name

        file_name = folder_name + "/rl_model_0_steps.zip"
        log("saving initial model in {}".format(file_name))
        model.save(file_name)

        model.learn(total_timesteps=conf['n_ts'], callback=everyNtimestepscallback)
        print("updating EWC params after training")
        model.update_ewc_params(task_id=int(conf["index"]))

        file_name = folder_name + "/rl_model_" + str(conf['n_ts']) + "_steps.zip"
        log("saving final model in {}".format(file_name))
        model.save(file_name)

        log("Calculation finished in " + duration_string(time.time() - start))

    ##########################################################################

    if evaluate:

        if model is None:
            # if we are evaluating directly without running training
            model = PPO.load(conf['log_dir'] + conf['load_model_file_name'])
            eval_env = get_env(conf)
        log("* start evaluation")
        mean_reward = calculate_mean_reward(model=model, env=eval_env, render=render, n_runs=1, seed=131)
        log("* evaluation finished; avg_rew: {0}".format(mean_reward))
        print("* evaluation finished; avg_rew: {0}".format(mean_reward))
        return float(mean_reward)




