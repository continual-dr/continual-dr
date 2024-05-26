class Config:

    def __init__(self):
        pass

    @staticmethod
    def get_config_dict():
        config_dict = {

            ###################################### GENERAL PARAMETERS ##################################################
            # the experiment type: reacher or grasper
            'experiment_type': 'reacher',

            # environment-specific parameter, e.g, number of controllable joints (should be 2 for reacher and 7 for grasper)
            'number_joints': 2,

            # whether the RL algorithm uses customized or stable-baseline specific parameters
            'custom_hps': True,

            # the directory in which the logs will be saved
            'log_dir': "./experiments",

            # the root of the repo
            'root_directory': "/home/continual-dr",

            # the environment key, needs to be "manipulator_environment" for correct comunication with the simulator
            'env_key': 'manipulator_environment',

            # communication_protocol with the simulator
            'communication_type': 'GRPC',

            # the ip address of the simulator, if you run the python code and the simulator inside docker it should be:
            # 'localhost" for Linux
            # 'host.docker.internal' for Windows
            # an actual ip address of another machine if the simulator is running on a seperate machine
            'ip_address': 'localhost',
            #'ip_address': 'host.docker.internal',

            # port number for communication with the simulator
            'port_number': '9092',

            # the seed used for generating pseudo-random sequences
            'seed': 17,

            # the state of the RL agent in case of numeric values for manipulators 'a' for angles only or
            # or 'av' for angles and velocities
            'state': 'a',

            # after every X timesteps to save the model during training
            'save_frequency': 40_000,

            # type of run: 0 - evaluation, 1 - training
            'mode': 1,

            ############################### RL algroitham (PPO)-SPECIFIC PARAMETERS ####################################
            #based on: https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/ppo2/ppo2.py

            # number of environments running in parallel
            'n_envs': 64,

            # number of timesteps to run the training
            'n_ts': 10_000_000,

            # whether to load a pretrained model (False means starting from scratch)
            'load_model_file_name': False,

            # adjust the policy network pi: policy; vf: value
            'net_arch': [dict(pi=[64, 64], vf=[64, 64])],

            # The number of steps to run for each environment per update (i.e. batch size is n_steps * n_env where n_env
            # is number of environment copies running in parallel) - default: 128
            'n_steps': 256,

            # Number of training minibatches per update. For recurrent policies, the number of environments run in
            # parallel should be a multiple of nminibatches.
            #'nminibatches': 32,

            'batch_size': 512,

            # Factor for trade - off of bias vs variance for Generalized Advantage Estimator - default: 0.95
            'lam': 0.95,

            # Discount factor - default: 0.99 - (most common), 0.8 to 0.9997
            'gamma': 0.99,

            # Number of epoch when optimizing the surrogate - 4
            'noptepochs': 10,

            # Entropy coefficient for the loss calculation - multiplied by entropy and added to loss 0.01
            'ent_coef': 0.0,

            # learning rate - can be a function - default 2.5 e-4
            'learning_rate': 2.5e-4,

            # whether to enable linear decay of the learning rate as a function of the timesteps
            'linear_decay': False,

            # Clipping parameter, it can be a function - default: 0.2 - can be between 0.1-0.3
            'cliprange': 0.1,

            # Value function coefficient for the loss calculation - default: 0.5
            'vf_coef': 0.5,

            # The maximum value for the gradient clipping - default: 0.5
            'max_grad_norm': 0.5,

            # Whether to use state=dependent exploration
            'use_sde': False,

            ###################################### RANDOMIZATION PARAMETERS #######################################
            
            'strategy': 'ideal',    
            'random_torque': False,
            'random_latency': False,
            'random_noise': False,

            #################################### EWC SPECIIC PARAMETERS ###########################################

            'ewc_buffer_limit': 500,
            'ewc_replay_batch_size': 32,
            'ewc_replay_samples': 5000,
            'ewc_lambda': 5000,
            'ewc_online_gamma':  0.95,
            'ewc_online_fisher_norm': True,
            'ewc_load_only_buffers': False
        }

        return config_dict
