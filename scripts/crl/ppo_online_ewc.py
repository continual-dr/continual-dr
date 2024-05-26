import io
import pathlib
from typing import Any, Dict, Optional, Type, TypeVar, Union

import numpy as np
from tqdm import trange

import torch as th

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import GymEnv, Schedule

from .ppo_ewc import PPO_EWC


SelfPPO_ONLINE_EWC = TypeVar("SelfPPO_ONLINE_EWC", bound="PPO_ONLINE_EWC")

class PPO_ONLINE_EWC(PPO_EWC):

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        ewc_num_parallel_envs: int = 4,
        ewc_buffer_limit: int = int(5e5),
        ewc_replay_batch_size: int = 32,
        ewc_replay_samples: int = int(5e5),
        ewc_lambda: float = 400.0,
        # Online EWC hyperparams
        ewc_online_gamma: float = 0.95, # From P&C paper (table 2)
        ewc_online_fisher_norm: bool = True  # Normalize Fisher info
    ):
        
        super().__init__(policy,
                         env,
                         learning_rate,
                         n_steps,
                         batch_size,
                         n_epochs,
                         gamma,
                         gae_lambda,
                         clip_range,
                         clip_range_vf,
                         normalize_advantage,
                         ent_coef,
                         vf_coef,
                         max_grad_norm,
                         use_sde,
                         sde_sample_freq,
                         target_kl,
                         tensorboard_log,
                         policy_kwargs,
                         verbose,
                         seed,
                         device,
                         _init_setup_model,
                         ewc_num_parallel_envs,
                         ewc_buffer_limit,
                         ewc_replay_batch_size,
                         ewc_replay_samples,
                         ewc_lambda
                         )

        # Online EWC parameters
        self.ewc_online_gamma = ewc_online_gamma
        self.ewc_online_fisher_norm = ewc_online_fisher_norm

    def unnorm_fisher(self):

        print('Computing unnormalized fisher coefficients')

        # Initialize a replay buffer
        replay_buffer = ReplayBuffer(
            buffer_size=self.ewc_buffer_limit*self.ewc_num_parallel_envs, # SB3 adjusts buffersize by dividing by n_envs
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            device=self.device,
            n_envs=self.ewc_num_parallel_envs,
            handle_timeout_termination=True)

        # Fill the replay buffer
        obs = self.env.reset()
        print('Filling replay buffer for loglikelihood computation')
        for _ in trange(self.ewc_buffer_limit):
            action, _states = self.predict(obs, deterministic=True)
            new_obs, rewards, dones, info = self.env.step(action)
            replay_buffer.add(obs=obs,
                              next_obs=new_obs,
                              action=action,
                              reward=rewards,
                              done=dones,
                              infos=info)
            obs = new_obs

        if self.verbose >= 1:
            print(f"Replay buffer contains {replay_buffer.size()} tuples")

        # Calculate log likelihood by sampling data from replay buffer
        log_likelihoods = []
        for _ in range(self.ewc_replay_samples):
            # Sample replay buffer
            replay_data = replay_buffer.sample(self.ewc_replay_batch_size)
            obs, actions, next_obs, dones, rewards = replay_data

            # I need the log likelihood of an action given a state
            dist = self.policy.get_distribution(obs.to(self.device))
            log_likelihood = dist.log_prob(actions.to(self.device))
            log_likelihoods.append(log_likelihood)

        log_likelihood = th.cat(log_likelihoods).mean()

        # Compute the gradients of the log likelihood w.r.t. the actor parameters
        # and also store the parameters of the actors (optimized for the current task)
        for n, p in self.policy.named_parameters():
            # TODO: This operation generates nans on cuda (torch 1.10.2+cu102), ok on cpu
            grad_log_liklihood = th.autograd.grad(log_likelihood,
                                                  p,
                                                  retain_graph=True,
                                                  allow_unused=True)

            # Only consider parameters from the actor
            # Parameters of the critic do not contribute to the log likelihood computation
            # TODO: Do we need to regularize the log_std? Including it drives down the fisher
            # coefficient of all other parameters to close to 0
            if grad_log_liklihood[0] is not None and 'log_std' not in n:
                param_name = n.replace('.', '__')
                buffer_name_unnorm_fisher = f'buffer_{param_name}_unnorm_fisher'

                # Fisher info is the square of the grad of the log likelihood
                unnorm_fisher = grad_log_liklihood[0].data.clone()**2

                if self.ewc_current_task == 0:
                    # If this is task 0, there is no previous unnormalized Fisher
                    # Register the fisher info buffer
                    self.policy.register_buffer(buffer_name_unnorm_fisher,
                                                # unnorm_fisher.data.clone()**2) <- The **2 operation is probably a bug
                                                unnorm_fisher.data.clone())
                else:
                    # If this is not the first task, the existing buffer needs to be overwritten
                    # Update the normalized fisher info the existing buffer
                    setattr(self.policy, buffer_name_unnorm_fisher, unnorm_fisher)

    def norm_update_fisher(self):

        print('Normalizing and updating fisher coefficients in model buffer')

        # TODO: self.ewc_online_fisher_norm is not checked

        # Initialize the max to a low value
        fisher_max = -th.tensor(float('inf')).to(self.device)
        # Initialize the min to a high value
        fisher_min = th.tensor(float('inf')).to(self.device)

        # Find the max and min values in the unnormalized Fisher info matrices
        for n,p in self.policy.named_buffers():
            if 'unnorm_fisher' in n:
                p_max = th.max(p)
                p_min = th.min(p)
                if p_max > fisher_max:
                    fisher_max = p_max.data.clone()
                if p_min < fisher_min:
                    fisher_min = p_min.data.clone()

        # We store the buffers separately so as not to mutate them
        # while we are iterating over them in the next for loop
        buffer_names, buffer_params = list(), list()
        for n,p in self.policy.named_buffers():
            buffer_names.append(n)
            buffer_params.append(p.data.clone())

        # Iterate over the stored buffers, normalize and update them
        for n,p in zip(buffer_names, buffer_params):
        
            # We go through each unnormalized fisher buffer
            # For each unnormalized fisher buffer, a normalized version
            # is created/updated
            if 'unnorm_fisher' in n:

                # Normalize the current fisher info
                fisher = (p - fisher_min) / (fisher_max - fisher_min)

                # Find the name of the normalized buffer
                buffer_name_fisher = n.replace('_unnorm', '')

                # Create or update the normalized fisher buffer

                # If this is task 0, there is no previous Fisher
                fisher_star = None
                if self.ewc_current_task == 0:

                    # Since this is the first time fisher info is computed
                    fisher_star = fisher

                    # Register the fisher info buffer
                    self.policy.register_buffer(buffer_name_fisher,
                                                fisher_star.data.clone())
                else:

                    # Fetch the previous fisher info
                    fisher_star_previous = getattr(self.policy, buffer_name_fisher)

                    # Update the running average of the fisher info
                    # As per eq. 9 in P&C paper
                    fisher_star = (self.ewc_online_gamma * fisher_star_previous) + fisher

                    # Update the fisher info buffer
                    setattr(self.policy, buffer_name_fisher, fisher_star)

    def update_map(self):

        print('Creating/updating buffers for MAP parameters')

        # Get a list of the normalized fisher buffer names
        fisher_buffer_names = [n for n,_ in self.policy.named_buffers() if 'unnorm' not in n]

        # We need to update buffers for only those buffers 
        # for which fisher info has been computed
        for n, p in self.policy.named_parameters():

            # Parameter names contain '.', which needs to be replaced
            param_name = n.replace('.', '__')

            # Name of the corresponding fisher buffer
            buffer_name_fisher = f'buffer_{param_name}_fisher'

            # Create/update a param buffer only if fisher exists
            if buffer_name_fisher in fisher_buffer_names:

                # Register/update the MAP parameters
                buffer_name_param = f'buffer_{param_name}_param'

                if self.ewc_current_task == 0:
                    self.policy.register_buffer(buffer_name_param, 
                                                p.data.clone())
                else:
                    setattr(self.policy, 
                            buffer_name_param, 
                            p.data.clone())

    def update_ewc_params(self, task_id:int) -> None:
        """_summary_

        Note: task_id is unused and is kept here so that we do not have to reimplement the train method
        """

        # Compute unnormalized fisher matrices
        self.unnorm_fisher()

        # Normalize and update fisher matrices
        self.norm_update_fisher()

        # Create/update buffers for MAP parameters
        self.update_map()

        if self.verbose >= 1:
            print("Existing buffers:")
            for n,p in self.policy.named_buffers():
                print(f"name: {n}, shape:  {p.shape}, sum: {p.sum().item()}")

        # Update the task number
        self.ewc_current_task += 1

    def get_ewc_loss(self) -> th.Tensor:

        ewc_losses = list()
        if self.ewc_current_task == 0:
            return th.zeros(())
        else:
            for n,p in self.policy.named_parameters():
                param_name = n.replace('.', '__')

                try:
                    param_task = getattr(self.policy, f'buffer_{param_name}_param')
                    fisher_info = getattr(self.policy, f'buffer_{param_name}_fisher')

                    ewc_losses.append((fisher_info * (p - param_task) ** 2).sum())
                except AttributeError as err:
                    # Parameters of the critic do not have buffers and raise an Attribute error
                    #if self.verbose >= 1:
                    #    print(err)
                    ...
        ewc_loss = (self.ewc_lambda / 2.0) * sum(ewc_losses)
        return ewc_loss

    @classmethod
    def load_from_ewc(
        cls: Type[SelfPPO_ONLINE_EWC],
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        ewc_online_gamma: float,
        ewc_online_fisher_norm:bool=True,
        env: Optional[GymEnv] = None,
        verbose: int = 0,
        device: Union[th.device, str] = "auto",
        tensorboard_log: Optional[str] = None,
        conf = None,
    ) -> SelfPPO_ONLINE_EWC:      
        """
        Loads an EWC model (saved after the first task), 
        removes the existing EWC buffers and creates the buffers
        needed for OnlineEWC. The returned onlineEWC model acts
        as if it was already trained on the first task (task_id=0).

        All model arguments that are common between EWC and onlineEWC
        are set from the loaded EWC model. The arguments that are specific
        to onlineEWC (ewc_online_gamma, ewc_online_fisher_norm) need
        to be provided when this class method is called.
        """

        # Load the EWC model
        model_ewc = PPO_EWC.load(
            path,
            only_buffers=False, 
            verbose=True)
    
        if conf is not None:
            # usually for task 0 ewc_lambda is 0
            print("setting ewc_lambda from " + str(model_ewc.ewc_lambda) + " to " + str(conf["ewc_lambda"]))
            model_ewc.ewc_lambda = conf["ewc_lambda"]
        
        # Only works if just one task has been learned
        assert model_ewc.ewc_current_task == 1, f"Cannot load EWC model for task {model_ewc.ewc_current_task}"
        
        if conf is None:
            # Initialize an PPO_ONLINE_EWC model
            model_online_ewc = PPO_ONLINE_EWC(
                "MlpPolicy",
                env=env,
                ewc_num_parallel_envs=model_ewc.ewc_num_parallel_envs,
                ewc_buffer_limit=model_ewc.ewc_buffer_limit,
                ewc_replay_batch_size=model_ewc.ewc_replay_batch_size,
                ewc_replay_samples=model_ewc.ewc_replay_samples,
                ewc_lambda=model_ewc.ewc_lambda,
                ewc_online_gamma=ewc_online_gamma,
                ewc_online_fisher_norm=ewc_online_fisher_norm,
                tensorboard_log=tensorboard_log,
                device=device
                )
        else:
            model_online_ewc = PPO_ONLINE_EWC("MlpPolicy",
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
                        ewc_lambda=conf['ewc_lambda'],
                        ewc_online_gamma=conf['ewc_online_gamma'],
                        ewc_online_fisher_norm=conf['ewc_online_fisher_norm']
                    )
        model_online_ewc.set_parameters(path, exact_match=False)

        # we have to set the task ID to 1 if we load from previous model which trained on the first task already
        model_online_ewc.ewc_current_task = 1

        assert model_ewc.ewc_fisher_norm, f"Cannot use EWC to create onlineEWC if model_ewc.ewc_fisher_norm={model_ewc.ewc_fisher_norm}"

        # Copy the relevant fisher buffers from EWC to onlineEWC 
        # and change buffer names where necessary
        ewc_buffers = [buff_name for buff_name, _ in model_ewc.policy.named_buffers()]
        for ewc_buff_name, ewc_buff in model_ewc.policy.named_buffers():

            online_ewc_buff_name = ewc_buff_name.replace('_task0', '')
            model_online_ewc.policy.register_buffer(online_ewc_buff_name, 
                                                    ewc_buff.data.clone())
            
        return model_online_ewc
