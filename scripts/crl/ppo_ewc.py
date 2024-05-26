import warnings
import io
import pathlib
from typing import Any, Dict, Optional, Type, TypeVar, Union

import numpy as np
from tqdm import trange

import torch as th
from gym import spaces
from torch.nn import functional as F

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import explained_variance

# For EWC
from stable_baselines3.common.buffers import ReplayBuffer

# For the `load()` method
from stable_baselines3.common.save_util import load_from_zip_file, recursive_setattr
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import (
    check_for_correct_spaces,
    get_system_info,
)

SelfPPO_EWC = TypeVar("SelfPPO_EWC", bound="PPO_EWC")

class PPO_EWC(PPO):

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
        # Default EWC params as per "https://arxiv.org/pdf/1612.00796.pdf" (page 12)
        ewc_num_parallel_envs: int = 4,
        ewc_buffer_limit: int = int(5e5),
        ewc_replay_batch_size: int = 32,
        ewc_replay_samples: int = int(5e5),
        ewc_lambda: float = 400.0,
        ewc_fisher_norm: bool = True  # Normalize Fisher info
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
                         _init_setup_model)

        # EWC parameters
        self.ewc_num_parallel_envs = ewc_num_parallel_envs
        self.ewc_buffer_limit = ewc_buffer_limit
        self.ewc_replay_batch_size = ewc_replay_batch_size
        self.ewc_replay_samples = ewc_replay_samples
        self.ewc_lambda = ewc_lambda
        self.ewc_fisher_norm = ewc_fisher_norm

        # For tracking the tasks
        self.ewc_current_task = 0

    @classmethod  # noqa: C901
    def load(
        cls: Type[SelfPPO_EWC],
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        verbose: bool = False,
        env: Optional[GymEnv] = None,
        device: Union[th.device, str] = "auto",
        custom_objects: Optional[Dict[str, Any]] = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        only_buffers: bool = False,
        **kwargs,
    ) -> SelfPPO_EWC:
        """
        Load the model from a zip-file.
        Warning: ``load`` re-creates the model from scratch, it does not update it in-place!
        For an in-place load use ``set_parameters`` instead.

        :param path: path to the file (or a file-like) where to
            load the agent from
        :param env: the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model) has priority over any saved environment
        :param device: Device on which the code should run.
        :param custom_objects: Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            ``keras.models.load_model``. Useful when you have an object in
            file that can not be deserialized.
        :param print_system_info: Whether to print system info from the saved model
            and the current system info (useful to debug loading issues)
        :param force_reset: Force call to ``reset()`` before training
            to avoid unexpected behavior.
            See https://github.com/DLR-RM/stable-baselines3/issues/597
        :param only_buffers: If True, only load EWC buffers from the disk and initialize trainable
            paramaters similar to how a new model would be initialized.
        :param kwargs: extra arguments to change the model when loading
        :return: new model instance with loaded parameters
        """
        if print_system_info:
            print("== CURRENT SYSTEM INFO ==")
            get_system_info()

        data, params, pytorch_variables = load_from_zip_file(
            path,
            device=device,
            custom_objects=custom_objects,
            print_system_info=print_system_info,
        )        

        # Remove stored device information and replace with ours
        if "policy_kwargs" in data:
            if "device" in data["policy_kwargs"]:
                del data["policy_kwargs"]["device"]

        if "policy_kwargs" in kwargs and kwargs["policy_kwargs"] != data["policy_kwargs"]:
            raise ValueError(
                f"The specified policy kwargs do not equal the stored policy kwargs."
                f"Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}"
            )

        if "observation_space" not in data or "action_space" not in data:
            raise KeyError("The observation_space and action_space were not given, can't verify new environments")

        if env is not None:
            # Wrap first if needed
            env = cls._wrap_env(env, data["verbose"])
            # Check if given env is valid
            check_for_correct_spaces(env, data["observation_space"], data["action_space"])
            # Discard `_last_obs`, this will force the env to reset before training
            # See issue https://github.com/DLR-RM/stable-baselines3/issues/597
            if force_reset and data is not None:
                data["_last_obs"] = None
            # `n_envs` must be updated. See issue https://github.com/DLR-RM/stable-baselines3/issues/1018
            if data is not None:
                data["n_envs"] = env.num_envs
        else:
            # Use stored env, if one exists. If not, continue as is (can be used for predict)
            if "env" in data:
                env = data["env"]

        # noinspection PyArgumentList
        model = cls(  # pytype: disable=not-instantiable,wrong-keyword-args
            policy=data["policy_class"],
            env=env,
            device=device,
            _init_setup_model=False,  # pytype: disable=not-instantiable,wrong-keyword-args
        )

        # load parameters
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model._setup_model()

        # Trainable parameters are loaded from disk only when `only_buffers=False` 
        # Otherwise we only load the buffers and the trainable params are initialized
        # from scratch
        if not only_buffers:
            try:
                # put state_dicts back in place
                model.set_parameters(params, exact_match=False, device=device)
            except RuntimeError as e:
                # Patch to load Policy saved using SB3 < 1.7.0
                # the error is probably due to old policy being loaded
                # See https://github.com/DLR-RM/stable-baselines3/issues/1233
                if "pi_features_extractor" in str(e) and "Missing key(s) in state_dict" in str(e):
                    #### `exact_match=False` is necessary! we load missing buffers later
                    #### Buffers exist in params, but the empty model's state_dict does not have the buffer names, 
                    #### so when values are set using `model.set_parameters` with ȩxact_match=True`, we get an error.
                    model.set_parameters(params, exact_match=False, device=device)
                    warnings.warn(
                        "You are probably loading a model saved with SB3 < 1.7.0, "
                        "we deactivated exact_match so you can save the model "
                        "again to avoid issues in the future "
                        "(see https://github.com/DLR-RM/stable-baselines3/issues/1233 for more info). "
                        f"Original error: {e} \n"
                        "Note: the model should still work fine, this only a warning."
                    )
                else:
                    raise e
    
        #### Buffers are not loaded from a file automatically
        #### Check: https://github.com/pytorch/pytorch/issues/28340#issuecomment-545922436
        #### Set the buffers which are not set by `set_parameters`
        for p_k, p_v in params['policy'].items():

            #### We distinguish buffers from params using the name
            if p_k.startswith('buffer_'):

                # Register the buffer in the model
                model.policy.register_buffer(p_k, p_v)
        
        if verbose:
            print(f'Created buffers for previous tasks')

        # put other pytorch variables back in place
        if pytorch_variables is not None:
            for name in pytorch_variables:
                # Skip if PyTorch variable was not defined (to ensure backward compatibility).
                # This happens when using SAC/TQC.
                # SAC has an entropy coefficient which can be fixed or optimized.
                # If it is optimized, an additional PyTorch variable `log_ent_coef` is defined,
                # otherwise it is initialized to `None`.
                if pytorch_variables[name] is None:
                    continue
                # Set the data attribute directly to avoid issue when using optimizers
                # See https://github.com/DLR-RM/stable-baselines3/issues/391
                recursive_setattr(model, name + ".data", pytorch_variables[name].data)

        # Sample gSDE exploration matrix, so it uses the right device
        # see issue #44
        if model.use_sde:
            model.policy.reset_noise()  # pytype: disable=attribute-error

        if verbose:
            print('Model loading complete')

        return model

    def norm_fisher(self, task_id:int):

        print(f'Normalizing fisher coefficients in model buffer for task {task_id}')

        # Initialize the max to a low value
        fisher_max = -th.tensor(float('inf')).to(self.device)
        # Initialize the min to a high value
        fisher_min = th.tensor(float('inf')).to(self.device)

        # Find the max and min values in the unnormalized Fisher info matrices
        for n,p in self.policy.named_buffers():
            # Consider only the fisher buffers for the current task
            if f'_task{task_id}_' in n and 'fisher' in n:
                p_max = th.max(p)
                p_min = th.min(p)
                if p_max > fisher_max:
                    fisher_max = p_max.data.clone()
                if p_min < fisher_min:
                    fisher_min = p_min.data.clone()

        if self.verbose >= 1:
            print(f'Fisher max: {fisher_max}, min: {fisher_min}')

        # We store the buffers separately so as not to mutate them
        # while we are iterating over them in the next for loop
        buffer_names, buffer_params = list(), list()
        for n,p in self.policy.named_buffers():
            
            # Consider only the fisher buffers for the current task
            if f'_task{task_id}_' in n and 'fisher' in n:
                buffer_names.append(n)
                buffer_params.append(p.data.clone())

        # Iterate over the stored buffers, normalize and update them
        for n,p in zip(buffer_names, buffer_params):

            # Make sure that we are only changing the fisher buffers for the current task
            assert f'_task{task_id}_' in n and 'fisher' in n

            # Check values before normalizing
            if self.verbose >= 1:
                print(f'Before normalization {n}, max: {th.max(p)}, min: {th.min(p)}')

            # Normalize the current fisher info
            fisher = (p - fisher_min) / (fisher_max - fisher_min)

            # Update the fisher info buffer
            setattr(self.policy, n, fisher)

            # Check values after normalizing
            if self.verbose >= 1:
                print(f'Normalized buffer {n}, max: {th.max(fisher)}, min: {th.min(fisher)}')

    def update_ewc_params(self,
                          task_id: int) -> None:
        """_summary_

        Args:
            task_id (int): ID of latest task to be learned.
        """

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
        print("Filling replay buffer")
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
                param = p

                # Fisher info is the square of the grad of the log likelihood
                fisher_info = grad_log_liklihood[0].data.clone()**2

                # Store the model parameters for the current task
                self.policy.register_buffer(
                    f'buffer_task{task_id}_{param_name}_param', param.data.clone())

                # Store the fisher info for the current task
                self.policy.register_buffer(
                    f'buffer_task{task_id}_{param_name}_fisher',
                    #fisher_info.data.clone()**2) # **2 is most likely a bug
                    fisher_info.data.clone())
                
        # Normalize the fisher info for the current task (if required)
        if self.ewc_fisher_norm:
            self.norm_fisher(task_id)

        if self.verbose >= 1:
            print("New buffers created:")
            for n,p in self.policy.named_buffers():
                if f"task{task_id}" in n:
                    print(f"name: {n}, shape:  {p.shape}, sum: {p.sum().item()}")

        # Update the task number
        self.ewc_current_task += 1

    def get_ewc_loss(self) -> th.Tensor:

        ewc_losses = list()
        if self.ewc_current_task == 0:
            return th.zeros(())
        else:
            for past_task_id in range(self.ewc_current_task):

                for n,p in self.policy.named_parameters():
                    param_name = n.replace('.', '__')

                    try:
                        param_task = getattr(self.policy, f'buffer_task{past_task_id}_{param_name}_param')
                        fisher_info = getattr(self.policy, f'buffer_task{past_task_id}_{param_name}_fisher')

                        ewc_losses.append((fisher_info * (p - param_task) ** 2).sum())
                    except AttributeError as err:
                        # Parameters of the critic do not have buffers and raise an Attribute error
                        #if self.verbose >= 1:
                        #    print(err)
                        ...

            ewc_loss = (self.ewc_lambda / 2.0) * sum(ewc_losses)
            return ewc_loss

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        ewc_losses = []
        clip_fractions = []

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            # TODO: Comment out if this is a problem for envs other than lunar lander
            #if self.verbose >= 1 and 'LunarLander' in self.env.envs[0].spec.id:
            #    print(f'Engine power: {self.env.env_method("get_main_engine_power")}')
            
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                ewc_loss = self.get_ewc_loss()
                ewc_losses.append(ewc_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + ewc_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/ewc_loss", np.mean(ewc_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
