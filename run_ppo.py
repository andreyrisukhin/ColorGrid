import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import Mapping, Sequence, Tuple
from dataclasses import dataclass
import wandb
from fire import Fire
from tqdm import tqdm
import os
from pettingzoo import ParallelEnv

from color_maze import ColorMaze, ColorMazeRewards, NUM_COLORS

@dataclass
class StepData:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    individual_rewards: np.ndarray
    shared_rewards: np.ndarray
    dones: torch.Tensor
    action_log_probs: torch.Tensor
    action_entropies: np.ndarray
    values: torch.Tensor
    loss: float
    explained_var: float
    goal_info: torch.Tensor
    collected_blocks_goal: torch.Tensor # 0 or 1
    collected_blocks_incorrect: torch.Tensor # 0 or 1


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, device, use_lstm=False):
        super().__init__()
        self.hidden_size = 192
        self.device = device
        self.use_lstm = use_lstm

        # Initial network structure referenced "Emergent Social Learning via Multi-agent Reinforcement Learning": https://arxiv.org/abs/2010.00581
        self.conv_network = nn.Sequential(
            layer_init(nn.Conv2d(observation_space.shape[0], 32, kernel_size=3, stride=1, padding=0)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)),
            nn.LeakyReLU(),
        ).to(device)
        self.projection_linear = nn.Sequential(
            layer_init(nn.Linear(43264, 192)).to(device),
            nn.Tanh(),
        )
        self.feature_linear = nn.Sequential(
            layer_init(nn.Linear(192 + NUM_COLORS, 192)),
            nn.Tanh(),
            layer_init(nn.Linear(192, 192)),
            nn.Tanh(),
        ).to(device)
        if self.use_lstm:
            self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True).to(device)
        self.policy_network = nn.Sequential(
            layer_init(nn.Linear(self.hidden_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_space.n), std=0.01),
        ).to(device)
        self.value_network = nn.Sequential(
            layer_init(nn.Linear(192, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        ).to(device)
        self.auxiliary_goalinfo_network = nn.Sequential(
            layer_init(nn.Linear(192, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, NUM_COLORS))
        ).to(device)

    def forward(self, x, goal_info, prev_hidden_and_cell_states: tuple | None = None):
        batch_size = x.size(0)

        # Apply conv network
        features = self.conv_network(x)

        # Flatten convolution output channels into linear input
        # New shape: (batch_size, flattened_size)
        features = features.flatten(start_dim=1)
        features = self.projection_linear(features)

        # Append one-hot reward encoding
        features = torch.cat((features, goal_info), dim=1)
        features = self.feature_linear(features)

        if self.use_lstm:
            # Pass through LSTM; add singular sequence length dimension for LSTM input
            features = features.reshape(batch_size, 1, -1)
            if prev_hidden_and_cell_states is not None:
                prev_hidden_states = prev_hidden_and_cell_states[0].reshape(1, batch_size, self.hidden_size)
                prev_cell_states = prev_hidden_and_cell_states[1].reshape(1, batch_size, self.hidden_size)
                prev_hidden_and_cell_states = (prev_hidden_states, prev_cell_states)
            features, (hidden_states, cell_states) = self.lstm(features, prev_hidden_and_cell_states)
            # features: (batch_size, 1, feature_size) --> (batch_size, feature_size)
            features = features[:, -1, ...].squeeze(1)
        else:
            hidden_states, cell_states = torch.zeros((batch_size, self.hidden_size), device=self.device), torch.zeros((batch_size, self.hidden_size), device=self.device)

        return (
            self.policy_network(features),
            self.value_network(features),
            self.auxiliary_goalinfo_network(features),
            (hidden_states, cell_states)
        )

    def get_value(self, x, goal_info, prev_hidden_and_cell_states: tuple | None = None):
        return self.forward(x, goal_info=goal_info, prev_hidden_and_cell_states=prev_hidden_and_cell_states)[1]

    def get_action_and_value(self, x, goal_info, action=None, prev_hidden_and_cell_states: tuple | None = None, sampling_temperature: float = 1.0):
        logits, value, goalinfo_logits, (hidden_states, cell_states) = self(x, goal_info=goal_info, prev_hidden_and_cell_states=prev_hidden_and_cell_states)
        probs = Categorical(logits=logits)
        if action is None:
            sampling_probs = Categorical(logits=logits / sampling_temperature)
            action = sampling_probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value, goalinfo_logits, (hidden_states, cell_states)

def step(
        envs: Sequence[ParallelEnv],
        models: Mapping[str, ActorCritic],
        optimizers: Mapping[str, optim.Optimizer],
        num_steps: int,
        batch_size: int,
        minibatch_size: int,
        gamma: float,
        gae_lambda: float,
        ppo_update_epochs: int,
        norm_advantage: bool,
        clip_param: float,
        clip_vloss: bool,
        entropy_coef: float,
        value_func_coef: float,
        max_grad_norm: float,
        seeds: list[int],
        training_agents: dict[str, bool],
        target_kl: float | None,
        block_penalty_coef: float,
        sampling_temperature: float = 1.0,
        goalinfo_loss_coef: float = 0,
) -> Tuple[dict[str, StepData], int]:
    """
    Implementation is based on https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py and adapted for multi-agent
    """
    observation_space_shapes = {
        agent: envs[0].observation_spaces[agent]["observation"].shape # type: ignore
        for agent in models
    }

    goal_info_shapes = {
        agent: envs[0].observation_spaces[agent]["goal_info"].shape # type: ignore
        for agent in models
    }

    observation_space_shapes = {key: value for key, value in observation_space_shapes.items() if value is not None}
    assert len(observation_space_shapes) == len(models)
    action_space_shapes = {
        agent: envs[0].action_space.shape
        for agent in models
    }
    action_space_shapes = {key: value for key, value in action_space_shapes.items() if value is not None}
    assert len(action_space_shapes) == len(models)

    all_observations = {agent: torch.zeros((num_steps, len(envs)) + observation_space_shapes[agent]).to(models[agent].device) for agent in models}  # shape: (128, 4) + (5, 32, 32) -> (128, 4, 5, 32, 32)

    all_goal_info = {
       agent: torch.zeros((num_steps, len(envs)) + goal_info_shapes[agent]).to(models[agent].device)
       for agent in models
    }
    all_actions = {agent: torch.zeros((num_steps, len(envs)) + action_space_shapes[agent]).to(models[agent].device) for agent in models}
    all_logprobs = {agent: torch.zeros((num_steps, len(envs))).to(models[agent].device) for agent in models}
    all_rewards = {agent: torch.zeros((num_steps, len(envs))).to(models[agent].device) for agent in models}
    all_individual_rewards = {agent: np.zeros((num_steps, len(envs))) for agent in models}
    all_shared_rewards = {agent: np.zeros((num_steps, len(envs))) for agent in models}
    all_dones = {agent: torch.zeros((num_steps, len(envs))).to(models[agent].device) for agent in models}
    all_values = {agent: torch.zeros((num_steps, len(envs))).to(models[agent].device) for agent in models}
    all_entropies = {agent: np.zeros((num_steps, len(envs))) for agent in models}
    all_collect_blocks_goal = {agent: torch.zeros((num_steps, len(envs))) for agent in models} 
    all_collect_blocks_incorrect = {agent: torch.zeros((num_steps, len(envs))) for agent in models} # "Did agent collect a bad block? 0 or 1 for each step, for each env"

    # num_steps + 1 so that indexing by step gives the *input* states at that step
    lstm_hidden_states = {agent: torch.zeros((num_steps + 1, len(envs), models[agent].hidden_size)).to(models[agent].device) for agent in models}
    lstm_cell_states = {agent: torch.zeros((num_steps + 1, len(envs), models[agent].hidden_size)).to(models[agent].device) for agent in models}

    next_observation_dicts, info_dicts = list(zip(*[env.reset(seed=seed, options={"block_penalty_coef": block_penalty_coef}) for env, seed in zip(envs, seeds)])) # [env1{leader:{obs:.., goal_info:..}, follower:{..}} , env2...]

    next_observations = {
        agent: torch.stack([obs_dict[agent]["observation"] for obs_dict in next_observation_dicts])
        for agent in models
    }
    next_goal_info = {
        agent: np.array([obs_dict[agent]["goal_info"] for obs_dict in next_observation_dicts])
        for agent in models
    }
    next_goal_info = {agent: torch.tensor(next_goal_info[agent], dtype=torch.float32).to(models[agent].device) for agent in models}
    next_dones = {agent: torch.zeros(len(envs)).to(models[agent].device) for agent in models}

    next_individual_rewards = {
        agent: np.array([info_dict[agent]["individual_reward"] for info_dict in info_dicts])
        for agent in models
    }

    for step in range(num_steps):
        step_actions = {}

        for agent, model in models.items():
            all_observations[agent][step] = next_observations[agent]
            all_goal_info[agent][step] = next_goal_info[agent]
            all_dones[agent][step] = next_dones[agent]

            with torch.no_grad():
                action, logprob, entropy, value, goalinfo_logits, (hidden_states, cell_states) = model.get_action_and_value(
                    next_observations[agent],
                    next_goal_info[agent],
                    prev_hidden_and_cell_states=(lstm_hidden_states[agent][step], lstm_cell_states[agent][step]),
                    sampling_temperature=sampling_temperature
                )
                step_actions[agent] = action.cpu().numpy()
                lstm_hidden_states[agent][step + 1] = hidden_states  # step + 1 so that indexing by step gives the *input* states at that step
                lstm_cell_states[agent][step + 1] = cell_states  # step + 1 so that indexing by step gives the *input* states at that step

                all_actions[agent][step] = action
                all_logprobs[agent][step] = logprob
                all_values[agent][step] = value.flatten()
                all_entropies[agent][step] = entropy.flatten().cpu().numpy()

        # Convert step_actions from dict of lists to list of dicts
        step_actions = [{agent: step_actions[agent][i] for agent in step_actions} for i in range(len(step_actions[list(models.keys())[0]]))]

        # Step the environment, and return results.
        next_observation_dicts, reward_dicts, terminated_dicts, truncation_dicts, info_dicts, blocks_collected_dicts = list(zip(*[env.step(step_actions[i]) for i, env in enumerate(envs)]))
        
        next_observations = {agent: torch.stack([obs_dict[agent]['observation'] for obs_dict in next_observation_dicts]) for agent in models}
        next_goal_info = {agent: np.array([obs_dict[agent]['goal_info'] for obs_dict in next_observation_dicts]) for agent in models}
        rewards = {agent: np.array([reward_dict[agent] for reward_dict in reward_dicts]) for agent in models}
        next_individual_rewards = {agent: np.array([info_dict[agent]["individual_reward"] for info_dict in info_dicts]) for agent in models}
        next_shared_rewards = {agent: np.array([info_dict[agent]["shared_reward"] for info_dict in info_dicts]) for agent in models}
        for agent in models:
            all_rewards[agent][step] = torch.tensor(rewards[agent]).to(models[agent].device).view(-1)
            all_individual_rewards[agent][step] = next_individual_rewards[agent].reshape(-1)
            all_shared_rewards[agent][step] = next_shared_rewards[agent].reshape(-1)
            
            # blocks_collected_dict is 1 at a time. Append the latest value to appropriate tracker.
            all_collect_blocks_goal[agent][step] = torch.tensor([bloc_dict[agent]["goal"] for bloc_dict in blocks_collected_dicts]).to(models[agent].device)
            all_collect_blocks_incorrect[agent][step] = torch.tensor([bloc_dict[agent]["incorrect"] for bloc_dict in blocks_collected_dicts]).to(models[agent].device)

        next_dones = {agent: np.logical_or([int(terminated[agent]) for terminated in terminated_dicts], [int(truncated[agent]) for truncated in truncation_dicts]) for agent in models}
        num_goals_switched = sum(env.goal_switched for env in envs) # type: ignore

        # Convert to tensors
        next_goal_info = {agent: torch.tensor(next_goal_info[agent], dtype=torch.float32).to(models[agent].device) for agent in models}
        next_dones = {agent: torch.tensor(next_dones[agent], dtype=torch.float32).to(models[agent].device) for agent in models}

    explained_var = {}
    acc_losses = {agent: 0 for agent in models}
    for agent, model in models.items():
        # bootstrap values if not done
        with torch.no_grad():
            next_values = model.get_value(next_observations[agent], next_goal_info[agent], prev_hidden_and_cell_states=(lstm_hidden_states[agent][-1], lstm_cell_states[agent][-1])).reshape(1, -1)
            advantages = torch.zeros_like(all_rewards[agent]).to(model.device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1 - next_dones[agent]
                    nextvalues = next_values
                else:
                    nextnonterminal = 1 - all_dones[agent][t + 1]
                    nextvalues = all_values[agent][t + 1]
                delta = all_rewards[agent][t] + gamma * nextvalues * nextnonterminal - all_values[agent][t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + all_values[agent]

        # flatten the batch
        b_obs = all_observations[agent].reshape((-1,) + observation_space_shapes[agent])  # (-1, 5, xBoundary, yBoundary)
        b_logprobs = all_logprobs[agent].reshape(-1)
        b_goal_info = all_goal_info[agent].reshape((-1,) + goal_info_shapes[agent])
        b_actions = all_actions[agent].reshape((-1,) + action_space_shapes[agent])
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = all_values[agent].reshape(-1)
        b_lstm_hidden_states = lstm_hidden_states[agent].reshape((-1, model.hidden_size))
        b_lstm_cell_states = lstm_cell_states[agent].reshape((-1, model.hidden_size))

        if training_agents[agent]:
            # Optimizing the policy and value network
            b_inds = np.arange(batch_size)
            clipfracs = []
            for epoch in range(ppo_update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue, goalinfo_logits, _ = model.get_action_and_value(
                        b_obs[mb_inds], 
                        goal_info=b_goal_info.long()[mb_inds], 
                        action=b_actions.long()[mb_inds],
                        prev_hidden_and_cell_states=(b_lstm_hidden_states[mb_inds], b_lstm_cell_states[mb_inds]),
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > clip_param).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if norm_advantage:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_param, 1 + clip_param)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -clip_param,
                            clip_param,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    # Auxiliary goalinfo prediction loss
                    ce_loss_func = torch.nn.CrossEntropyLoss()
                    goalinfo_loss = ce_loss_func(goalinfo_logits, b_goal_info.long()[mb_inds].argmax(dim=-1).view(-1))

                    entropy_loss = entropy.mean()
                    loss = pg_loss - entropy_coef * entropy_loss + v_loss * value_func_coef + goalinfo_loss * goalinfo_loss_coef
                    acc_losses[agent] += loss.detach().cpu().item()

                    optimizers[agent].zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizers[agent].step()

                if target_kl is not None and approx_kl > target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var[agent] = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    step_result = {
        agent: StepData(
            observations=all_observations[agent].cpu(),
            goal_info=all_goal_info[agent].cpu(),
            actions=all_actions[agent].cpu(),
            rewards=all_rewards[agent].cpu(),
            individual_rewards=all_individual_rewards[agent],
            shared_rewards=all_shared_rewards[agent],
            dones=all_dones[agent].cpu(),
            action_log_probs=all_logprobs[agent].cpu(),
            action_entropies=all_entropies[agent],
            values=all_values[agent].cpu(),
            loss=acc_losses[agent] / ppo_update_epochs,
            explained_var=explained_var[agent],
            collected_blocks_goal=all_collect_blocks_goal[agent], # Dict[str, Tensor] {"leader": Tensor(num_steps, len(envs)), "follower": Tensor(num_steps, len(envs))} 
            collected_blocks_incorrect=all_collect_blocks_incorrect[agent], # Dict[str, Tensor] {"leader": Tensor(num_steps, len(envs)), "follower": Tensor(num_steps, len(envs))} 
        )
        for agent in models
    }
    return step_result, num_goals_switched


def train(
        run_name: str | None = None,  # Run name for saving results and for weights & biases logging
        resume_iter: int | None = None,  # The iteration from the run to resume. Will look for checkpoints in the folder corresponding to run_name.
        resume_wandb_id: str | None = None,  # W&B run ID to resume from. Required if providing resume_iter.
        leader_only: bool = False,  # If True, configures the environment with a single agent.
        warmstart_leader_path: str | None = None,  # If provided, loads an existing leader checkpoint at the start
        warmstart_follower_path: str | None = None,  # If provided, loads an existing follower checkpoint at the start
        use_lstm: bool = False,  # Whether to use an LSTM in the network architecture
        compile: bool = False,  # If True, uses torch.compile. May not be supported in all environments.
        # Frozen expert leader params
        frozen_leader: bool = False,
        # Block reward parameters
        positive_reward: float = 1.0,  # Reward for correct pickup
        negative_reward: float = -1.0,  # Reward for incorrect pickup
        # Env params
        block_density: float = 0.05,  # Density of blocks populating the environment grid.
        no_block_penalty_until: int = 0,  # The timestep until which block penalty is 0
        full_block_penalty_at: int = 0,  # The timestep at which block penalty reaches 1 (linearly increasing)
        asymmetric: bool = False,  # True if the follower should NOT get goal info
        block_swap_prob: float = ((NUM_COLORS - 1) / NUM_COLORS) * 1/32,  # Probability of swapping the block positions
        reward_shaping_func: str | None = None,  # If provided, the name of reward shaping function to use. Must correspond to a method under ColorMazeRewards.
        reward_shaping_timesteps: int = 0,  # If reward_shaping_func, the number of timesteps to keep reward shaping active for
        reward_shaping_close_threshold: int = 0,  # If reward_shaping_func, the threshold at which two agents are determined to be "close"
        reward_shaping_penalty: int = 0,  # If reward_shaping_func, the penalty applied if the shaping condition is violated
        use_hemisphere: bool = False,  # If True, restricts each agent to half of the grid
        # PPO params
        total_timesteps: int = 500000,  # Total number of environment timesteps to run the PPO training loop for
        learning_rate: float = 1e-4,  # default set from "Emergent Social Learning via Multi-agent Reinforcement Learning"
        num_envs: int = 4,  # Number of environments to collect rollouts in parallel
        num_steps_per_rollout: int = 128,  # Number of steps in each rollout
        gamma: float = 0.99,  # discount factor
        gae_lambda: float = 0.95,  # lambda for general advantage estimation
        num_minibatches: int = 4,  # Number of minibatches to split each batch of rollouts into for updates
        ppo_update_epochs: int = 4,  # Number of epochs to do weight updates for each minibatch
        norm_advantage: bool = True,  # toggle advantage normalization
        clip_param: float = 0.2,  # surrogate clipping coefficient
        clip_vloss: bool = True,  # toggle clipped loss for value function
        entropy_coef: float = 0.01,  # entropy loss coefficient
        value_func_coef: float = 0.5,  # value function loss coefficient
        max_grad_norm: float = 0.5,  # max gradnorm for gradient clipping
        target_kl: float | None = None,  # target KL divergence threshold
        sampling_temperature: float = 1.0,  # temperature to sample with for rollout collection
        goalinfo_loss_coef: float = 0.0,  # coefficient for supervised goal info cross-entropy loss
        # Config params
        save_data_iters: int = 100, # Save data every save_data_iters iterations from num_iterations, calculated below
        checkpoint_iters: int = 0,  # Checkpoint model and optimizer states every checkpoint_iters iterations
        save_only_one_ckpt: bool = False,  # If True, deletes older checkpoints after saving the new one
        debug_print: bool = False,  # Whether to enable debug mode
        log_to_wandb: bool = True,  # Whether to enable logging to weights and biases
        seed: int = 42,  # Random seed
):
    
    if resume_iter:
        assert resume_wandb_id is not None, "Must provide W&B ID to resume from checkpoint"

    if log_to_wandb:
        wandb.init(entity='wandb_id', project='project', name=run_name, resume=('must' if resume_wandb_id else False), id=resume_wandb_id)
    os.makedirs(f'results/{run_name}', exist_ok=True)

    torch.manual_seed(seed)
    env_seeds = [seed + i for i in range(num_envs)]

    batch_size = num_envs * num_steps_per_rollout
    minibatch_size = batch_size // num_minibatches
    num_iterations = total_timesteps // batch_size

    # Conditionally use reward shaping based on args
    if reward_shaping_func:
        reward_shaping_cls = ColorMazeRewards(close_threshold=reward_shaping_close_threshold, penalty=reward_shaping_penalty)
        reward_shaping = getattr(reward_shaping_cls, reward_shaping_func)
        envs = [ColorMaze(leader_only=leader_only, block_density=block_density, asymmetric=asymmetric, reward_shaping_fns=[reward_shaping], block_swap_prob=block_swap_prob, device=DEVICE, positive_reward=positive_reward, negative_reward=negative_reward, is_unique_hemispheres_env=use_hemisphere) for _ in range(num_envs)]
    else:
        envs = [ColorMaze(leader_only=leader_only, block_density=block_density, asymmetric=asymmetric, block_swap_prob=block_swap_prob, device=DEVICE, positive_reward=positive_reward, negative_reward=negative_reward, is_unique_hemispheres_env=use_hemisphere) for _ in range(num_envs)]

    if torch.cuda.device_count() > 1:
        model_devices = {
            'leader': 'cuda:0',
            'follower': 'cuda:1'
        }
    else:
        model_devices = {
            'leader': DEVICE,
            'follower': DEVICE
        }

    # Observation and action spaces are the same for leader and follower
    act_space = envs[0].action_space
    leader_obs_space = envs[0].observation_spaces['leader']
    leader = ActorCritic(leader_obs_space['observation'], act_space, model_devices['leader'], use_lstm=use_lstm)  # type: ignore
    leader_optimizer = optim.Adam(leader.parameters(), lr=learning_rate, eps=1e-5)
    if leader_only:
        models = {'leader': leader}
        optimizers = {'leader': leader_optimizer}
    else:
        follower_obs_space = envs[0].observation_spaces['follower']
        # follower uses LSTM if asymmetric is true
        follower = ActorCritic(follower_obs_space['observation'], act_space, model_devices['follower'], use_lstm=use_lstm) # type: ignore
        follower_optimizer = optim.Adam(follower.parameters(), lr=learning_rate, eps=1e-5)
        models = {'leader': leader, 'follower': follower}
        optimizers = {'leader': leader_optimizer, 'follower': follower_optimizer}

    if resume_iter:
        # Load checkpoint state to resume run
        print(f"Resuming from iteration {resume_iter}")
        for agent_name, model in models.items():
            model_path = f'results/{run_name}/{agent_name}_iteration={resume_iter}.pth'
            optimizer_path = f'results/{run_name}/{agent_name}_optimizer_iteration={resume_iter}.pth'
            state_dict = torch.load(model_path)
            patched_state_dict = {}
            for key in state_dict:
                if "_orig_mod." in key:
                    patched_state_dict[key.replace("_orig_mod.", "")] = state_dict[key]
                else:
                    patched_state_dict[key] = state_dict[key]
            model.load_state_dict(patched_state_dict)
            optimizers[agent_name].load_state_dict(torch.load(optimizer_path))
    else:
        if warmstart_leader_path:
            print(f"Warmstarting leader model from {warmstart_leader_path}")
            state_dict = torch.load(warmstart_leader_path)
            patched_state_dict = {}
            for key in state_dict:
                if "_orig_mod." in key:
                    patched_state_dict[key.replace("_orig_mod.", "")] = state_dict[key]
                else:
                    patched_state_dict[key] = state_dict[key]
            leader.load_state_dict(patched_state_dict)
            optimizer_path = warmstart_leader_path.replace('iteration', 'optimizer_iteration')
            optimizers['leader'].load_state_dict(torch.load(optimizer_path))
        if warmstart_follower_path:
            print(f"Warmstarting follower model from {warmstart_follower_path}")
            state_dict = torch.load(warmstart_follower_path)
            patched_state_dict = {}
            for key in state_dict:
                if "_orig_mod." in key:
                    patched_state_dict[key.replace("_orig_mod.", "")] = state_dict[key]
                else:
                    patched_state_dict[key] = state_dict[key]
            follower.load_state_dict(patched_state_dict)
            optimizer_path = warmstart_follower_path.replace('iteration', 'optimizer_iteration')
            optimizers['follower'].load_state_dict(torch.load(optimizer_path))

    if compile:
        for name, model in models.items():
            models[name] = torch.compile(model, mode='reduce-overhead') # type: ignore

    assert no_block_penalty_until <= full_block_penalty_at
    if full_block_penalty_at == 0:
        penalty_inc_per_step = 0
        get_block_penalty_coef = lambda step: 1
    else:
        penalty_inc_per_step = 1 / (full_block_penalty_at - no_block_penalty_until)
        get_block_penalty_coef = lambda step: 0 if step <= no_block_penalty_until else min(1, penalty_inc_per_step * (step - no_block_penalty_until))

    training_agents = {
        'leader': not frozen_leader,
        'follower': True
    }

    print(f'Running for {num_iterations} iterations using {num_envs} envs with {batch_size=} and {minibatch_size=}')
    print(f'No block penalty until {no_block_penalty_until}, full {negative_reward} penalty after {full_block_penalty_at} timesteps. Increment by {penalty_inc_per_step} per timestep')


    for iteration in tqdm(range(num_iterations), total=num_iterations):
        if resume_iter and iteration <= resume_iter:
            continue

        if reward_shaping_func and iteration * batch_size > reward_shaping_timesteps:
            # Disable reward shaping if timestep threshold is exceeded
            envs = [ColorMaze(leader_only=leader_only, block_density=block_density, asymmetric=asymmetric, block_swap_prob=block_swap_prob, device=DEVICE, positive_reward=positive_reward, negative_reward=negative_reward, is_unique_hemispheres_env=use_hemisphere) for _ in range(num_envs)]

        block_penalty_coef = get_block_penalty_coef(iteration * batch_size)
        step_results, num_goals_switched = step(
            envs=envs,
            models=models,
            optimizers=optimizers,
            num_steps=num_steps_per_rollout,
            batch_size=batch_size,
            minibatch_size=minibatch_size,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ppo_update_epochs=ppo_update_epochs,
            norm_advantage=norm_advantage,
            clip_param=clip_param,
            clip_vloss=clip_vloss,
            entropy_coef=entropy_coef,
            value_func_coef=value_func_coef,
            max_grad_norm=max_grad_norm,
            seeds=env_seeds,
            target_kl=target_kl,
            block_penalty_coef=block_penalty_coef,
            sampling_temperature=sampling_temperature,
            goalinfo_loss_coef=goalinfo_loss_coef,
            training_agents=training_agents
        )

        metrics = {}
        for agent, results in step_results.items(): # type: ignore
            metrics[agent] = {
                'loss': results.loss,
                'explained_var': results.explained_var,
                'reward': results.rewards.sum(dim=0).mean(),  # Sum along step dim and average along env dim
                'individual_reward': results.individual_rewards.sum(axis=0).mean(),
                'positive_individual_reward': (results.individual_rewards > 0).sum(axis=0).mean(),
                'shared_reward': results.shared_rewards.sum(axis=0).mean(),
                'action_entropy': results.action_entropies.mean(),
                'collected_goal_blocks': results.collected_blocks_goal.sum(dim=0).mean(),
                'collected_incorrect_blocks': results.collected_blocks_incorrect.sum(dim=0).mean(),
            }
        metrics['timesteps'] = (iteration + 1) * batch_size
        metrics['num_goals_switched'] = num_goals_switched
        metrics['block_penalty'] = block_penalty_coef

        if log_to_wandb:
            wandb.log(metrics, step=iteration)

        if save_data_iters and iteration % save_data_iters == 0:
            # Transpose observations so the dims are (env, step, ...observation_shape)
            observation_states = step_results['leader'].observations.transpose(0, 1)  # type: ignore 
            # Transpose goal_infos into shape: (env, step, goal_dim)
            goal_infos = step_results['leader'].goal_info.transpose(0, 1) # type: ignore
            for i in range(observation_states.size(0)):
                trajectory = observation_states[i].numpy()
                goal_infos_i = goal_infos[i].numpy()
                os.makedirs(f'trajectories/{run_name}', exist_ok=True)
                np.save(f"trajectories/{run_name}/trajectory_{iteration=}_env={i}.npy", trajectory)
                np.save(f"trajectories/{run_name}/goal_info_{iteration=}_env={i}.npy", goal_infos_i)

        if debug_print:
            print(f"iter {iteration}: {metrics}")

        if checkpoint_iters and iteration % checkpoint_iters == 0:
            print(f"Saving models at epoch {iteration}")
            for agent_name, model in models.items():
                torch.save(model.state_dict(), f'results/{run_name}/{agent_name}_{iteration=}.pth')
                optimizer = optimizers[agent_name]
                torch.save(optimizer.state_dict(), f'results/{run_name}/{agent_name}_optimizer_{iteration=}.pth')

                if save_only_one_ckpt:
                    os.remove(f'results/{run_name}/{agent_name}_iteration={iteration - checkpoint_iters}.pth')
                    os.remove(f'results/{run_name}/{agent_name}_optimizer_iteration={iteration - checkpoint_iters}.pth')

        for i in range(len(envs)):
            env_seeds[i] += len(envs)

    for agent_name, model in models.items():
        print(iteration)
        torch.save(model.state_dict(), f'results/{run_name}/{agent_name}_{iteration=}.pth')
        optimizer = optimizers[agent_name]
        torch.save(optimizer.state_dict(), f'results/{run_name}/{agent_name}_optimizer_{iteration=}.pth')

if __name__ == '__main__':
    Fire(train)
    