import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import Mapping, Sequence
from dataclasses import dataclass
import wandb
from fire import Fire
from tqdm import tqdm
import os
from pettingzoo import ParallelEnv

from color_maze import ColorMaze
from color_maze import ColorMazeRewards

@dataclass
class StepData:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    loss: float
    explained_var: float
    goal_info: torch.Tensor


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        
        # Network structure from "Emergent Social Learning via Multi-agent Reinforcement Learning": https://arxiv.org/abs/2010.00581
        self.conv_network = nn.Sequential(
            layer_init(nn.Conv2d(observation_space.shape[0], 32, kernel_size=3, stride=3, padding=0)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)),
            nn.LeakyReLU(),
            nn.Flatten(start_dim=1),  # flatten all dims except batch-wise 
        )
        self.feature_network = nn.Sequential(
            layer_init(nn.Linear(64*6*6 + 3, 192)),
            nn.Tanh(),
            nn.LSTM(192, 192, batch_first=True)
        )
        self.policy_network = nn.Sequential(
            layer_init(nn.Linear(192, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_space.n), std=0.01),
        )
        self.value_network = nn.Sequential(
            layer_init(nn.Linear(192, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def forward(self, x, goal_info):
        features = self.conv_network(x)
        # add one-hot reward encoding
        features = torch.cat((features, goal_info), dim=1)
        features, (hidden_states, cell_states) = self.feature_network(features)
        return self.policy_network(features), self.value_network(features)

    def get_value(self, x, goal_info):
        return self.forward(x, goal_info=goal_info)[1]

    def get_action_and_value(self, x, goal_info, action=None):
        logits, value = self(x, goal_info=goal_info)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value


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
        target_kl: float | None,
        hist_len: int,
) -> dict[str, StepData]:
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
    
    all_observations = {agent: torch.zeros((num_steps, len(envs)) + observation_space_shapes[agent]).to(DEVICE) for agent in models}
    all_goal_info = {
       agent: torch.zeros((num_steps, len(envs)) + goal_info_shapes[agent]).to(DEVICE)
       for agent in models
    }
    all_actions = {agent: torch.zeros((num_steps, len(envs)) + action_space_shapes[agent]).to(DEVICE) for agent in models}
    all_logprobs = {agent: torch.zeros((num_steps, len(envs))).to(DEVICE) for agent in models}
    all_rewards = {agent: torch.zeros((num_steps, len(envs))).to(DEVICE) for agent in models}
    all_dones = {agent: torch.zeros((num_steps, len(envs))).to(DEVICE) for agent in models}
    all_values = {agent: torch.zeros((num_steps, len(envs))).to(DEVICE) for agent in models}

    # def most_recent_observations(observations):
    #     return torch.cat(torch.repeat(observations[0], hist_len), observations)[-hist_len:]

    next_observation_dicts, _ = list(zip(*[env.reset() for env in envs]))
    # next_observations = {
    #     agent: np.array([obs_dict[agent]["observation"][-hist_len:] for obs_dict in next_observation_dicts])
    #     for agent in models
    # }
    # next_goal_info = {
    #     agent: np.array([obs_dict[agent]["goal_info"][-hist_len:] for obs_dict in next_observation_dicts])
    #     for agent in models
    # }
    next_observations = {
        agent: np.array([obs_dict[agent]["observation"] for obs_dict in next_observation_dicts])
        for agent in models
    }
    next_goal_info = {
        agent: np.array([obs_dict[agent]["goal_info"] for obs_dict in next_observation_dicts])
        for agent in models
    }
    # next_observations = {agent: np.array([obs_dict[agent][-hist_len:] for obs_dict in next_observation_dicts]) for agent in models}
    next_observations = {agent: torch.tensor(next_observations[agent]).to(DEVICE) for agent in models}
    next_goal_info = {agent: torch.tensor(next_goal_info[agent], dtype=torch.float32).to(DEVICE) for agent in models}
    # next_observations = {agent: most_recent_observations(torch.tensor(next_observations[agent].to(DEVICE))) for agent in models}
    next_dones = {agent: torch.zeros(len(envs)).to(DEVICE) for agent in models}

    for step in range(num_steps):
        step_actions = {}

        for agent, model in models.items():
            all_observations[agent][step] = next_observations[agent]
            all_goal_info[agent][step] = next_goal_info[agent]
            all_dones[agent][step] = next_dones[agent]

            with torch.no_grad():
                action, logprob, _, value = model.get_action_and_value(next_observations[agent], next_goal_info[agent])
                step_actions[agent] = action.cpu().numpy()

                all_actions[agent][step] = action
                all_logprobs[agent][step] = logprob
                all_values[agent][step] = value.flatten()

        # Convert step_actions from dict of lists to list of dicts
        step_actions = [{agent: step_actions[agent][i] for agent in step_actions} for i in range(len(step_actions[list(models.keys())[0]]))]

        next_observation_dicts, reward_dicts, terminated_dicts, truncation_dicts, _ = list(zip(*[env.step(step_actions[i]) for i, env in enumerate(envs)]))
        next_observations = {agent: np.array([obs_dict[agent]['observation'] for obs_dict in next_observation_dicts]) for agent in models}
        next_goal_info = {agent: np.array([obs_dict[agent]['goal_info'] for obs_dict in next_observation_dicts]) for agent in models}
        rewards = {agent: np.array([reward_dict[agent] for reward_dict in reward_dicts]) for agent in models}
        for agent in models:
            all_rewards[agent][step] = torch.tensor(rewards[agent]).to(DEVICE).view(-1)
        next_dones = {agent: np.logical_or([int(terminated[agent]) for terminated in terminated_dicts], [int(truncated[agent]) for truncated in truncation_dicts]) for agent in models}

        # Convert to tensors
        next_observations = {agent: torch.tensor(next_observations[agent]).to(DEVICE) for agent in models}
        next_goal_info = {agent: torch.tensor(next_goal_info[agent], dtype=torch.float32).to(DEVICE) for agent in models}
        next_dones = {agent: torch.tensor(next_dones[agent], dtype=torch.float32).to(DEVICE) for agent in models}

    explained_var = {}
    acc_losses = {agent: 0 for agent in models}
    for agent, model in models.items():
        # bootstrap values if not done
        with torch.no_grad():
            next_values = model.get_value(next_observations[agent], next_goal_info[agent]).reshape(1, -1)
            advantages = torch.zeros_like(all_rewards[agent]).to(DEVICE)
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

        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(ppo_update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = model.get_action_and_value(
                    b_obs[mb_inds], 
                    goal_info=b_goal_info.long()[mb_inds], 
                    action=b_actions.long()[mb_inds]
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

                entropy_loss = entropy.mean()
                loss = pg_loss - entropy_coef * entropy_loss + v_loss * value_func_coef
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
            dones=all_dones[agent].cpu(),
            action_log_probs=all_logprobs[agent].cpu(),
            values=all_values[agent].cpu(),
            loss=acc_losses[agent] / ppo_update_epochs,
            explained_var=explained_var[agent]
        )
        for agent in models
    }
    return step_result


def train(
        run_name: str | None = None,
        # PPO params
        total_timesteps: int = 500000,
        learning_rate: float = 1e-4,  # default set from "Emergent Social Learning via Multi-agent Reinforcement Learning"
        num_envs: int = 4,
        num_steps_per_rollout: int = 128,
        gamma: float = 0.99,  # discount factor
        gae_lambda: float = 0.95,  # lambda for general advantage estimation
        num_minibatches: int = 4,
        ppo_update_epochs: int = 4,
        norm_advantage: bool = True,  # toggle advantage normalization
        clip_param: float = 0.2,  # surrogate clipping coefficient
        clip_vloss: bool = True,  # toggle clipped loss for value function
        entropy_coef: float = 0.01,
        value_func_coef: float = 0.5,
        max_grad_norm: float = 0.5,  # max gradnorm for gradient clipping
        target_kl: float | None = None,  # target KL divergence threshold
        # Config params
        save_data_iters: int = 100,
        checkpoint_iters: int = 0,
        debug_print: bool = False,
        log_to_wandb: bool = True,
        hist_len: int = 5,
        seed: int = 42,
):
    if log_to_wandb:
        wandb.init(entity='kavel', project='help-the-human', name=run_name)
    os.makedirs(f'results/{run_name}', exist_ok=True)

    torch.manual_seed(seed)

    batch_size = num_envs * num_steps_per_rollout
    minibatch_size = batch_size // num_minibatches
    num_iterations = total_timesteps // batch_size

    penalize_follower_close_to_leader = ColorMazeRewards(close_threshold=10, timestep_expiry=500).penalize_follower_close_to_leader
    envs = [ColorMaze(history_length=1) for _ in range(num_envs)] # To add reward shaping functions, init as ColorMaze(reward_shaping_fns=[penalize_follower_close_to_leader])

    # Observation and action spaces are the same for leader and follower
    leader_obs_space = envs[0].observation_spaces['leader']
    follower_obs_space = envs[0].observation_spaces['follower']
    act_space = envs[0].action_space

    leader = ActorCritic(leader_obs_space['observation'], act_space).to(DEVICE)  # type: ignore
    follower = ActorCritic(follower_obs_space['observation'], act_space).to(DEVICE) # type: ignore
    leader_optimizer = optim.Adam(leader.parameters(), lr=learning_rate, eps=1e-5)
    follower_optimizer = optim.Adam(follower.parameters(), lr=learning_rate, eps=1e-5)
    models = {'leader': leader, 'follower': follower}
    optimizers = {'leader': leader_optimizer, 'follower': follower_optimizer}

    print(f'Running for {num_iterations} iterations using {num_envs} envs with {batch_size=} and {minibatch_size=}')

    for iteration in tqdm(range(num_iterations), total=num_iterations):
        step_results = step(
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
            target_kl=target_kl,
            hist_len=hist_len
        )

        metrics = {}
        for agent, results in step_results.items():
            metrics[agent] = {
                'loss': results.loss,
                'explained_var': results.explained_var,
                'reward': results.rewards.sum(dim=0).mean()  # Sum along step dim and average along env dim
            }

        if log_to_wandb:
            wandb.log(metrics, step=iteration)

        if save_data_iters and iteration % save_data_iters == 0:
            observation_states = step_results['leader'].observations.transpose(0, 1)  # Transpose so the dims are (env, step, ...observation_shape)
            goal_infos = step_results['leader'].goal_info.transpose(0, 1)
            for i in range(observation_states.size(0)):
                # TODO this will need to be updated once the leader can see true reward. We ought to log it too, to see when it changes during inspection.
                trajectory = observation_states[i].numpy()
                os.makedirs(f'trajectories/{run_name}', exist_ok=True)
                np.save(f"trajectories/{run_name}/trajectory_{iteration=}_env={i}.npy", trajectory)
                np.save(f"trajectories/{run_name}/goal_info_{iteration=}_env={i}.npy", goal_infos)

        if debug_print:
            print(f"iter {iteration}: {metrics}")

        if checkpoint_iters and iteration % checkpoint_iters == 0:
            print(f"Saving models at epoch {iteration}")
            for agent_name, model in models.items():
                torch.save(model.state_dict(), f'results/{run_name}/{agent_name}_{iteration=}.pth')

    for agent_name, model in models.items():
        torch.save(model.state_dict(), f'results/{run_name}/{agent_name}_{iteration=}.pth')


if __name__ == '__main__':
    Fire(train)
