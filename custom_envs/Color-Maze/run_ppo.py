import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import Any, Mapping
from dataclasses import dataclass

import wandb
from fire import Fire
from tqdm import tqdm
import os


from replay_trajectory import replay_trajectory

from src import color_maze

@dataclass
class StepData:
    observation: torch.Tensor
    action: int
    reward: int
    terminated: bool
    action_log_prob: torch.Tensor
    value: torch.Tensor

    def __iter__(self):
        return iter((self.observation, self.action, self.reward, self.terminated, self.action_log_prob, self.value))


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class ActorCritic(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()

        # Network structure from "Emergent Social Learning via Multi-agent Reinforcement Learning": https://arxiv.org/abs/2010.00581
        self.shared_network = nn.Sequential(
            nn.Conv2d(observation_space.shape[0], 32, kernel_size=3, stride=3, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Flatten(start_dim=1),  # flatten all dims except batch-wise
            nn.Linear(64*6*6, 192),
            nn.Tanh(),
            nn.LSTM(192, 192, batch_first=True),
        )
        self.policy_network = nn.Sequential(
            nn.Linear(192, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n),
        )
        self.value_network = nn.Sequential(
            nn.Linear(192, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        features, (hidden_states, cell_states) = self.shared_network(x)
        return self.policy_network(features), self.value_network(features)


def collect_data(
        env: color_maze.ColorMaze,
        models: Mapping[str, nn.Module],
        num_steps: int
) -> tuple[dict[str, list[StepData]], dict[str, int]]:
    obs, _ = env.reset()
    data = {agent: [] for agent in env.agents}
    sum_rewards = {agent: 0 for agent in env.agents}
    for _ in range(num_steps):
        action_log_probs = {}
        values = {}
        actions = {}
        for agent in env.agents:
            model = models[agent]
            # Unsqueeze observation to have batch size 1 and flatten the grid into 1-dimension
            obs_tensor = torch.tensor(obs[agent], dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                logits, value = model(obs_tensor)
                dist = Categorical(logits=logits)
                action = dist.sample()

            actions[agent] = action.item()
            action_log_probs[agent] = logits.detach()
            values[agent] = value.item()

        obs, rewards, terminateds, truncations, _ = env.step(actions)

        for agent in env.agents:
            step_data = StepData(
                observation=obs_tensor,
                action=actions[agent],
                reward=rewards[agent],
                terminated=terminateds[agent],
                action_log_prob=action_log_probs[agent],
                value=values[agent]
            )
            data[agent].append(step_data)
            sum_rewards[agent] += rewards[agent]

        if terminateds[env.agents[0]]:
            # The environment terminates for all agents at the same time
            break

    return data, sum_rewards


def ppo_update(
        models: Mapping[str, nn.Module],
        optimizers: Mapping[str, optim.Optimizer],
        data: dict[str, Any],
        epochs: int,
        gamma: float,
        clip_param: float
):
    acc_losses = {model: 0 for model in models}

    for agent, agent_data in data.items():
        rewards = []
        model = models[agent]
        optimizer = optimizers[agent]
        discounted_reward = 0
        observations, actions, observed_rewards, dones, old_log_probs, old_values = zip(*agent_data)


        for obs, action, reward, done, old_log_prob, value in agent_data:
            discounted_reward = reward + gamma * discounted_reward
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        # old_states = torch.squeeze(torch.stack(observations, dim=0)).detach()
        # old_actions = torch.squeeze(torch.stack(actions, dim=0)).detach()
        old_logprobs = torch.squeeze(torch.stack(old_log_probs, dim=0)).detach()
        old_values = torch.squeeze(torch.stack([torch.tensor(val, device=DEVICE) for val in old_values], dim=0)).detach()

        advantages = rewards - old_values
        advantages = advantages.unsqueeze(1)

        for epoch in range(epochs):
            new_logprobs = []
            new_values = []
            for obs in observations:
                new_log_prob, new_value = model(obs)
                new_logprobs.append(new_log_prob)
                new_values.append(new_value)

            new_logprobs = torch.squeeze(torch.stack(new_logprobs, dim=0))
            new_values = torch.squeeze(torch.stack(new_values, dim=0))

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(new_logprobs - old_logprobs.detach())

            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-clip_param, 1+clip_param) * advantages

            # final loss of clipped objective PPO
            loss_func = nn.MSELoss()
            loss = -torch.min(surr1, surr2) + 0.5 * loss_func(new_values, rewards)  # - 0.01 * dist_entropy
            acc_losses[agent] += loss.detach().mean().item()
            
            # take gradient step
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

    return {agent: acc_losses[agent] / (epochs * len(data)) for agent in acc_losses}


def train(
        run_name: str | None = None,
        learning_rate: float = 1e-4,  # default set from "Emergent Social Learning via Multi-agent Reinforcement Learning"
        num_epochs: int = 1000,
        num_steps_per_epoch: int = 1000,
        ppo_epochs: int = 4,
        gamma: float = 0.99,
        clip_param: float = 0.2,
        save_data_epochs: int = 100,
        checkpoint_epochs: int = 0,
        debug_print: bool = False,
        log_to_wandb: bool = True
):
    if log_to_wandb:
        wandb.init(entity='kavel', project='help-the-human', name=run_name)
    os.makedirs(f'results/{run_name}', exist_ok=True)

    env = color_maze.ColorMaze()

    # Observation and action spaces are the same for leader and follower
    obs_space = env.observation_space('leader')
    act_space = env.action_space('leader')

    leader = ActorCritic(obs_space, act_space).to(DEVICE)
    follower = ActorCritic(obs_space, act_space).to(DEVICE)
    leader_optimizer = optim.Adam(leader.parameters(), lr=learning_rate)
    follower_optimizer = optim.Adam(follower.parameters(), lr=learning_rate)
    models = {'leader': leader, 'follower': follower}
    optimizers = {'leader': leader_optimizer, 'follower': follower_optimizer}

    print(f'Training for {num_epochs} epochs of {num_steps_per_epoch} steps each on device={DEVICE}')
    for epoch in tqdm(range(num_epochs), total=num_epochs):
        metrics = {'leader': {}, 'follower': {}}

        data, sum_rewards = collect_data(env, models, num_steps_per_epoch)
        metrics['leader']['reward'] = sum_rewards['leader']
        metrics['follower']['reward'] = sum_rewards['follower']

        trajectory = None
        if save_data_epochs and epoch % save_data_epochs == 0:
            observation_states = [step_data.observation.cpu().numpy() for step_data in data['leader']]
            trajectory = np.concatenate(observation_states, axis=0)  # Concatenate along the batch axis
            os.makedirs(f"trajectories/{run_name}", exist_ok=True)
            np.save(f"trajectories/{run_name}/trajectory_{epoch=}.npy", trajectory)

        losses = ppo_update(models, optimizers, data, ppo_epochs, gamma, clip_param)

        metrics['leader']['loss'] = losses['leader']
        metrics['follower']['loss'] = losses['follower']
        if log_to_wandb:
            wandb.log(metrics, step=epoch)

        if debug_print:
            print(f"ep {epoch}: {metrics}")
            if trajectory is None:
                observation_states = [step_data.observation.cpu().numpy() for step_data in data['leader']]
                trajectory = np.concatenate(observation_states, axis=0)  # Concatenate along the batch axis
            replay_trajectory(trajectory)

        if checkpoint_epochs and epoch % checkpoint_epochs == 0:
            print(f"Saving models at epoch {epoch}")
            torch.save(leader.state_dict(), f'results/{run_name}/leader_{epoch=}.pth')
            torch.save(follower.state_dict(), f'results/{run_name}/follower_{epoch=}.pth')

    torch.save(leader.state_dict(), f'results/{run_name}/leader_{epoch=}.pth')
    torch.save(follower.state_dict(), f'results/{run_name}/follower_{epoch=}.pth')


if __name__ == '__main__':
    Fire(train)
