import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import Any, Mapping

from src import color_maze


"""
TODOs

What kind of network architectures?
- add wandb logging
- visualize agent actions
- lab expectations doc / tweet thread


"""
class ActorCritic(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()

        # Network structure from "Emergent Social Learning via Multi-agent Reinforcement Learning": https://arxiv.org/abs/2010.00581
        self.shared_network = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=3, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Flatten(start_dim=1),  # flatten all dims except batch-wise
            nn.Linear(64*6*6, 192),  # TODO check nums
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
):
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
            obs_tensor = torch.tensor(obs[agent], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits, value = model(obs_tensor)
                dist = Categorical(logits=logits)
                action = dist.sample()

            actions[agent] = action.item()
            action_log_probs[agent] = logits
            values[agent] = value.item()

        obs, rewards, terminateds, truncations, _ = env.step(actions)

        for agent in env.agents:
            data[agent].append((obs_tensor, action.item(), rewards[agent], terminateds[agent], action_log_probs[agent], values[agent]))
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
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        # old_states = torch.squeeze(torch.stack(observations, dim=0)).detach()
        # old_actions = torch.squeeze(torch.stack(actions, dim=0)).detach()
        old_logprobs = torch.squeeze(torch.stack(old_log_probs, dim=0)).detach()
        old_values = torch.squeeze(torch.stack([torch.tensor(val) for val in old_values], dim=0)).detach()

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


env = color_maze.ColorMaze()
# Observation and action spaces are the same for leader and follower
obs_space = env.observation_space('leader')
act_space = env.action_space('leader')

SAVE_DATA = False

LR = 1e-4  # from "Emergent Social Learning via Multi-agent Reinforcement Learning"

leader = ActorCritic(obs_space, act_space)
follower = ActorCritic(obs_space, act_space)
leader_optimizer = optim.Adam(leader.parameters(), lr=LR)
follower_optimizer = optim.Adam(follower.parameters(), lr=LR)

num_epochs = 100
num_steps_per_epoch = 1000
ppo_epochs = 4
gamma = 0.99
clip_param = 0.2

models = {'leader': leader, 'follower': follower}
optimizers = {'leader': leader_optimizer, 'follower': follower_optimizer}

for epoch in range(num_epochs):
    metrics = {'leader': {}, 'follower': {}}

    data, sum_rewards = collect_data(env, models, num_steps_per_epoch)
    metrics['leader']['reward'] = sum_rewards['leader']
    metrics['follower']['reward'] = sum_rewards['follower']

    if SAVE_DATA:
        # TODO serialize the episode trajectory for future use
        pass

    losses = ppo_update(models, optimizers, data, ppo_epochs, gamma, clip_param)

    metrics['leader']['loss'] = losses['leader']
    metrics['follower']['loss'] = losses['follower']
    print(f"ep {epoch}: {metrics}")
