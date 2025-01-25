"""
Two players: a leader, and a follower. Multiple colors of blocks.
The leader moves into blocks of a particular color to score.
The follower can do the same. If the follower moves into a different color, score decreases (visible?)

Spawn: leader in top left, follower in bottom right. Blocks in random locations, not over other entities.
Movement: leader and follower share moveset, one grid up, down, left, right.
"""
from gymnasium.spaces import Discrete, Box, MultiDiscrete # Fundamental Spaces - https://gymnasium.farama.org/api/spaces/
from gymnasium.spaces import Dict as DictSpace # Composite Spaces - Dict is best for fixed number of unordered spaces.

from pettingzoo import ParallelEnv

import numpy as np
import torch
from typing import Callable, Tuple, List, Any
from dataclasses import dataclass
from enum import Enum
from copy import copy, deepcopy

class Moves(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class Boundary(Enum):
    # Invariant: bounds are inclusive
    x1 = 0
    y1 = 0
    x2 = 31
    y2 = 31

xBoundary = Boundary.x2.value + 1 - Boundary.x1.value
yBoundary = Boundary.y2.value + 1 - Boundary.y1.value
NUM_COLORS = 3
NUM_MOVES = 4

class IDs(Enum):
    RED = 0
    BLUE = 1
    GREEN = 2
    LEADER = 3
    FOLLOWER = 4

class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    DEFAULT = '\033[0m'

@dataclass 
class Agent:
    '''Agent class to store x and y coordinates of the agent. Automatically creates __init__ and __repr__ methods.'''
    x: int
    y: int

    x_limit_low: int
    x_limit_high: int
    y_limit_low: int
    y_limit_high: int

    def is_legal(self, x:int, y:int) -> bool:
        '''Check if the agent is within the (inclusive!) bounds of the environment.'''
        return self.x_limit_low <= x <= self.x_limit_high and self.y_limit_low <= y <= self.y_limit_high

class ColorMazeRewards():
    '''Class to organize reward functions for the ColorMaze environment.
    
    tldr: 
        To define new reward_shaping_function, go to color_maze.ColorMazeRewards and add it there. 
        To use them, go to run_ppo.py and when initializing a ColorMaze() environment, init as ColorMaze(reward_shaping_fns=[REWARD_SHAPINGS_HERE])
    
    Invariant [!]: All reward shaping functions take args: agents dictionary, rewards dictionary; and return the rewards dictionary.'''

    def __init__(self, close_threshold: int = 10, penalty: float = 0.1) -> None:
        self.close_threshold = close_threshold
        self.penalty = abs(penalty)

    def penalize_follower_close_to_leader(self, agents: dict[str, Agent], rewards, **kwargs):
        '''Penalize the follower if it is close to the leader.'''
        leader = agents["leader"]
        follower = agents["follower"]
        if abs(leader.x - follower.x) + abs(leader.y - follower.y) < self.close_threshold:
            rewards["follower"] -= self.penalty
        return rewards

    def penalize_leader_close_to_follower(self, agents: dict[str, Agent], rewards, **kwargs):
        '''Penalize the leader if it is close to the follower.'''
        leader = agents["leader"]
        follower = agents["follower"]
        if abs(leader.x - follower.x) + abs(leader.y - follower.y) < self.close_threshold:
            rewards["leader"] -= self.penalty
        return rewards
    
    def _harmonic_distance_reward(self, x1, y1, x2, y2):
        '''Reward curve: iff agent not at block, 1/(distance + 1). Else, 0 (regular reward handles intersection).'''
        '''distance = abs(x_goal - x_agent) + abs(y_goal - y_agent)'''
        '''d=0 -> reward=0'''
        '''d=1 -> reward=1/2'''
        '''d=2 -> reward=1/3''' # If harmonic reward does not work, try something steeper.
        distance = torch.abs(x1 - x2) + torch.abs(y1 - y2)
        return 1 / (distance + 1)

    def potential_field(self, agents: dict[str, Agent], rewards, blocks: torch.Tensor, goal_block: IDs, incorrect_penalty_coef: float = 1, **kwargs):
        '''Reward the leader and follower based on their proximity to goal and incorrect blocks. Inspired by (+), (-) electric potential.'''
        assert isinstance(blocks, torch.Tensor)
        discount_factor = 0.2 # The most rewarding* spot (surrounded by 4) is less rewarding than one goal block pickup.
        # *Technically, surrounded by infinitely many goal blocks is most rewarding. Do the math to tune later, this is unlikely and we have 48 hours.
        leader = agents["leader"]
        follower = agents["follower"]
        goal_positions = torch.nonzero(blocks[goal_block.value] == 1, as_tuple=False)
        goal_positions_x = goal_positions[:, 0]
        goal_positions_y = goal_positions[:, 1]
        leader_x_rep = torch.full_like(goal_positions_x, leader.x)
        leader_y_rep = torch.full_like(goal_positions_y, leader.y)
        follower_x_rep = torch.full_like(goal_positions_x, follower.x)
        follower_y_rep = torch.full_like(goal_positions_y, follower.y)
        
        rewards["leader"] += torch.sum(self._harmonic_distance_reward(leader_x_rep, leader_y_rep, goal_positions_x, goal_positions_y) * discount_factor).item()
        rewards["follower"] += torch.sum(self._harmonic_distance_reward(follower_x_rep, follower_y_rep, goal_positions_x, goal_positions_y) * discount_factor).item() # What is item? ChatGPT suggestion
        
        # Now, get incorrect positions (all other slices of 'blocks' except goal_block == 1)
        incorrect_positions = torch.nonzero(torch.any(blocks[:goal_block.value] == 1, dim=0) | torch.any(blocks[goal_block.value + 1:] == 1, dim=0), as_tuple=False)
        
        incorrect_positions_x = incorrect_positions[:, 0]
        incorrect_positions_y = incorrect_positions[:, 1]
        leader_x_rep = torch.full_like(incorrect_positions_x, leader.x)
        leader_y_rep = torch.full_like(incorrect_positions_y, leader.y)
        follower_x_rep = torch.full_like(incorrect_positions_x, follower.x)
        follower_y_rep = torch.full_like(incorrect_positions_y, follower.y)
        rewards["leader"] += incorrect_penalty_coef * torch.sum(self._harmonic_distance_reward(leader_x_rep, leader_y_rep, incorrect_positions_x, incorrect_positions_y) * discount_factor).item()
        rewards["follower"] += incorrect_penalty_coef * torch.sum(self._harmonic_distance_reward(follower_x_rep, follower_y_rep, incorrect_positions_x, incorrect_positions_y) * discount_factor).item()

        return rewards

class ColorMaze(ParallelEnv):
    
    def __init__(self, seed=None, leader_only: bool = False, block_density: float = 0.10, asymmetric: bool = False, 
                 nonstationary: bool = True, reward_shaping_fns: list[Callable]=[], block_swap_prob:float = 2/3*1/32, 
                 is_unique_hemispheres_env:bool=False, device: str = 'cuda', 
                 positive_reward: float = 1.0, negative_reward: float = -1.0):
        """Initializes the environment's random seed and sets up the environment.

        reward_shaping_fns: List of reward shaping function to be applied. The caller will need to import ColorMazeRewards and pass the functions from here.
        """

        # Randomness
        self.seed = seed  # For inspection
        if self.seed is None:
            self.seed = 42
        self.rng = np.random.default_rng(seed=self.seed)
        self.device = device
        
        # Block parameters
        self.goal_block = IDs.RED
        self.prob_block_switch = block_swap_prob # 1/32 is 2x For h=64 #0.01 # Uniformly at random, expect 1 switch every 100 timesteps.
        self.goal_switched = False
        self.block_penalty_coef = 1
        self.nonstationary = nonstationary
        self.negative_reward = negative_reward
        self.positive_reward = positive_reward

        self.is_unique_hemispheres_env = is_unique_hemispheres_env 
        # When True: leader can only exist in left env side, x in [0, xBoundary // 2]. Follower x in [xBoundary // 2, xBoundary].
        if self.is_unique_hemispheres_env:
            self.leader_x_max_boundary = xBoundary // 2
            self.follower_x_min_boundary = (xBoundary // 2) + 1
        else:
            self.leader_x_max_boundary = Boundary.x2.value
            self.follower_x_min_boundary = Boundary.x1.value

        # Agents
        self.leader_only = leader_only
        if leader_only:
            self.possible_agents:List[str] = ["leader"]
            self.agents:List[str] = copy(self.possible_agents)
            self.leader = Agent(Boundary.x1.value, Boundary.y1.value, x_limit_low=Boundary.x1.value, x_limit_high=self.leader_x_max_boundary, y_limit_low=Boundary.y1.value, y_limit_high=Boundary.y2.value)
        else:
            self.possible_agents:List[str] = ["leader", "follower"]
            self.agents:List[str] = copy(self.possible_agents)
            self.leader = Agent(Boundary.x1.value, Boundary.y1.value, x_limit_low=Boundary.x1.value, x_limit_high=self.leader_x_max_boundary, y_limit_low=Boundary.y1.value, y_limit_high=Boundary.y2.value)
            self.follower = Agent(Boundary.x2.value, Boundary.y2.value, x_limit_low=self.follower_x_min_boundary, x_limit_high=Boundary.x2.value, y_limit_low=Boundary.y1.value, y_limit_high=Boundary.y2.value)
        self.asymmetric = asymmetric

        self.action_space = Discrete(NUM_MOVES)  # type: ignore # Moves: Up, Down, Left, Right

        # Blocks - invariant: for all (x, y) coordinates, no two slices are non-zero
        self.blocks = torch.zeros((NUM_COLORS, xBoundary, yBoundary), device=self.device)
        self.block_density = block_density
        
        self._n_channels = self.blocks.shape[0] + 2  # len(self.possible_agents)  # 5: 1 channel for each block color + 1 for each agent
        board_space = Box(low=0, high=1, shape=(self._n_channels, xBoundary, yBoundary), dtype=np.int32)
        self._observation_space = board_space
        goal_block_space = MultiDiscrete([2] * NUM_COLORS)

        if leader_only:
            self.observation_spaces = { # Python dict, not gym spaces Dict.
                "leader": DictSpace({
                    "observation": board_space,
                    "goal_info": goal_block_space
                }), 
            }
        else:
            self.observation_spaces = { # Python dict, not gym spaces Dict.
                "leader": DictSpace({
                    "observation": board_space,
                    "goal_info": goal_block_space
                }), 
                "follower": DictSpace({
                    "observation": board_space,
                    "goal_info": goal_block_space
                })
            }

        # Environment duration
        self.timestep = 0
        self._MAX_TIMESTEPS = 1000

        # Reward shaping
        if len(reward_shaping_fns) > 0:
            assert not self.leader_only, "Reward shaping is not supported for leader_only mode."
        self.reward_shaping_fns = reward_shaping_fns

    def _maybe_randomize_goal_block(self):
        if self.nonstationary and self.rng.random() < self.prob_block_switch:
            other_colors = list(range(NUM_COLORS))
            other_colors.remove(self.goal_block.value)            
            self.goal_block = IDs(self.rng.choice(other_colors))            
            self.goal_switched = True
            
    def _convert_to_observation(self, blocks: torch.Tensor) -> torch.Tensor:
        """
        Converts the internal state of the environment into an observation that can be used by the agent.
        
        The observation is a 3D numpy array where each cell (0 or 1) represents the presence of an object in that cell in the maze.

        Returns:
            numpy.ndarray: The observation array.
        """
        leader_position = torch.zeros((1, xBoundary, yBoundary), device=self.device)
        leader_position[0, self.leader.x, self.leader.y] = 1

        follower_position = torch.zeros((1, xBoundary, yBoundary), device=self.device)
        if not self.leader_only:
            follower_position[0, self.follower.x, self.follower.y] = 1

        # self.blocks.shape is (3, 32, 32)
        # leader_position.shape is (1, 32, 32)
        # follower_position.shape is (1, 32, 32)

        observation = torch.cat((self.blocks, leader_position, follower_position), dim=0)

        # Ensure that observation is a 2d array
        assert observation.shape == (self._n_channels, xBoundary, yBoundary)
        return observation

    def set_state_to_observation(self, observation: np.ndarray):
        """
        Converts the format returned from _convert_to_observation
        into the internal env state representation.
        *Overrides* [!] the current env state with the given observation.
        """
        leader_places = observation[IDs.LEADER.value].reshape((xBoundary, yBoundary))
        follower_places = observation[IDs.FOLLOWER.value].reshape((xBoundary, yBoundary))
        assert leader_places.sum() == 1
        assert self.leader_only or follower_places.sum() == 1
        tensor_observation = torch.tensor(observation, device=self.device)
        self.blocks = tensor_observation[IDs.RED.value : IDs.GREEN.value + 1]
        assert self.blocks.shape == (NUM_COLORS, xBoundary, yBoundary)
        self.leader.x, self.leader.y = np.argwhere(leader_places).flatten()
        if follower_places.sum() == 1:
            self.follower.x, self.follower.y = np.argwhere(follower_places).flatten()
        else:
            self.leader_only = True

    def reset(self, *, seed=None, options=None) -> Tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, Any]]]:
        """Reset the environment to a starting point."""
        if seed is not None:
            self.seed = seed
        else:
            self.seed = 42
        self.rng = np.random.default_rng(seed=self.seed)

        if options is None:
            options = {}
        if "block_penalty_coef" in options:
            assert isinstance(options["block_penalty_coef"], float) or isinstance(options["block_penalty_coef"], int)
            self.block_penalty_coef = abs(options["block_penalty_coef"])

        self.agents = copy(self.possible_agents)
        self.timestep = 0
        self.goal_switched = False

        # Randomize initial locations
        self.leader.x = self.rng.integers(Boundary.x1.value, self.leader_x_max_boundary, endpoint=True)
        self.leader.y = self.rng.integers(Boundary.y1.value, Boundary.y2.value, endpoint=True)
        if not self.leader_only:
            self.follower.x = self.leader.x
            self.follower.y = self.leader.y
            while (self.follower.x, self.follower.y) == (self.leader.x, self.leader.y):
                self.follower.x = self.rng.integers(self.follower_x_min_boundary, Boundary.x2.value, endpoint=True)
                self.follower.y = self.rng.integers(Boundary.y1.value, Boundary.y2.value, endpoint=True)

        self.blocks = torch.zeros((NUM_COLORS, xBoundary, yBoundary), device=self.device)

        # Randomly place X% blocks (in a 32x32, and 10%, 34 blocks of each color)
        n_blocks_each_color = int((xBoundary * yBoundary * self.block_density) // NUM_COLORS)
        if self.is_unique_hemispheres_env:
            # Relies on consume and spawn respawning in same hemisphere if is_unique_hemispheres_env=True.
            n_blocks_each_color_left_hemisphere = n_blocks_each_color // 2
            n_blocks_each_color_right_hemisphere = (n_blocks_each_color // 2) + 1
            left_position_choices = [(x, y) for x in range(xBoundary // 2) for y in range(yBoundary) if (x, y) != (self.leader.x, self.leader.y) and (x, y) != (self.follower.x, self.follower.y)]
            right_position_choices = [(x, y) for x in range(xBoundary // 2 + 1, xBoundary) for y in range(yBoundary) if (x, y) != (self.leader.x, self.leader.y) and (x, y) != (self.follower.x, self.follower.y)]

            left_block_positions = self.rng.choice(left_position_choices, size=(3, n_blocks_each_color_left_hemisphere), replace=False)
            right_block_positions = self.rng.choice(right_position_choices, size=(3, n_blocks_each_color_right_hemisphere), replace=False)
            left_block_positions = np.insert(left_block_positions, [0], [[[0]], [[1]], [[2]]], axis=2).reshape(-1, 3)
            right_block_positions = np.insert(right_block_positions, [0], [[[0]], [[1]], [[2]]], axis=2).reshape(-1, 3)

            self.blocks[left_block_positions[:, 0], left_block_positions[:, 1], left_block_positions[:, 2]] = 1
            self.blocks[right_block_positions[:, 0], right_block_positions[:, 1], right_block_positions[:, 2]] = 1
        else:
            position_choices = [(x, y) for x in range(xBoundary) for y in range(yBoundary) if (x, y) != (self.leader.x, self.leader.y) and (x, y) != (self.follower.x, self.follower.y)]
            block_positions = self.rng.choice(position_choices, size=(3, n_blocks_each_color), replace=False)
            block_positions = np.insert(block_positions, [0], [[[0]], [[1]], [[2]]], axis=2).reshape(-1, 3)
            self.blocks[block_positions[:, 0], block_positions[:, 1], block_positions[:, 2]] = 1

        if self.nonstationary:
            self.goal_block = self.rng.choice(np.array([IDs.RED, IDs.GREEN, IDs.BLUE]))

        observation = self._convert_to_observation(self.blocks)
        goal_info = np.zeros(NUM_COLORS)
        goal_info[self.goal_block.value] = 1

        if self.leader_only:
            observations = {
                "leader": {
                    "observation": observation,
                    "goal_info": goal_info
                },
            }
        else:
            observations = {
                "leader": {
                    "observation": observation,
                    "goal_info": goal_info
                },
                "follower": {
                    "observation": observation,
                    "goal_info": np.zeros_like(goal_info) if self.asymmetric else goal_info
                }
            }

        # Get info, necessary for proper parallel_to_aec conversion
        infos = {a: {"individual_reward": 0} for a in self.agents}
        return observations, infos

    def _consume_and_spawn_block(self, color_idx: int, x: int, y: int, blocks: torch.Tensor):
        blocks[color_idx, x, y] = 0
        # x_high is exclusive
        if self.is_unique_hemispheres_env: # Ensure block is spawned in the same hemisphere.
            if x <= xBoundary // 2:
                x_low = Boundary.x1.value
                x_high = xBoundary // 2 + 1
            else:
                x_low = xBoundary // 2 + 1
                x_high = Boundary.x2.value + 1
        else:
            x_low = Boundary.x1.value
            x_high = Boundary.x2.value + 1

        # Find a different cell that is not occupied (leader, follower, existing block) and set it to this block.
        # Also make sure no other color is present there      
        leader_position = torch.zeros((1, xBoundary, yBoundary), device=self.device)
        leader_position[0, self.leader.x, self.leader.y] = 1

        follower_position = torch.zeros((1, xBoundary, yBoundary), device=self.device)
        if not self.leader_only:
            follower_position[0, self.follower.x, self.follower.y] = 1

        observation = torch.cat((self.blocks, leader_position, follower_position), dim=0)
        zero_indices = torch.argwhere(torch.all((observation[:, x_low:x_high, :] == 0), dim=0))
        i = torch.randint(low=0, high=len(zero_indices), size=(1,))
        x = zero_indices[i, 0] + x_low
        y = zero_indices[i, 1]
        blocks[color_idx, x, y] = 1
        return

    def step(self, actions):
        """
        Takes an action for all agents in environment, and assigns rewards.
        """
        def _move(x, y, action, agent:Agent):
            """
            Always call _move for the leader first in a given timestep. The leader is favored in collisions with follower. 
            """
            new_x, new_y = x, y
            if action == Moves.UP.value and y < agent.y_limit_high:
                new_y += 1
            elif action == Moves.DOWN.value and y > agent.y_limit_low:
                new_y -= 1
            elif action == Moves.LEFT.value and x > agent.x_limit_low:
                new_x -= 1
            elif action == Moves.RIGHT.value and x < agent.x_limit_high:
                new_x += 1
    
            if (new_x, new_y) == (self.leader.x, self.leader.y):
                return x, y
            else:
                return new_x, new_y
        
        leader_action = actions["leader"]
        self.leader.x, self.leader.y = _move(self.leader.x, self.leader.y, leader_action, self.leader)
        if not self.leader_only:
            follower_action = actions["follower"]
            self.follower.x, self.follower.y = _move(self.follower.x, self.follower.y, follower_action, self.follower)

        blocks_collected = {agent: {"goal": 0, "incorrect": 0} for agent in self.agents}

        # Give rewards
        individual_rewards = {}
        shared_reward = 0
        x_pos = [self.leader.x, self.follower.x] if not self.leader_only else [self.leader.x]
        y_pos = [self.leader.y, self.follower.y] if not self.leader_only else [self.leader.y]
        for agent, x, y in zip(self.agents, x_pos, y_pos):
            individual_rewards[agent] = 0

            if self.blocks[self.goal_block.value, x, y]:
                shared_reward += self.positive_reward
                individual_rewards[agent] += self.positive_reward
                self._consume_and_spawn_block(self.goal_block.value, x, y, self.blocks)
                blocks_collected[agent]["goal"] += 1
            else:
                for non_reward_block_idx in [i for i in range(self.blocks.shape[0]) if i != self.goal_block.value]:
                    if self.blocks[non_reward_block_idx, x, y]:
                        shared_reward += self.negative_reward * self.block_penalty_coef
                        individual_rewards[agent] += self.negative_reward * self.block_penalty_coef
                        self._consume_and_spawn_block(non_reward_block_idx, x, y, self.blocks)
                        blocks_collected[agent]["incorrect"] += 1
                        break  # Can't step on two non-rewarding blocks at once

        rewards = {agent: shared_reward for agent in self.agents}

        # Get infos
        # Copy before applying reward shaping so we log the rewards without potential field, etc.
        infos = {
            a: {
                "individual_reward": deepcopy(individual_rewards[a]),
                "shared_reward": deepcopy(rewards[a]),
            } 
            for a in self.agents
        }

        # Apply reward shaping
        for reward_shaping_function in self.reward_shaping_fns:
            rewards = reward_shaping_function(dict({'leader': self.leader, 'follower': self.follower}), rewards, blocks=self.blocks, goal_block=self.goal_block, incorrect_penalty_coef=self.negative_reward)
            individual_rewards = reward_shaping_function(dict({'leader': self.leader, 'follower': self.follower}), individual_rewards, blocks=self.blocks, goal_block=self.goal_block, incorrect_penalty_coef=self.negative_reward)

        # Check termination conditions
        termination = False
        if self.timestep > self._MAX_TIMESTEPS:
            termination = True
        self.timestep += 1

        # Formatting by agent for the return types
        if (self.agents == []):
            assert False
        terminateds = {a: termination for a in self.agents}        
        if termination:
            self.agents = []

        if ('leader' not in terminateds.keys()):
            assert False

        # Maybe update goal block
        self.goal_switched = False
        self._maybe_randomize_goal_block()

        observation = self._convert_to_observation(self.blocks)
        goal_info = np.zeros(NUM_COLORS)
        goal_info[self.goal_block.value] = 1

        if self.leader_only:
            observations = {
                "leader": {
                    "observation": observation,
                    "goal_info": goal_info
                },
            }
        else:
            observations = {
                "leader": {
                    "observation": observation,
                    "goal_info": goal_info
                },
                "follower": {
                    "observation": observation,
                    "goal_info": np.zeros_like(goal_info) if self.asymmetric else goal_info
                }
            }
        truncateds = terminateds
        return observations, individual_rewards, terminateds, truncateds, infos, blocks_collected

    def print_with_goal_color(self, element, goal_index):
        """Print the element with color if it matches the goal index."""
        if element == "L":
            print(f"{Colors.YELLOW}{element}{Colors.DEFAULT}", end="")
        elif element == "F":
            print(f"{Colors.YELLOW}{element}{Colors.DEFAULT}", end="")
        elif element == "R" and goal_index == IDs.RED.value:
            print(f"{Colors.RED}{element}{Colors.DEFAULT}", end="")
        elif element == "G" and goal_index == IDs.GREEN.value:
            print(f"{Colors.GREEN}{element}{Colors.DEFAULT}", end="")
        elif element == "B" and goal_index == IDs.BLUE.value:
            print(f"{Colors.BLUE}{element}{Colors.DEFAULT}", end="")
        else:
            print(element, end="")

    def render(self, current_goal=None):
        """Render the environment."""
        grid = np.full((Boundary.x2.value + 1, Boundary.y2.value + 1), ".")
        leader_symbol = "L"
        follower_symbol = "F"
        for x, y in np.argwhere(self.blocks[IDs.RED.value].cpu().numpy()):
            grid[x, y] = "R"
        for x, y in np.argwhere(self.blocks[IDs.GREEN.value].cpu().numpy()):
            grid[x, y] = "G"
        for x, y in np.argwhere(self.blocks[IDs.BLUE.value].cpu().numpy()):
            grid[x, y] = "B"

        leader_x, leader_y = self.leader.x, self.leader.y
        grid[leader_x, leader_y] = leader_symbol
        if not self.leader_only:
            grid[self.follower.x, self.follower.y] = follower_symbol

        current_goal = self.goal_block.value

        # If current_goal is provided, highlight it in its color
        if current_goal is not None:
            for x, y in np.argwhere(self.blocks[current_goal].cpu().numpy()):
                self.print_with_goal_color(grid[x, y], current_goal)

        # Flip it so y is increasing upwards
        grid = np.flipud(grid.T)
        for row in grid:
            for element in row:
                self.print_with_goal_color(element, current_goal)
            print()  # Print newline after each row

    def set_goal_block(self, goal_block: IDs):
        """Set the goal block for the environment. Used by render only!!! Dangerous otherwise."""
        self.goal_block = goal_block
        