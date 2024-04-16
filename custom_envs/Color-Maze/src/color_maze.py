"""
Two players: a leader, and a follower. Multiple colors of blocks.
The leader moves into blocks of a particular color to score.
The follower can do the same. If the follower moves into a different color, score decreases (visible?)

Spawn: leader in top left, follower in bottom right. Blocks in random locations, not over other entities.
Movement: leader and follower share moveset, one grid up, down, left, right.
"""

import functools
import random
from copy import copy

import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete, Box, Space, Dict

from pettingzoo import ParallelEnv

from enum import Enum
class Moves(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
class Boundary(Enum):
    # Inv: bounds are inclusive
    x1 = 0
    y1 = 0
    x2 = 5
    y2 = 5
xBoundary = Boundary.x2.value + 1 - Boundary.x1.value
yBoundary = Boundary.y2.value + 1 - Boundary.y1.value
class IDs(Enum):
    RED = 0
    BLUE = 1
    GREEN = 2
    LEADER = 3
    FOLLOWER = 4

# parallel_env = parallel_wrapper_fn(env) # RockPaperScissors had this, referenced by RLlib example.
# We think it's unneeded because ColorMaze extends ParallelEnv.

class ColorMaze(ParallelEnv):
    """The metadata holds environment constants.
    
    The "name" metadata allows the environment to be pretty printed.
    """
    metadata = {
        "name:": "color_maze_v0",
    }

    def __init__(self):
        """The init method takes in environment arguments.
        
        Defines the following attributes:
        - possible agents (leader, follower)
        - leader coordinates (x,y)
        - follower coordinates (x,y)
        - red blocks [(x,y), (x,y), ...]
        - green blocks [(x,y), (x,y), ...]
        - blue blocks [(x,y), (x,y), ...]
        - timestep

        Spaces are defined in the action_space and observation_spaces methods.
        If not overridden, spaces are inferred from self.observation_spaces and self.action_space.
        """

        self.possible_agents = ["leader", "follower"]
        self.leader_x = None
        self.leader_y = None
        self.follower_x = None
        self.follower_y = None
        # Inv: for all (x, y) coordinates, no two slices are non-zero
        self.blocks = np.zeros((3, xBoundary, yBoundary))
        self.timestep = 0
        self._MAX_TIMESTEPS = 1000
        self._action_space = Discrete(4)
        # self._observation_space = Dict({
            # "observation": Box(low=self._RED_ID, high=IDs.FOLLOWER.value, shape=(xBoundary, yBoundary), dtype=np.int32),
            # "action_mask": MultiDiscrete(4 * [2], dtype=np.int32) # [2, 2, 2, 2] represents 4 dimensions, 2 values each as the action space.
        # })
        self._n_channels = 1  # unused at the moment
        self._observation_space = Box(low=0, high=len(IDs), shape=(xBoundary, yBoundary), dtype=np.int32)

        self.observation_spaces = {
            agent: self._observation_space
            for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: self._action_space
            for agent in self.possible_agents
        }

        self.observation_space = lambda agent: self._observation_space
        self.action_space = lambda agent: self._action_space

    def _convert_to_observation(self):
        """
        Converts the internal state of the environment into an observation that can be used by the agent.
        
        The observation is a 2D numpy array where each cell represents the state of that cell in the maze. The possible values are:
        - 0: Empty cell
        - 1: Red block
        - 2: Blue block
        - 3: Green block
        - 4: Leader agent
        - 5: Follower agent
        AHA, this is different from what we were using! AR now transitioning to enum, resolve this 0 as empty. 

        The observation is constructed by first creating 3 separate 2D arrays to represent the red, blue, and green blocks. These are then combined into a single observation array, and the positions of the leader and follower agents are set in the appropriate cells.
        
        Returns:
            numpy.ndarray: The 2D observation array.
        """
        red_vals = np.where(self.blocks[IDs.RED.value], IDs.RED.value, 0)
        blue_vals = np.where(self.blocks[IDs.BLUE.value], IDs.BLUE.value, 0)
        green_vals = np.where(self.blocks[IDs.GREEN.value], IDs.GREEN.value, 0)
        observation = red_vals + blue_vals + green_vals
        observation[self.leader_x, self.leader_y] = IDs.LEADER.value
        observation[self.follower_x, self.follower_y] = IDs.FOLLOWER.value
        # Ensure that observation is a 2d array
        assert observation.ndim == 2
        assert observation.shape == (xBoundary, yBoundary)
        # observation = observation.reshape((xBoundary, yBoundary, self._n_channels))
        return observation.astype(np.int32)

    def reset(self, *, seed=None, options=None):
        """Reset the environment to a starting point.
        
        """

        self.agents = copy(self.possible_agents)
        self.timestep = 0
        # TODO randomize initial locations
        self.leader_x = Boundary.x1.value
        self.leader_y = Boundary.y1.value
        self.follower_x = Boundary.x2.value
        self.follower_y = Boundary.y2.value

        self.blocks = np.zeros((3, xBoundary, yBoundary))
        self._consume_and_spawn_block(IDs.RED.value, 0, 0)
        self._consume_and_spawn_block(IDs.GREEN.value, 0, 0)
        self._consume_and_spawn_block(IDs.BLUE.value, 0, 0)

        self.goal_block = IDs.RED  # TODO to introduce non-stationarity, change this at some point

        observation = self._convert_to_observation()
        # observations = {
            # # Leader starts in bottom left, so can't go down or left
            # 'leader': {'observation': observation, 'action_mask': np.array([1, 0, 0, 1], dtype=np.int32)},
            # # Follower starts in top right, so can't go up or right
            # 'follower': {'observation': observation, 'action_mask': np.array([0, 1, 1, 0], dtype=np.int32)}
        # }
        observations = {
            agent: observation
            for agent in self.agents
        }

        # Get dummy info, necessary for proper parallel_to_aec conversion
        infos = {a: {} for a in self.agents}
        return observations, infos

    def _consume_and_spawn_block(self, color_idx, x, y) -> None:
        self.blocks[color_idx, x, y] = 0
        # Find a different cell with value 0 and set it to 1
        # Also make sure no other color is present there
        
        zero_indices = np.argwhere(np.all((self.blocks == 0), axis=0))
        np.random.shuffle(zero_indices)
        for new_xy in zero_indices:
            if ((new_xy[0] == self.leader_x and new_xy[1] == self.leader_y) or
                (new_xy[0] == self.leader_x and new_xy[1] == self.leader_y)):
                continue

            self.blocks[color_idx, new_xy[0], new_xy[1]] = 1
            return
        assert False, "No cell with value 0 found to update."

    def step(self, actions):
        """
        Takes an action for current agent (specified by agent selection)

        Update the following:
        - timestep
        - infos
        - rewards
        - leader x and y
        - follower x and y
        - terminations
        - truncations
        - Any internal state used by observe() or render()
        """
        leader_action = actions["leader"]
        follower_action = actions["follower"]

        def _move(x, y, action):
            """
            Always call _move for the leader first in a given timestep
            """
            new_x, new_y = x, y
            if action == Moves.UP.value and y < Boundary.y2.value:
                new_y += 1
            elif action == Moves.DOWN.value and y > Boundary.y1.value:
                new_y -= 1
            elif action == Moves.LEFT.value and x > Boundary.x1.value:
                new_x -= 1
            elif action == Moves.RIGHT.value and x < Boundary.x2.value:
                new_x += 1
    
            if (new_x, new_y) == (self.leader_x, self.leader_y):
                return x, y
            else:
                return new_x, new_y
        
        self.leader_x, self.leader_y = _move(self.leader_x, self.leader_y, leader_action)
        self.follower_x, self.follower_y = _move(self.follower_x, self.follower_y, follower_action)

        # Make action masks
        leader_action_mask = np.ones(4)
        follower_action_mask = np.ones(4)
        for action_mask, x, y in zip([leader_action_mask, follower_action_mask], [self.leader_x, self.follower_x], [self.leader_y, self.follower_y]):
            if x == Boundary.x1.value:
                action_mask[Moves.LEFT.value] = 0  # cant go left
            if x == Boundary.x2.value:
                action_mask[Moves.RIGHT.value] = 0  # cant go right
            if y == Boundary.y1.value:
                action_mask[Moves.DOWN.value] = 0  # cant go down
            if y == Boundary.y2.value:
                action_mask[Moves.UP.value] = 0  # cant go up

        # Give rewards
        shared_reward = 0
        for agent, x, y in zip(["leader", "follower"], [self.leader_x, self.follower_x], [self.leader_y, self.follower_y]):
            if self.blocks[self.goal_block.value, x, y]:
                shared_reward = 1
                self._consume_and_spawn_block(self.goal_block.value, x, y)
            else:
                for non_reward_block_idx in [i for i in range(self.blocks.shape[0]) if i != self.goal_block.value]:
                    if self.blocks[non_reward_block_idx, x, y]:
                        shared_reward = -1
                        self._consume_and_spawn_block(non_reward_block_idx, x, y)
                        break

        rewards = {'leader': shared_reward, 'follower': shared_reward}

        # Check termination conditions
        termination = False
        if self.timestep > self._MAX_TIMESTEPS:
            termination = True
        self.timestep += 1

        # Get dummy infos (not used in this example)
        infos = {a: {} for a in self.agents}


        # Formatting by agent for the return types
        terminateds = {a: termination for a in self.agents}

        if termination:
            self.agents = []

        observation = self._convert_to_observation()
        # observations = {
            # 'leader': {'observation': observation, 'action_mask': leader_action_mask},
            # 'follower': {'observation': observation, 'action_mask': follower_action_mask}
        # }
        observations = {
            agent: observation
            for agent in self.agents
        }
        truncateds = terminateds
        return observations, rewards, terminateds, truncateds, infos

    def render(self):
        """Render the environment."""
        grid = np.full((Boundary.x2.value + 1, Boundary.y2.value + 1), " ")
        grid[self.leader_x, self.leader_y] = "L"
        grid[self.follower_x, self.follower_y] = "F"
        for x, y in np.argwhere(self.blocks[IDs.RED.value]):
            grid[x, y] = "R"
        for x, y in np.argwhere(self.blocks[IDs.GREEN.value]):
            grid[x, y] = "G"
        for x, y in np.argwhere(self.blocks[IDs.BLUE.value]):
            grid[x, y] = "B"

        # Flip it so y is increasing upwards
        grid = np.flipud(grid.T)
        print(grid)
