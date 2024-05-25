from pettingzoo.test import parallel_api_test
import os

from color_maze import ColorMaze
from manual_policy import ManualPolicy
from a_star_policy import AStarAgent


if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    env = ColorMaze(is_unique_hemispheres_env=False, nonstationary=False)
    observations, _ = env.reset(seed=42)
    manual_policy_1 = ManualPolicy(env, agent_id=1)
    # manual_policy_2 = ManualPolicy(env, agent_id=1)
    a_star_policy = AStarAgent(env, agent_id=0)
    env.render()
    print(env.goal_block)
    score = 0

    steps_per_rollout = 128
    step = 0

    while env.agents and step < steps_per_rollout:
        actions = {
            'leader': a_star_policy(observations[env.agents[0]], agent=env.agents[0]),
            'follower': manual_policy_1(observations[env.agents[1]], agent=env.agents[1])
        }
        observations, rewards, _, _, _ = env.step(actions)
        score += rewards['leader']
        # don't we want score += rewards['follower'] as well?

        os.system('cls' if os.name == 'nt' else 'clear')
        env.render()
        print(f'Score: {score} | Goal: {env.goal_block} | Step: {step}/{steps_per_rollout} | Reward Shaping: {env.reward_shaping_fns}') # The reward functions will probably not print nicely. TODO print their names.
        step += 1
    env.close()
