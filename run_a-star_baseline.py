import os
from color_maze import ColorMaze
from a_star_policy import AStarAgent
import numpy as np
import tqdm
import time

def run_a_star_simulation(num_runs=5): 
    os.system('cls' if os.name == 'nt' else 'clear')
    final_leader_scores, final_follower_scores = np.zeros(num_runs), np.zeros(num_runs)
    for seed in tqdm.tqdm(range(num_runs)):
        print(f'\STARTING RUN {seed}')
        start_time = time.time()
        env = ColorMaze(seed=seed)
        env.reset(seed=seed)
        a_star_policy_leader = AStarAgent(env.goal_block)
        a_star_policy_follower = AStarAgent()
        env.render()
        leader_score = 0
        follower_score = 0
        steps_per_rollout = 128
        step = 0

        while env.agents and step < steps_per_rollout:
            actions = {
                'leader': a_star_policy_leader(env, agent=env.leader),
                'follower': a_star_policy_follower(env, agent=env.follower)
            }
            _, rewards, _, _, _, _, _ = env.step(actions)
            leader_score += rewards['leader']
            follower_score += rewards['follower']
            if rewards['leader'] == 1:
                a_star_policy_follower.goal_block_color = a_star_policy_leader.goal_block_color

            os.system('cls' if os.name == 'nt' else 'clear')
            step += 1
        
        env.close()  
        print(f"Final leader score at {step} steps: {leader_score}")
        print(f"Final follower score at {step} steps: {follower_score}")

        final_leader_scores[seed] = leader_score
        final_follower_scores[seed] = follower_score

        print(f'\Run {seed} took {round(time.time() - start_time, 2)} seconds.')
        
    final_leader_mean = np.mean(final_leader_scores)
    final_follower_mean = np.mean(final_follower_scores)

    final_leader_std = np.std(final_leader_scores)
    final_follower_std = np.std(final_follower_scores)
    
    print(f"Leader mean score: {final_leader_mean} | Leader score std: {final_leader_std}")
    print(f"Follower mean score: {final_follower_mean} | Follower score std: {final_follower_std}")
    
    
if __name__ == '__main__':
    run_a_star_simulation()
