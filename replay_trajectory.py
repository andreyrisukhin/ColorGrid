import numpy as np
from fire import Fire

from color_maze import ColorMaze, IDs, Colors

# Function to print text with color based on the goal index
def print_goal_with_color(goal_idx):
    if goal_idx == IDs.RED.value:
        color = Colors.RED
    elif goal_idx == IDs.BLUE.value:
        color = Colors.BLUE
    elif goal_idx == IDs.GREEN.value:
        color = Colors.GREEN
    else:
        color = Colors.DEFAULT  # Default color if goal index doesn't match known colors
    return color + str(IDs(goal_idx).name) + Colors.DEFAULT

def replay_trajectory(trajectory: np.ndarray, goal_info: np.ndarray | None = None):
    assert trajectory.ndim == 4  # (minibatch_step, channel, height, width) 
    assert goal_info is None or goal_info.ndim == 2 # (minibatch step, goal_dim = 3)

    env = ColorMaze()
    env.reset()
    for step in range(trajectory.shape[0]):
        obs = trajectory[step]
        print(f"Step {step}:")
        env.set_state_to_observation(obs)
        if goal_info is not None:
            goal_idx = np.argmax(goal_info[step])
            env.set_goal_block(IDs(goal_idx))
            env.render()
            print(f'Current goal: {print_goal_with_color(goal_idx)}')
        input("Press any key to continue:")


def main(trajectory_filepath: str, goal_info_filepath: str | None = None):
    trajectory = np.load(trajectory_filepath)
    if goal_info_filepath is None:
        goal_info_filepath = trajectory_filepath.replace('trajectory_', 'goal_info_')
    goal_info = np.load(goal_info_filepath)
    replay_trajectory(trajectory, goal_info)


if __name__ == '__main__':
    Fire(main)
