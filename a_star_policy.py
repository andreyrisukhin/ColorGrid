import heapq
import numpy as np
import color_maze
import torch

def heuristic(agent_pos, goal_pos):
        """
        Manhattan distance as the heuristic function
        """
        return torch.sum(torch.abs(agent_pos - goal_pos), dim=-1)
        
def a_star_search(env, agent):
    """
    Performs A* search to find the shortest path for the leader to the closest correct block
    """
    blocks = env.blocks
    start_pos = torch.tensor([agent.x, agent.y], device=env.device)
    goal_block = env.goal_block.value
    goal_positions = torch.nonzero(blocks[goal_block] == 1, as_tuple=True)
    
    if len(goal_positions[0]) == 0:
        assert False, "No correct blocks found"

    # Calculate the closest correct block to the leader
    goal_positions_tensor = torch.stack(goal_positions, dim=1)
    goal_dists = heuristic(start_pos.repeat(goal_positions_tensor.shape[0], 1), goal_positions_tensor)
    closest_goal_idx = torch.argmin(goal_dists)
    goal_pos = goal_positions_tensor[closest_goal_idx]

    # Initialize the open and closed sets
    open_set = [(0 + heuristic(start_pos, goal_pos).item(), start_pos.tolist(), [])]
    heapq.heapify(open_set)
    closed_set = set()

    # Loop until the open set is empty or the goal is reached
    while open_set:
        current_cost, current_pos, current_path = heapq.heappop(open_set)

        # If the goal is reached, return the path
        if current_pos == list(goal_pos.tolist()):
            return current_path

        closed_set.add(tuple(current_pos))

        # Generate the neighbors
        neighbors = [
            (current_pos[0], current_pos[1] + 1),
            (current_pos[0], current_pos[1] - 1),
            (current_pos[0] + 1, current_pos[1]),
            (current_pos[0] - 1, current_pos[1]),
        ]

        for neighbor in neighbors:
            neighbor_pos = torch.tensor(neighbor, dtype=torch.long, device=env.device)
            # Check if the neighbor is within the boundaries and not a wall
            if agent.is_legal(neighbor_pos[0].item(), neighbor_pos[1].item()):
                
                # if neighbor_pos isn't a goal block, penalize
                penalize = False
                for penalty_color in set((color_maze.IDs.BLUE, color_maze.IDs.RED, color_maze.IDs.GREEN)) - set((env.goal_block, )):
                    if blocks[penalty_color.value][neighbor_pos[0], neighbor_pos[1]] == 1:
                        new_cost = 999999
                        penalize = True
                    
                if not penalize:
                    new_cost = current_cost - heuristic(torch.tensor(current_pos, device=env.device), goal_pos).item() + 1 + heuristic(neighbor_pos, goal_pos).item()
                new_path = current_path + [tuple(neighbor_pos.tolist())]

                # Check if the neighbor is in the closed set or the open set
                if tuple(neighbor_pos.tolist()) in closed_set:
                    continue
                in_open_set = False
                for idx, entry in enumerate(open_set):
                    if entry[1] == neighbor_pos.tolist():
                        in_open_set = True
                        if entry[0] > new_cost:
                            open_set[idx] = (new_cost, neighbor_pos.tolist(), new_path)
                        break

                if not in_open_set:
                    heapq.heappush(open_set, (new_cost, neighbor_pos.tolist(), new_path))

    # Inv: Path should never be empty
    assert False, "No path found"
    
def get_move(cur_pos, next_cell):
    x1, y1 = cur_pos
    x2, y2 = next_cell
    if x1 < x2:
        return color_maze.Moves.RIGHT.value
    elif x1 > x2:
        return color_maze.Moves.LEFT.value
    elif y1 < y2:
        return color_maze.Moves.UP.value
    else:
        return color_maze.Moves.DOWN.value

class AStarAgent:
    "Uses A* search to find the shortest path to the closest correct block for the leader."
    def __init__(self, initial_goal_block_color=None):
        self.goal_block_color = initial_goal_block_color
        
    def __call__(self, env, agent):         
        cur_pos = agent.x, agent.y
        if self.goal_block_color == None:
            # check if the surrounding blocks are empty
            blocks = env.blocks
            
            neighbors = [
                (cur_pos[0], cur_pos[1] + 1),
                (cur_pos[0], cur_pos[1] - 1),
                (cur_pos[0] + 1, cur_pos[1]),
                (cur_pos[0] - 1, cur_pos[1]),
            ]

            for neighbor in neighbors:
                neighbor_pos = torch.tensor(neighbor, dtype=torch.long, device=env.device)
                # Check if the neighbor is within the boundaries and not a wall
                if agent.is_legal(neighbor_pos[0].item(), neighbor_pos[1].item()):
                    # Check if neighbor is an empty cell
                    neighbor_is_empty = True
                    for block_id in color_maze.IDs.RED, color_maze.IDs.BLUE, color_maze.IDs.GREEN:
                        if blocks[block_id.value][neighbor_pos[0], neighbor_pos[1]] == 1:
                            neighbor_is_empty = False
                            break

                    if neighbor_is_empty:
                        return get_move(cur_pos, neighbor_pos)
                
            # random move if all four neighbors are not empty
            return np.random.choice([color_maze.Moves.UP.value, color_maze.Moves.DOWN.value, color_maze.Moves.LEFT.value, color_maze.Moves.RIGHT.value])
        
        next_cell = a_star_search(env, agent)[0] # we only need the next step to move
        # Optimization is to only run a_star_search if the goal color just switched or a block was just picked up
        return get_move(cur_pos, next_cell)
