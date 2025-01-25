# ColorGrid

This repository contains the code, checkpoints, and environment trajectory visualizations for the paper [ColorGrid: A Multi-Agent Environment for Goal Inference and Assistance](https://arxiv.org/abs/2501.10593)

---

## Installation  

1. Clone the repository:  
   ```bash  
   git clone https://github.com/andreyrisukhin/ColorGrid.git
   cd ColorGrid
   conda create -n color_grid python=3.12
   conda activate color_grid
   pip install -r requirements.txt
   ```  

## Usage:

To train a follower agent with a warm-started leader run ```python run_ppo.py --run_name <run_name> --total_timesteps <time_steps> --frozen_leader True --warmstart_leader_path results/test_run/leader_iteration=39061.pth --num_envs 16 --num_steps_per_rollout 128 --save_data_iters 1000 --checkpoint_iters 1000 --ppo_update_epochs 4 --seed 0 --block_density 0.10 --goalinfo_loss_coef 0 --asymmetric --log_to_wandb False --use_lstm --positive_reward 2 --negative_reward 1```

See `run_ppo.py` for a description on the run parameters.

## Contributing

Contact [Ben Caffee](bncaffee@uw.edu) if you have questions.