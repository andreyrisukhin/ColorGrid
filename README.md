# ColorGrid

This repository contains the code, checkpoints, and environment trajectory visualizations for the paper [ColorGrid: A Multi-Agent Environment for Goal Inference and Assistance](https://arxiv.org/abs/2501.10593).  Andrey Risukhin<sup>∗</sup> Kavel Rao<sup>∗</sup> Ben Caffee<sup>†</sup> Alan Fan<sup>†</sup>, University of Washington.

- ∗ and † denote equal contribution
- Correspondence to Ben Caffee (bncaffee@uw.edu).

---

## Installation  

Clone the repository:  
   ```bash  
   git clone https://github.com/andreyrisukhin/ColorGrid.git
   cd ColorGrid
   conda create -n color_grid python=3.12
   conda activate color_grid
   pip install -r requirements.txt
   ```  

## Usage:

1. Download the [pre-trained weights](https://drive.google.com/drive/folders/1Z3-yM7rk6VPaNrFpyo65PGeZLGa4VwYW?usp=sharing) and place them into the project directory.

2. To train a follower agent with a "warm-started" leader (pre-trained weight) run ```python run_ppo.py --run_name <run_name> --total_timesteps <time_steps> --frozen_leader True --warmstart_leader_path leader_iteration=39061.pth --num_envs 16 --num_steps_per_rollout 128 --save_data_iters 1000 --checkpoint_iters 1000 --ppo_update_epochs 4 --seed 0 --block_density 0.10 --goalinfo_loss_coef 0 --asymmetric --log_to_wandb False --use_lstm --positive_reward 2 --negative_reward 1```

See `run_ppo.py` for a description on the run parameters and other files for running baselines, manually playtesting the environment, etc.

## Contributing

Please feel free to make pull requests.
