#!/bin/bash

# SBATCH --partition=long    
# SBATCH --mem=20G                                        # Ask for 20 GB of RAM
# SBATCH --time=2:00:00                                   # The job will run for 2 hours


# 1. Load the required modules
module --quiet load anaconda/3

# 2. Load your environment
conda activate "env"

module load cuda/11.3

# 3. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR
python job.py --wandb=<wandb project name> --log_dir=<log dir>/$SLURM_ARRAY_TASK_ID --data_dir=<data_dir> --part=$SLURM_ARRAY_TASK_ID --total_parts=$SLURM_ARRAY_TASK_COUNT