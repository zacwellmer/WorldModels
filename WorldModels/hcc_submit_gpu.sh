#!/bin/sh
#SBATCH --time=24:00:00          # Maximum run time in hh:mm:ss
#SBATCH --nodes=1               # Run on one node
#SBATCH --ntasks=9              # Number of tasks to run, 1CPU per task
#SBATCH --mem=60000             # Maximum memory required (in megabytes)
#SBATCH --job-name=default_479  # Job name (to track progress)
#SBATCH --partition=cse479,cse479_preempt      # Partition on which to run job
#SBATCH --gres=gpu:1            # Don't change this, it requests a GPU
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out

module load singularity
singularity exec --pwd /home/ubuntu/WM/WorldModels/ -B $PWD:/home/ubuntu/WM/WorldModels/ docker://jallen33/wm bash ./launch_scripts/carracing.bash
