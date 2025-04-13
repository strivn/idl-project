#!/bin/bash
#SBATCH --account=cis250068p
#SBATCH --job-name=citation_finetune_lora 
#SBATCH --output=%j_asr.log
#SBATCH --time=30:00:00 
#SBATCH --nodes=1
#SBATCH -p GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=iwiryadi
#SBATCH --no-requeue
#SBATCH --qos=normal

# Load PyTorch module
module load AI/pytorch_23.02-1.13.1-py3

# Run the notebook
papermill notebooks/finetune_lora.ipynb \
    logs/${SLURM_JOB_ID}_finetune.log
