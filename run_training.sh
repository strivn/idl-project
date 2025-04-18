#!/bin/bash
#SBATCH --account=cis250068p
#SBATCH --job-name=citation_lora_finetuning
#SBATCH --output=logs/%j_clf.log
#SBATCH --time=4:00:00 
#SBATCH --nodes=1
#SBATCH -p GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=iwiryadi
#SBATCH --no-requeue
#SBATCH --qos=normal

# Load PyTorch module
module load AI/pytorch_23.02-1.13.1-py3

# Run the notebook
papermill notebooks/latest_finetune.ipynb \
    logs/${SLURM_JOB_ID}_papermill_clf.log
