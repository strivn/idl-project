#!/bin/bash
#SBATCH -J fine_exclusion             
#SBATCH -p GPU-shared             
#SBATCH --gres=gpu:v100-32:4      
#SBATCH -t 20:00:00               
#SBATCH --mem=64G                 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jhwang4
#SBATCH --output=%j_exclusion.log
#SBATCH --nodes=1
#SBATCH --no-requeue
#SBATCH --qos=normal

module load AI/pytorch_23.02-1.13.1-py3

cd /ocean/projects/cis250068p/jhwang4/idl-project/notebooks/

papermill fine_exclusion.ipynb \
    ${SLURM_JOB_ID}_fine_exclusion.log
