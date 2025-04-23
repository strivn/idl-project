#!/bin/bash
#SBATCH -J run_exclusion             # 작업 이름
#SBATCH -p GPU-shared             # 사용할 파티션 (예: RM, EM, GPU 등)
#SBATCH --gres=gpu:v100-32:4      # GPU 사용 시
#SBATCH -t 40:00:00               # 최대 실행 시간 (HH:MM:SS)
#SBATCH --mem=64G                 # 메모리 요구량
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jhwang4
#SBATCH --output=%j_exclusion.log
#SBATCH --nodes=1
#SBATCH --no-requeue
#SBATCH --qos=normal

# 필요한 모듈 로드
module load AI/pytorch_23.02-1.13.1-py3

# 작업 디렉토리 이동
cd /ocean/projects/cis250068p/jhwang4/idl-project/notebooks/

# 실행할 명령어
#python run_search.py --search_type linear
# Run the notebook
papermill run_exclusion.ipynb \
    ${SLURM_JOB_ID}_exclusion.log
