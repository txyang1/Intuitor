#!/bin/bash -l
#
# ==== 资源与作业基本信息 ====
#SBATCH --job-name=math_grpo_meanreward_0908
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:4
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
# 如需特定分区或账号，取消注释并修改：
# #SBATCH --partition=tinygpu
# #SBATCH --account=your_account
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# ==== 环境导出策略（按你们文档要求）====
#SBATCH --export=NONE      # 不从提交端继承环境
unset SLURM_EXPORT_ENV    # 但允许本脚本环境传给 srun

set -euxo pipefail
mkdir -p logs

# 1) 模块与 Conda
module purge
module load python
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
fi
conda activate intuitor

# 2) 代理
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export HTTP_PROXY=$http_proxy
export HTTPS_PROXY=$https_proxy
export NO_PROXY=localhost,127.0.0.1,::1

# 3) 训练所需环境变量
unset ROCR_VISIBLE_DEVICES
export ACCELERATE_LOG_LEVEL=info
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
# 若不想把 WANDB Key 写入脚本，可在提交时用 --export 传入（见下文）
# export WANDB_API_KEY=***your_key***

# 4) 线程/通信（可选但常用）
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$((12000 + RANDOM % 10000))
# export NCCL_DEBUG=INFO

# 5) 启动作业（让 srun 明确继承 CPU/GPU 配额）
srun --ntasks=1 --gres=gpu:a100:4 --cpus-per-task=$SLURM_CPUS_PER_TASK \
     bash math_grpo.sh
#sbatch --export=WANDB_API_KEY=7526e35bcef1a5880516f1a32352cd3e6c5a4d8a run_math_grpo2.sh
#squeue -u $USER
#scancel <jobid>         # 只取消一个
#scancel -u $USER        # 取消我所有作业（谨慎）