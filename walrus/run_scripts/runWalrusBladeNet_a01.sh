#!/bin/bash
#SBATCH -J WalrusBladeNet
#SBATCH -N 1
#SBATCH -p a01
#SBATCH -o /WORK/PUBLIC/xuqy_work/run/log/WalrusBladeNet/%x-%j.out
#SBATCH -e /WORK/PUBLIC/xuqy_work/run/log/WalrusBladeNet/%x-%j.err
#SBATCH --no-requeue
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00

set -eo pipefail

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "请使用 sbatch 提交此脚本。" >&2
    exit 1
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH=/WORK/PUBLIC/xuqy_work/walrus:${PYTHONPATH:-}
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

source /WORK/PUBLIC/xuqy_work/miniconda3/etc/profile.d/conda.sh
conda activate walrus
export WALRUS_PYTHON="$CONDA_PREFIX/bin/python"

PROJECT_PATH="/WORK/PUBLIC/xuqy_work/walrus"
CONFIG_PATH="$PROJECT_PATH/runs/bladenet_setup/finetune.yaml"
RUN_DIR="$PROJECT_PATH/runs/bladenet_a01_${SLURM_JOB_ID}"

cd "$PROJECT_PATH" || { echo "无法进入目录 $PROJECT_PATH"; exit 1; }

echo " Current Directory   : $(pwd)"
echo " Start Time          : $(date)"
echo " SLURM Node          : ${SLURM_NODELIST:-unknown}"
echo " CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}"
echo " Config              : $CONFIG_PATH"
echo " Output              : $RUN_DIR"
echo " GPU Info            :"
nvidia-smi
echo "================ Starting Walrus BladeNet Training ================"

set +e
bash ./walrus/run_scripts/finetune_bladenet.sh "$CONFIG_PATH" \
    "folder_override=$RUN_DIR" \
    "hydra.run.dir=$RUN_DIR/hydra" \
    "$@"
exit_code=$?
set -e

echo "================ EXIT CODE: ${exit_code} ================"
echo " End Time            : $(date)"
exit "$exit_code"
