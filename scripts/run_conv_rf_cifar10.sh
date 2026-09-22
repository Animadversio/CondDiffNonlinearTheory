#!/bin/bash
#SBATCH -p kempner_h100
#SBATCH -A kempner_binxuwang_lab
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 0-08:00
#SBATCH -J conv_rf_cifar10
#SBATCH -o /n/home12/binxuwang/Github/CondDiffNonlinearTheory/logs/conv_rf_cifar10_%j.out
#SBATCH -e /n/home12/binxuwang/Github/CondDiffNonlinearTheory/logs/conv_rf_cifar10_%j.err

set -euo pipefail

REPO=/n/home12/binxuwang/Github/CondDiffNonlinearTheory
STORE=${STORE_DIR:-/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang}
ARTIFACT_DIR=${ARTIFACT_DIR:-${STORE}/CondDiffNonlinearTheory/conv_rf_cifar10}
MODE=${MODE:-pilot}

mkdir -p "${REPO}/logs" "${ARTIFACT_DIR}"
cd "${REPO}"
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR=/tmp/matplotlib
export XDG_CACHE_HOME=/tmp/xdg-cache
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}" "${XDG_CACHE_HOME}/torch/kernels"

echo "[slurm] job=${SLURM_JOB_ID:-local} mode=${MODE}"
echo "[slurm] host=$(hostname)"
echo "[slurm] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L 2>/dev/null || true

PYTHON=${PYTHON:-/n/home12/binxuwang/.conda/envs/torch2/bin/python}
"${PYTHON}" -c "import torch; print('[torch] cuda', torch.cuda.is_available(), 'count', torch.cuda.device_count()); print('[torch] version', torch.__version__)"

if [[ "${MODE}" == "smoke" ]]; then
    "${PYTHON}" scripts/conv_rf_cifar10.py --smoke --device cuda
elif [[ "${MODE}" == "pilot" ]]; then
    "${PYTHON}" scripts/conv_rf_cifar10.py \
        --tag pilot_${SLURM_JOB_ID:-local} \
        --artifact_dir "${ARTIFACT_DIR}" \
        --n_samples 2000 \
        --n_noise 2 \
        --batch_size 256 \
        --sigmas 0.3 0.45 0.65 0.85 1.2 1.7 2.2 \
        --num_filters 4 16 64 \
        --kernel_sizes 3 5 \
        --seeds 0 \
        --device cuda
elif [[ "${MODE}" == "full" ]]; then
    "${PYTHON}" scripts/conv_rf_cifar10.py \
        --tag full_${SLURM_JOB_ID:-local} \
        --artifact_dir "${ARTIFACT_DIR}" \
        --n_samples 10000 \
        --n_noise 5 \
        --batch_size 256 \
        --sigmas 0.05 0.1 0.2 0.3 0.45 0.65 0.85 1.2 1.7 2.2 3.0 5.0 10.0 \
        --num_filters 1 2 4 8 16 32 64 128 \
        --kernel_sizes 3 5 7 \
        --seeds 0 1 2 \
        --device cuda
else
    echo "Unknown MODE=${MODE}; expected smoke, pilot, or full" >&2
    exit 2
fi

echo "[slurm] done"
