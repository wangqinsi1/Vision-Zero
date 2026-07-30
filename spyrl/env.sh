# Shared environment for every SpyRL training script. Sourced, never executed directly.
#
# Everything below is overridable from the outside, e.g.
#   SPYRL_MODEL=Qwen/Qwen3-8B SPYRL_OUTPUT_DIR=/mnt/localssd/spyrl bash spyrl/train_summarization.sh

# Resolve the repo root from this file's location so scripts work from any cwd.
SPYRL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SPYRL_DIR}/.." && pwd)"
export REPO_ROOT

# ---------------------------------------------------------------------------- experiment defaults
export SPYRL_MODEL="${SPYRL_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
export SPYRL_NUM_GPUS="${SPYRL_NUM_GPUS:-8}"
export SPYRL_NNODES="${SPYRL_NNODES:-1}"
export SPYRL_TOTAL_EPOCHS="${SPYRL_TOTAL_EPOCHS:-1}"
export SPYRL_SAVE_FREQ="${SPYRL_SAVE_FREQ:-5}"
# "console", "wandb", "tensorboard", ... -- passed verbatim to trainer.logger.
export SPYRL_LOGGER="${SPYRL_LOGGER:-[\"console\"]}"

# ---------------------------------------------------------------------------- output locations
# Checkpoints and rollout dumps. Point this at fast local storage on a cluster.
export SPYRL_OUTPUT_DIR="${SPYRL_OUTPUT_DIR:-${REPO_ROOT}/outputs}"
export SPYRL_CACHE_DIR="${SPYRL_CACHE_DIR:-${SPYRL_OUTPUT_DIR}/cache}"

# ---------------------------------------------------------------------------- SpyRL modules
# verl loads these by file path (data.custom_cls.path / custom_reward_function.path).
export SPYRL_DATASET_DIR="${REPO_ROOT}/verl/utils/dataset"
export SPYRL_REWARD="${REPO_ROOT}/verl/utils/spyrl_reward.py"
export SPYRL_REWARD_NO_SPY="${REPO_ROOT}/verl/utils/spyrl_no_spy_reward.py"

# ---------------------------------------------------------------------------- compile caches
# Keep the chatty JIT caches off shared filesystems; they slow down cold start badly.
export TORCHINDUCTOR_CACHE_DIR="${SPYRL_CACHE_DIR}/torchinductor_${SLURM_PROCID:-0}"
export TRITON_CACHE_DIR="${SPYRL_CACHE_DIR}/triton_${SLURM_PROCID:-0}"
export XDG_CACHE_HOME="${SPYRL_CACHE_DIR}/xdg"
export VLLM_CACHE_ROOT="${XDG_CACHE_HOME}/vllm"
export DG_JIT_CACHE_DIR="${XDG_CACHE_HOME}/deep_gemm"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${VLLM_CACHE_ROOT}" "${DG_JIT_CACHE_DIR}"

# Compile for the local GPU only (90 = H100/H200, 80 = A100). Unset to let torch probe.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-90}"

# ---------------------------------------------------------------------------- logging
export RAY_DEDUP_LOGS=0
export FLASHINFER_LOGLEVEL=3
export FLASHINFER_LOGDEST=stdout

# Called by each training script once it knows its experiment name.
spyrl_setup_experiment() {
    export SPYRL_EXPERIMENT="$1"
    export SPYRL_CKPT_DIR="${SPYRL_OUTPUT_DIR}/${SPYRL_EXPERIMENT}"
    # Human-readable dump of the first game of each step -- read this to sanity-check the game.
    export SPYRL_ROLLOUT_LOG="${SPYRL_CKPT_DIR}/rollouts.txt"
    mkdir -p "${SPYRL_CKPT_DIR}"
}
