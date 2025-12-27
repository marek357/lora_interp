#!/usr/bin/env bash
set -euo pipefail

# Simple launcher to run r/k grid on a single 8-GPU node without Slurm.
# Dynamically picks a free GPU (no active compute process) before launching each job.
# Customize via env vars: R_VALUES, K_VALUES, NUM_GPUS, GPU_LIST, TRAINING_CFG, EXPERIMENT, ROOT, GPU_WAIT_SECONDS, MODEL_ROOT, STATE_DIR.

R_VALUES=(${R_VALUES:-512 1024 2304 4096 8192 16384 32768})
K_VALUES=(${K_VALUES:-1 2 4 8 16 32 64 128 256 512})
# GPU_LIST lets you pin to specific devices (space-separated). If unset, use 0..NUM_GPUS-1
GPU_LIST=(${GPU_LIST:-})
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)}
GPU_WAIT_SECONDS=${GPU_WAIT_SECONDS:-5}
TRAINING_CFG=${TRAINING_CFG:-dpo_fast}
EXPERIMENT=${EXPERIMENT:-topk_grid}
DATE_TAG=$(date +%Y-%m-%d)
ROOT=${ROOT:-outputs/manual_grid}
LOG_DIR="${ROOT}/logs"
RUN_DIR="${ROOT}/runs"
SWEEP_DIR="${ROOT}/sweeps"
MODEL_ROOT=${MODEL_ROOT:-${ROOT}/sweep_models}
STATE_DIR=${STATE_DIR:-${ROOT}/state}

mkdir -p "$LOG_DIR" "$RUN_DIR" "$SWEEP_DIR" "$STATE_DIR"

echo "Launching grid: |R|=${#R_VALUES[@]} x |K|=${#K_VALUES[@]} on ${NUM_GPUS} GPUs"

# Normalize GPU_LIST if provided
if [ ${#GPU_LIST[@]} -gt 0 ]; then
  NUM_GPUS=${#GPU_LIST[@]}
fi

declare -a PIDS=()
job_idx=0

# Return list of busy GPU indices (those with active compute processes)
busy_gpus() {
  nvidia-smi --query-compute-apps=gpu_index --format=csv,noheader 2>/dev/null \
    | sed '/^$/d' | sort -n | uniq
}

is_gpu_free() {
  local target="$1"
  if [ -z "$target" ]; then
    return 1
  fi
  if busy_gpus | grep -qx "$target"; then
    return 1
  fi
  return 0
}

next_free_gpu() {
  while true; do
    if [ ${#GPU_LIST[@]} -gt 0 ]; then
      for gpu in "${GPU_LIST[@]}"; do
        if is_gpu_free "$gpu"; then
          echo "$gpu"
          return 0
        fi
      done
    else
      for gpu in $(seq 0 $((NUM_GPUS - 1))); do
        if is_gpu_free "$gpu"; then
          echo "$gpu"
          return 0
        fi
      done
    fi
    sleep "$GPU_WAIT_SECONDS"
  done
}

launch_job() {
  local r="$1" k="$2" gpu_id="$3"
  local run_name="r${r}_k${k}"
  local log_file="${LOG_DIR}/${run_name}.log"
  local lock_path="${STATE_DIR}/${run_name}.lock"
  local done_path="${STATE_DIR}/${run_name}.done"

  # Skip if already completed
  if [ -f "$done_path" ]; then
    echo "[SKIP existing done] ${run_name}"
    return 0
  fi

  # Try to claim the run atomically; if another node claimed it, skip.
  if ! mkdir "$lock_path" 2>/dev/null; then
    echo "[SKIP claimed elsewhere] ${run_name}"
    return 0
  fi

  echo "[GPU ${gpu_id}] ${run_name} -> ${log_file}"

  CUDA_VISIBLE_DEVICES="${gpu_id}" \
  bash -lc "python main.py \
    training='${TRAINING_CFG}' \
    training.dpo_experiment='${EXPERIMENT}' \
    training.dpo_experiment.lora.r='${r}' \
    training.dpo_experiment.lora.k='${k}' \
    training.dump_path='${MODEL_ROOT}/r${r}_k${k}' \
    hydra.run.dir='${RUN_DIR}/${run_name}' \
    hydra.sweep.dir='${SWEEP_DIR}' \
    experiment_name='${run_name}' \
    && touch '${done_path}' || true; rm -rf '${lock_path}'" \
    >"${log_file}" 2>&1 &

  PIDS+=("$!")
}

for r in "${R_VALUES[@]}"; do
  for k in "${K_VALUES[@]}"; do
    gpu=$(next_free_gpu)
    launch_job "$r" "$k" "$gpu"
    job_idx=$(( job_idx + 1 ))

    # throttle to NUM_GPUS concurrent jobs
    while (( ${#PIDS[@]} >= NUM_GPUS )); do
      wait -n || true
      # prune finished PIDs
      TMP=()
      for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
          TMP+=("$pid")
        fi
      done
      PIDS=("${TMP[@]}")
    done
  done
done

# wait for remaining jobs
wait

echo "All grid jobs finished. Logs: ${LOG_DIR}"