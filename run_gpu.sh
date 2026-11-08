#!/bin/bash
# Runs the commands of a job list (one per line, from make_jobs.py) on the GPUs of this machine.
# Each GPU takes the next unstarted job as soon as it is free. Logs go to $WORK/logs/<job list>/.
#
# Usage:
#   bash run_gpu.sh $WORK/jobs/main.txt
#
# GPUS="0 1 2 3"   restricts the GPUs used (default: the devices in CUDA_VISIBLE_DEVICES if it is set,
#                  e.g. the MIG slices of a container, otherwise all GPUs listed by nvidia-smi)
# JOBS_PER_GPU=2   runs two jobs per GPU at a time (most runs use under 4 GB of GPU memory, but the
#                  Interaction CNN with 4-px slices needs about 67 GB; check CPU and RAM first, since every job also runs 4 data-loading workers)
#
# Finished runs are skipped, so the same job list can be started again after an interruption
# (delete the folder of an interrupted training run first; see README.md).
set -uo pipefail

JOBFILE="$(realpath "$1")"
WORK="${WORK:?set WORK to the working folder, e.g. export WORK=\$HOME/make}"
if [[ -z "${GPUS:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  GPUS="${CUDA_VISIBLE_DEVICES//,/ }"
fi
GPUS="${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"

JOBS=()
while IFS= read -r LINE; do
  JOBS+=("$LINE")
done < <(grep -v '^[[:space:]]*#' "$JOBFILE" | grep -v '^[[:space:]]*$')
SLOTS=()
for G in $GPUS; do
  for ((J = 0; J < JOBS_PER_GPU; J++)); do SLOTS+=("$G"); done
done

LOGDIR="$WORK/logs/$(basename "$JOBFILE" .txt)"
mkdir -p "$LOGDIR"
COUNTER="$LOGDIR/.next_job"
echo 0 > "$COUNTER"
echo "${#JOBS[@]} jobs on ${#SLOTS[@]} slots (GPUs: $GPUS); logs in $LOGDIR"

rmdir "$COUNTER.lock" 2>/dev/null
next_job() {  # mkdir is atomic, so only one worker at a time reads and advances the counter
  until mkdir "$COUNTER.lock" 2>/dev/null; do sleep 0.2; done
  I=$(cat "$COUNTER")
  echo $((I + 1)) > "$COUNTER"
  rmdir "$COUNTER.lock"
  echo "$I"
}

for ((S = 0; S < ${#SLOTS[@]}; S++)); do
  (
    GPU="${SLOTS[$S]}"
    while true; do
      I=$(next_job)
      ((I < ${#JOBS[@]})) || break
      LOG="$LOGDIR/job$(printf %04d "$I").log"
      echo "${JOBS[$I]}" > "$LOG"
      echo "[$(date +%H:%M)] GPU $GPU: start job $I"
      if CUDA_VISIBLE_DEVICES="$GPU" bash -c "${JOBS[$I]}" >> "$LOG" 2>&1; then
        echo "[$(date +%H:%M)] GPU $GPU: done  job $I"
      else
        echo "[$(date +%H:%M)] GPU $GPU: FAILED job $I (see $LOG)"
      fi
    done
  ) &
done
wait
echo "all jobs finished; failed jobs: $(grep -L -E 'test ROC-AUC|ROC-AUC [0-9]|already (finished|evaluated)' "$LOGDIR"/job*.log | wc -l)"
