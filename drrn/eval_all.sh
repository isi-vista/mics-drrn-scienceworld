#!/usr/bin/env bash

#SBATCH --job-name=evalDrrnScienceWorld
#SBATCH --account=borrowed
#SBATCH --partition=ephemeral
#SBATCH --qos=ephemeral
#SBATCH --array=0-29
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=0
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=1
#SBATCH --requeue
#SBATCH --output=results/%x-%j.out

set -euo pipefail

#if [[ "$#" -lt 1 ]] || [ "$1" = "--help" ]; then
#  echo "usage: bash $0 output_path [flags for src/generate_gold_splits.py]" >&2
#  exit 1
#fi

if ! command -v java 1>/dev/null 2>&1; then
  echo 'Java not found. Loading from Spack.' >&2
  spack load --only=package openjdk
fi

OUTPUT_PATH=$1
MODEL_PATH=$2
shift 2

hostname

task_output_dir="$OUTPUT_PATH/logs_$SLURM_ARRAY_TASK_ID"
task_model_path="$MODEL_PATH/logs_$SLURM_ARRAY_TASK_ID"
task_model_name=$(ls "$task_model_path"/model*)
task_model_name=${task_model_name##*model}
task_model_name=${task_model_name%%.pt}
echo "Model Path: $task_model_path"
echo "Model Name: $task_model_name"

if [ ! -d "$task_output_dir" ]
then
    echo "Directory $task_output_dir DOES NOT exist. Creating…"
    mkdir "$task_output_dir"
fi

echo "Running task $SLURM_ARRAY_TASK_ID of total $SLURM_ARRAY_TASK_COUNT tasks."
time python eval-on-task.py \
  --output_dir="$task_output_dir" \
  --num_envs=8 \
  --task_idx="$SLURM_ARRAY_TASK_ID" \
  --simplification_str=easy \
  --priority_fraction=0.50 \
  --memory_size=100000 \
  --env_step_limit=100 \
  --eval_set=dev \
  --historySavePrefix="drrn-results" \
  --model_path="$task_model_path" \
  --model_name="$task_model_name" \
  "$@"
