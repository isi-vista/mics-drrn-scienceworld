#!/usr/bin/env bash

#SBATCH --job-name=evalDrrnScienceWorld
#SBATCH --account=borrowed
#SBATCH --partition=ephemeral
#SBATCH --qos=ephemeral
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
TASK_ID=$3
shift 3

hostname


task_model_name=$(ls "$MODEL_PATH"/model*)
task_model_name=${task_model_name##*model}
task_model_name=${task_model_name%%.pt}
echo "Model Name: $task_model_name"

if [ ! -d "$OUTPUT_PATH" ]
then
    echo "Directory $OUTPUT_PATH DOES NOT exist. Creating…"
    mkdir "$OUTPUT_PATH"
fi

time python eval-on-task.py \
  --output_dir="$OUTPUT_PATH" \
  --num_envs=8 \
  --task_idx="$TASK_ID" \
  --simplification_str=easy \
  --priority_fraction=0.50 \
  --memory_size=100000 \
  --env_step_limit=100 \
  --eval_set=dev \
  --historySavePrefix="drrn-results" \
  --model_path="$MODEL_PATH" \
  --model_name="$task_model_name"
