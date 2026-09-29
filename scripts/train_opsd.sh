#!/usr/bin/env bash
# Single-round full-parameter on-policy self-distillation.
set -euo pipefail

: "${STUDENT_MODEL:?Set STUDENT_MODEL to a local model directory or model ID}"
: "${DATA_PATH:?Set DATA_PATH to the training JSONL file}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for the checkpoint}"

PACKAGE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
TEACHER_MODEL="${TEACHER_MODEL:-$STUDENT_MODEL}"
EPOCHS="${EPOCHS:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
MICRO_TRAIN_BATCH_SIZE="${MICRO_TRAIN_BATCH_SIZE:-8}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
TEACHER_UPDATE_FREQ="${TEACHER_UPDATE_FREQ:-999999}"
GENERATE_MAX_LEN="${GENERATE_MAX_LEN:-48}"
PROMPT_MAX_LEN="${PROMPT_MAX_LEN:-2048}"
MAX_LEN="${MAX_LEN:-2112}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"

mkdir -p "$OUTPUT_DIR/checkpoints"
export PYTHONPATH="$PACKAGE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export KDFLOW_MAX_IMAGE_PIXELS="${KDFLOW_MAX_IMAGE_PIXELS:-262144}"
export KDFLOW_ROLLOUT_DIRECT_WORKER=1
export KDFLOW_ROLLOUT_STRICT_HEALTH=1
export KDFLOW_ROLLOUT_SLEEP_DELAY=0
export TOKENIZERS_PARALLELISM=false

env -u PYTORCH_CUDA_ALLOC_CONF "$PYTHON_BIN" -m kdflow.cli.train_kd_on_policy \
  --num_nodes 1 --num_gpus_per_node 1 --backend fsdp2 \
  --train_batch_size "$TRAIN_BATCH_SIZE" \
  --micro_train_batch_size "$MICRO_TRAIN_BATCH_SIZE" \
  --learning_rate 2e-6 --lr_warmup_ratio 0.0 --num_epochs "$EPOCHS" \
  --save_steps 0 --save_path "$OUTPUT_DIR/checkpoints" \
  --bf16 True --gradient_checkpointing True \
  --student_name_or_path "$STUDENT_MODEL" \
  --teacher_name_or_path "$TEACHER_MODEL" \
  --attn_implementation "$ATTN_IMPLEMENTATION" \
  --lora_rank 0 --enable_thinking False \
  --rollout_batch_size "$ROLLOUT_BATCH_SIZE" \
  --rollout_num_engines 1 --rollout_tp_size 1 \
  --rollout_engine_concurrency 1 \
  --rollout_mem_fraction_static 0.25 \
  --temperature 1.0 --top_p 1.0 --n_samples_per_prompt 1 \
  --generate_max_len "$GENERATE_MAX_LEN" \
  --train_dataset_path "$DATA_PATH" \
  --max_len "$MAX_LEN" --prompt_max_len "$PROMPT_MAX_LEN" \
  --input_key student_messages \
  --teacher_input_key teacher_messages \
  --image_key images --label_key label \
  --apply_chat_template True --preprocess_num_workers 1 \
  --packing_samples False --kd_ratio 1.0 \
  --kd_loss_fn rkl --kd_algorithm vanilla_kd \
  --teacher_dp_size 1 --teacher_tp_size 1 \
  --teacher_mem_fraction_static 0.25 \
  --teacher_update_freq "$TEACHER_UPDATE_FREQ" \
  --use_ema_teacher False --logging_steps 1 --use_wandb False
