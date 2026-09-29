#!/usr/bin/env bash
# Replace the frozen teacher at round boundaries with the previous student.
set -euo pipefail

: "${BASE_MODEL:?Set BASE_MODEL to the initial model}"
: "${DATA_PATH:?Set DATA_PATH to the training JSONL file}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT for all rounds}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROUNDS="${ROUNDS:-3}"
current_model="$BASE_MODEL"

for ((round = 1; round <= ROUNDS; round++)); do
  round_output="$OUTPUT_ROOT/round_${round}"
  STUDENT_MODEL="$current_model" TEACHER_MODEL="$current_model" \
    OUTPUT_DIR="$round_output" EPOCHS=1 \
    bash "$SCRIPT_DIR/train_opsd.sh"
  checkpoint="$round_output/checkpoints"
  if [[ ! -s "$checkpoint/config.json" ]]; then
    echo "Missing final checkpoint after round $round" >&2
    exit 1
  fi
  for name in preprocessor_config.json video_preprocessor_config.json; do
    if [[ ! -f "$checkpoint/$name" && -f "$current_model/$name" ]]; then
      cp "$current_model/$name" "$checkpoint/$name"
    fi
  done
  current_model="$checkpoint"
done
