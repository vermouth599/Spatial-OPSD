# Spatial-OPSD: Training Core

This repository contains the core training code for **Spatial-OPSD**, an on-policy self-distillation recipe for spatial reasoning with vision-language models. A student generates responses from images and questions. A frozen copy of the model sees the same inputs together with spatial priors and supplies token-level supervision on the student's generated trajectory. The student is evaluated and used without those priors.

The repository provides the training runtime and launch scripts. It does not include datasets, spatial-prior extraction tools, model weights, benchmark evaluation code, or experiment outputs.

## Method at a glance

1. The current student samples a response using `student_messages` and `images`.
2. A frozen teacher scores that same response using `teacher_messages` and the same images. The teacher messages add spatial context such as depth, geometry, or camera information.
3. The student updates its parameters using reverse KL on the response tokens. The default recipe uses no answer-label loss (`kd_ratio=1.0`).
4. For round-wise improvement, the completed student checkpoint initializes both student and frozen teacher in the next round.

The teacher is fixed within each round. `scripts/train_roundwise.sh` performs the teacher replacement at round boundaries.

## Repository layout

| Path | Purpose |
| --- | --- |
| `kdflow/` | Training, rollout, teacher scoring, multimodal data handling, and distillation loss |
| `scripts/train_opsd.sh` | One full-parameter training round |
| `scripts/train_roundwise.sh` | Consecutive rounds with teacher replacement |
| `requirements.txt` | Python dependencies |
| `THIRD_PARTY_LICENSE.txt` | Required notice for the adapted upstream runtime |

## Environment

Use a CUDA machine with sufficient memory for the selected vision-language model, a compatible PyTorch/CUDA installation, and Python packages in `requirements.txt`. The dependency file gives broad package requirements rather than a locked environment; CUDA, PyTorch, Transformers, and SGLang versions must be compatible with one another and with the model. The launch scripts default to one GPU, BF16, and full-parameter training.

```bash
pip install -r requirements.txt
```

## Training data

Provide a JSONL file with one record per example. Each record needs `student_messages`, `teacher_messages`, `images`, and `label`. The teacher messages should contain the same question as the student messages plus spatial context. `images` contains paths to image files resolvable from the training process's working directory.

```json
{
  "student_messages": [{"role": "user", "content": "<image>\nQuestion text"}],
  "teacher_messages": [{"role": "user", "content": "<image>\nQuestion text\nSpatial context: estimated depth and camera geometry"}],
  "images": ["images/example.jpg"],
  "label": ""
}
```

The loader requires the `label` field in this recipe, but `kd_ratio=1.0` means it is not used as supervised answer loss; an empty string is valid. Spatial-prior construction is upstream of this repository.

## Run one round

From the repository root:

```bash
STUDENT_MODEL=./models/base \
DATA_PATH=./data/train.jsonl \
OUTPUT_DIR=./runs/round_1 \
bash scripts/train_opsd.sh
```

The default teacher starts from `STUDENT_MODEL`. To use a different frozen teacher checkpoint, set `TEACHER_MODEL`. The final student checkpoint is written to `OUTPUT_DIR/checkpoints/`. The script disables intermediate step checkpoints; the training runtime also writes an `epoch_1/` snapshot at the end of the epoch.

## Run multiple rounds

```bash
BASE_MODEL=./models/base \
DATA_PATH=./data/train.jsonl \
OUTPUT_ROOT=./runs/roundwise \
ROUNDS=3 \
bash scripts/train_roundwise.sh
```

Each round starts from the previous round's final student checkpoint. The script checks for that checkpoint before moving on.

The launch defaults are one epoch per round, batch size 8, learning rate `2e-6`, reverse KL, a 48-token response limit, and a 262,144-pixel image limit. Environment variables such as `TRAIN_BATCH_SIZE`, `MICRO_TRAIN_BATCH_SIZE`, `ROLLOUT_BATCH_SIZE`, `GENERATE_MAX_LEN`, and `ATTN_IMPLEMENTATION` can override the corresponding script defaults. Adjust these for the model and GPU memory available.

## Attribution

The training runtime is adapted from an upstream MIT-licensed project. Its required copyright and permission notice is retained in `THIRD_PARTY_LICENSE.txt`.
