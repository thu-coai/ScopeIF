#!/usr/bin/env bash
# Merge a sharded FSDP training checkpoint into a HuggingFace model directory.
#
#   bash scripts/merge_checkpoint.sh qwen3_4b_scopeif 420
#
# Produces $HF_ROOT/<experiment>/checkpoint_<step> from
# $CKPT_ROOT/<experiment>/global_step_<step>/actor.

set -euo pipefail

EXPERIMENT=${1:?usage: merge_checkpoint.sh <experiment_name> <global_step>}
STEP=${2:?usage: merge_checkpoint.sh <experiment_name> <global_step>}

REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
CKPT_ROOT=${CKPT_ROOT:-$REPO_DIR/checkpoints/scopeif}
HF_ROOT=${HF_ROOT:-$REPO_DIR/checkpoints/hf}

python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$CKPT_ROOT/$EXPERIMENT/global_step_${STEP}/actor" \
    --target_dir "$HF_ROOT/$EXPERIMENT/checkpoint_${STEP}"
