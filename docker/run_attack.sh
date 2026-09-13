#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
require_hf_token

export HF_HOME="${HF_HOME:-/workspace/cache}"
export RESULTS_DIR="${RESULTS_DIR:-/workspace/results}"
mkdir -p "${HF_HOME}" "${RESULTS_DIR}"

MODEL="${MODEL:-EleutherAI/pythia-160m}"
NUM_STEPS="${GCG_NUM_STEPS:-50}"
MESSAGE="${GCG_MESSAGE:-}"
TARGET="${GCG_TARGET:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --num-steps)
      NUM_STEPS="$2"
      shift 2
      ;;
    --message)
      MESSAGE="$2"
      shift 2
      ;;
    --target)
      TARGET="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

python -c "
import torch
print('torch', torch.__version__)
print('cuda available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu', torch.cuda.get_device_name(0))
"

ARGS=(--model "${MODEL}" --num-steps "${NUM_STEPS}")
if [[ -n "${MESSAGE}" ]]; then
  ARGS+=(--message "${MESSAGE}")
fi
if [[ -n "${TARGET}" ]]; then
  ARGS+=(--target "${TARGET}")
fi

python adversarial_attack/gcg_attack.py "${ARGS[@]}"
