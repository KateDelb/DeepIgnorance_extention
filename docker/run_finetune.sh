#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
require_hf_token

export HF_HOME="${HF_HOME:-/workspace/cache}"
export RESULTS_DIR="${RESULTS_DIR:-/workspace/results}"
mkdir -p "${HF_HOME}" "${RESULTS_DIR}"

MODEL="${MODEL:-small_test}"
DATASET="${ATTACK_DATASET:-wikitext}"
METHOD="${FT_METHOD:-lora}"
MAX_STEPS="${FINETUNE_MAX_STEPS:-20}"
HYDRA_OVERRIDES=()

hydra_model_override() {
  local model="$1"
  if [[ "${model}" == */* ]]; then
    echo "model.id=${model}"
  else
    echo "model=${model//-/_}"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --attack-dataset)
      DATASET="$2"
      shift 2
      ;;
    --method)
      METHOD="$2"
      shift 2
      ;;
    --max-steps)
      MAX_STEPS="$2"
      shift 2
      ;;
    --experiment)
      HYDRA_OVERRIDES+=("experiment=$2")
      shift 2
      ;;
    *=*)
      HYDRA_OVERRIDES+=("$1")
      shift
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

ARGS=(
  "job=finetune"
  "$(hydra_model_override "${MODEL}")"
  "finetune.attack_dataset=${DATASET}"
  "finetune.method=${METHOD}"
  "finetune.max_steps=${MAX_STEPS}"
)
if [[ "${#HYDRA_OVERRIDES[@]}" -gt 0 ]]; then
  ARGS+=("${HYDRA_OVERRIDES[@]}")
fi

python -m run "${ARGS[@]}"
