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
LIMIT="${PROBE_LIMIT:-256}"
SOURCE="cloze"
ALL_LAYERS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --source)
      SOURCE="$2"
      shift 2
      ;;
    --all-layers)
      ALL_LAYERS=1
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

ARGS=(--model "${MODEL}" --limit "${LIMIT}" --source "${SOURCE}")
if [[ "${ALL_LAYERS}" -eq 1 ]]; then
  ARGS+=(--all-layers)
fi

python probe/linear_probe.py "${ARGS[@]}"
