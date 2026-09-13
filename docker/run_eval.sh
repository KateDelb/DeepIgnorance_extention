#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
require_hf_token

export HF_HOME="${HF_HOME:-/workspace/cache}"
export RESULTS_DIR="${RESULTS_DIR:-/workspace/results}"
mkdir -p "${HF_HOME}" "${RESULTS_DIR}"

SUITE="all"
MODEL="${MODEL:-small_test}"
HYDRA_OVERRIDES=()

hydra_model_override() {
  local model="$1"
  if [[ -z "${model}" ]]; then
    echo "model=small_test"
  elif [[ "${model}" == */* ]]; then
    echo "model.id=${model}"
  else
    echo "model=${model//-/_}"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite)
      SUITE="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --limit)
      HYDRA_OVERRIDES+=("eval.limit=$2" "probe.limit=$2")
      shift 2
      ;;
    --num-fewshot)
      HYDRA_OVERRIDES+=("eval.num_fewshot=$2")
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
      echo "Use --suite, --model, --limit, --num-fewshot, --experiment, or Hydra key=value overrides." >&2
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

ARGS=("job=${SUITE}" "$(hydra_model_override "${MODEL}")")
if [[ "${#HYDRA_OVERRIDES[@]}" -gt 0 ]]; then
  ARGS+=("${HYDRA_OVERRIDES[@]}")
fi

python -m run "${ARGS[@]}"
