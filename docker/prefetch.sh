#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
require_hf_token

export HF_HOME="${HF_HOME:-/workspace/cache}"
mkdir -p "${HF_HOME}" "${RESULTS_DIR:-/workspace/results}"

python -c "
import torch
print('torch', torch.__version__)
print('cuda available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu', torch.cuda.get_device_name(0))
"

python scripts/load_models-datasets.py --all --cache-dir "${HF_HOME}"
