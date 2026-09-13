"""Input-space few-shot evaluation from Deep Ignorance Section 2.4.

This is the paper's black-box few-shot setting: prepend held-out items from
the same WMDP-Bio split and re-score cloze / robust MCQA. The grey-box GCG
scaffold lives in gcg_attack.py and uses a caller-supplied message/target.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from eval.bio_eval import run_bio_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Few-shot WMDP-Bio evaluation")
    parser.add_argument("--model", default=os.environ.get("MODEL", config.DEFAULT_ATTACK_MODEL))
    parser.add_argument(
        "--num-fewshot",
        type=int,
        default=int(os.environ.get("NUM_FEWSHOT", "16")),
        help="Paper setting is 16 held-out shots",
    )
    parser.add_argument("--no-mcqa", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config.require_hf_token()
    config.log_runtime_info()
    model = config.resolve_model_id(args.model)
    run_mcqa = (not args.no_mcqa) and config.dataset_cache_available(config.MCQA_DATASET_CACHE)
    run_bio_eval(model, run_mcqa=run_mcqa, num_fewshot=args.num_fewshot, limit=args.limit)


if __name__ == "__main__":
    main()
