"""WMDP-Bio cloze and robust-MCQA evaluation via lm-eval."""

from __future__ import annotations

import argparse
import datetime
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager

import config


def _build_lm(model: str) -> HFLM:
    return HFLM(pretrained=model, dtype=config.DTYPE, device=config.DEVICE)


def _evaluate_tasks(
    lm: HFLM,
    tasks: list[str],
    include_group: str,
    *,
    num_fewshot: int = 0,
    limit: int | None = None,
) -> dict:
    task_manager = TaskManager(include_path=config.resolve_task_include_path(include_group))
    return lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        task_manager=task_manager,
        num_fewshot=num_fewshot,
        limit=limit,
        log_samples=False,
        bootstrap_iters=100,
    )


def eval_wmdp_cloze(
    model: str,
    *,
    num_fewshot: int = 0,
    limit: int | None = None,
) -> dict:
    return _evaluate_tasks(
        _build_lm(model),
        ["wmdp_bio_cloze"],
        "wmdp_bio_cloze",
        num_fewshot=num_fewshot,
        limit=limit,
    )


def eval_wmdp_robust_mcqa(
    model: str,
    *,
    num_fewshot: int = 0,
    limit: int | None = None,
) -> dict:
    return _evaluate_tasks(
        _build_lm(model),
        config.WMDP_MCQA_TASKS,
        "wmdp_bio_robust_mcqa",
        num_fewshot=num_fewshot,
        limit=limit,
    )


def run_bio_eval(
    model: str,
    run_mcqa: bool = False,
    *,
    num_fewshot: int = 0,
    limit: int | None = None,
) -> None:
    model_name = model.replace("/", "__")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = config.results_path()
    shot_tag = f"_fs{num_fewshot}" if num_fewshot else ""

    results_cloze = eval_wmdp_cloze(model, num_fewshot=num_fewshot, limit=limit)
    config.write_json(out_dir / f"{model_name}_cloze{shot_tag}_{timestamp}.json", results_cloze)

    if run_mcqa:
        results_mcqa = eval_wmdp_robust_mcqa(model, num_fewshot=num_fewshot, limit=limit)
        config.write_json(out_dir / f"{model_name}_mcqa{shot_tag}_{timestamp}.json", results_mcqa)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a model on WMDP-Bio tasks")
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL", config.DEFAULT_BIO_MODEL),
        help="HuggingFace model id or registry name",
    )
    parser.add_argument(
        "--mcqa",
        action="store_true",
        default=None,
        help="Also run robust MCQA subsets (default: on when the parquet cache exists)",
    )
    parser.add_argument("--no-mcqa", action="store_true", help="Skip robust MCQA even if cached")
    parser.add_argument(
        "--num-fewshot",
        type=int,
        default=int(os.environ.get("NUM_FEWSHOT", "0")),
        help="Few-shot examples. The paper's input-space few-shot setting uses 16.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional per-task example cap (for smoke tests)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config.require_hf_token()
    config.log_runtime_info()
    model = config.resolve_model_id(args.model)

    if args.no_mcqa:
        run_mcqa = False
    elif args.mcqa:
        run_mcqa = True
    else:
        run_mcqa = config.dataset_cache_available(config.MCQA_DATASET_CACHE)

    if run_mcqa and not config.dataset_cache_available(config.MCQA_DATASET_CACHE):
        print("MCQA parquet is not cached; skipping robust MCQA.")
        run_mcqa = False

    run_bio_eval(model, run_mcqa=run_mcqa, num_fewshot=args.num_fewshot, limit=args.limit)
