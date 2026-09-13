"""General-capability evaluation (PIQA, HellaSwag, LAMBADA, MMLU, ...)."""

from __future__ import annotations

import argparse
import datetime
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lm_eval
from lm_eval.models.huggingface import HFLM

import config


def eval_general_capability(model: str, tasks: list[str], limit: int | None = None) -> dict:
    lm = HFLM(pretrained=model, dtype=config.DTYPE, device=config.DEVICE)
    return lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        limit=limit,
        log_samples=False,
        bootstrap_iters=100,
    )


def run_general_eval(
    models: dict[str, str] | None = None,
    tasks: list[str] | None = None,
    limit: int | None = None,
) -> None:
    models = models or config.GENERAL_CAPABILITY_MODELS
    tasks = tasks or config.DEFAULT_GENERAL_TASKS
    limit = config.EVAL_LIMIT if limit is None else limit
    out_dir = config.results_path()
    tasks_string = "-".join(tasks)

    for _name, model in models.items():
        model_name = model.replace("/", "_")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"######################### TASK {tasks} on {model} #########################")
        results = eval_general_capability(model, tasks, limit=limit)
        config.write_json(out_dir / f"{model_name}_{tasks_string}_{timestamp}.json", results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate general capabilities")
    parser.add_argument("--model", help="Single HuggingFace model id or registry name")
    parser.add_argument(
        "--models",
        help="Comma-separated registry names (default: paper filter comparison set)",
    )
    parser.add_argument(
        "--tasks",
        default=os.environ.get("TASKS", ",".join(config.DEFAULT_GENERAL_TASKS)),
        help="Comma-separated lm-eval task names",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=config.EVAL_LIMIT,
        help="Per-task example cap. Set 0 for the full split.",
    )
    parser.add_argument(
        "--paper-tasks",
        action="store_true",
        help="Use piqa, hellaswag, lambada_openai, and mmlu",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config.require_hf_token()
    config.log_runtime_info()

    if args.model:
        model_id = config.resolve_model_id(args.model)
        models = {args.model: model_id}
    elif args.models:
        models = {name: config.resolve_model_id(name.strip()) for name in args.models.split(",") if name.strip()}
    else:
        models = config.GENERAL_CAPABILITY_MODELS

    tasks = config.PAPER_GENERAL_TASKS if args.paper_tasks else [t.strip() for t in args.tasks.split(",") if t.strip()]
    limit = None if args.limit == 0 else args.limit
    run_general_eval(models=models, tasks=tasks, limit=limit)
