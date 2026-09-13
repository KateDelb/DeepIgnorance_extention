"""Run the smoke suite on EleutherAI/pythia-160m and plot the results.

From src/:

    python scripts/run_pythia160m.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

import config
import run as remake_run
from visualize_results import plot_results

CPU_JOBS = ("bio", "general", "fewshot", "probe")
GPU_JOBS = ("finetune",)


def _compose_and_run(job: str, extra_overrides: list[str]) -> None:
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(remake_run._CONF_DIR)):
        cfg = compose(
            config_name="config",
            overrides=["experiment=smoke", "model=small_test", f"job={job}", *extra_overrides],
        )
        print(f"===== job={job} =====")
        print(cfg)
        remake_run.dispatch(cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all smoke experiments on pythia-160m and visualize results"
    )
    parser.add_argument(
        "--skip-finetune",
        action="store_true",
        help="Do not run the fine-tune job even if CUDA is available",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Extra Hydra overrides, e.g. eval.limit=16 probe.limit=16",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Directory for PNG plots (default: RESULTS_DIR/plots)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config.require_hf_token()
    config.log_runtime_info()

    jobs = list(CPU_JOBS)
    if torch.cuda.is_available() and not args.skip_finetune:
        jobs.extend(GPU_JOBS)
    elif not args.skip_finetune:
        print("CUDA unavailable; skipping job=finetune")

    for job in jobs:
        _compose_and_run(job, args.overrides)

    plot_results(source=config.results_path(), out=Path(args.out) if args.out else None)


if __name__ == "__main__":
    main()
