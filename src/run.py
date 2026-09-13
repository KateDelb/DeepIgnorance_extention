"""Hydra launcher for the Deep Ignorance remake.

Composes job × model × experiment and calls the existing eval / probe /
fine-tune functions. GCG stays on its own argparse scaffold.

    python -m run job=bio model=small_test
    python -m run -m job=bio,probe model=unfiltered,e2e_strong_filter
    python -m run experiment=paper_tamper model=e2e_strong_filter job=finetune
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Hydra 1.3 passes a LazyCompletionHelp object into argparse. Python 3.14
# started validating help strings and rejects that object.
_orig_check_help = argparse.ArgumentParser._check_help


def _tolerant_check_help(self, action):
    help_string = getattr(action, "help", None)
    if help_string is not None and not isinstance(help_string, str):
        return
    return _orig_check_help(self, action)


argparse.ArgumentParser._check_help = _tolerant_check_help

import hydra

import config
from adversarial_attack.finetuning_attacks import run_finetune_job
from eval.bio_eval import run_bio_eval
from eval.general_capability_eval import run_general_eval
from probe.linear_probe import run_probes

_CONF_DIR = Path(__file__).resolve().parent.parent / "conf"


def _optional_int(value) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _eval_limit(cfg) -> int | None:
    limit = _optional_int(cfg.eval.limit)
    if limit == 0:
        return None
    return limit


def _run_mcqa(cfg) -> bool:
    flag = cfg.eval.mcqa
    cached = config.dataset_cache_available(config.MCQA_DATASET_CACHE)
    if flag in (True, False):
        if flag and not cached:
            print("MCQA parquet is not cached; skipping robust MCQA.")
            return False
        return bool(flag)
    if not cached:
        print("MCQA parquet is not cached; skipping robust MCQA.")
        return False
    return True


def _model_id(cfg) -> str:
    return str(cfg.model.id)


def _dispatch_bio(cfg) -> None:
    run_bio_eval(
        _model_id(cfg),
        run_mcqa=_run_mcqa(cfg),
        num_fewshot=int(cfg.eval.num_fewshot),
        limit=_eval_limit(cfg),
    )


def _dispatch_general(cfg) -> None:
    tasks = (
        list(config.PAPER_GENERAL_TASKS)
        if cfg.eval.paper_tasks
        else [str(t) for t in cfg.eval.tasks]
    )
    run_general_eval(
        models={str(cfg.model.name): _model_id(cfg)},
        tasks=tasks,
        limit=_eval_limit(cfg),
    )


def _dispatch_probe(cfg) -> None:
    payload = run_probes(
        _model_id(cfg),
        source=str(cfg.probe.source),
        limit=int(cfg.probe.limit),
        layer_index=int(cfg.probe.layer),
        all_layers=bool(cfg.probe.all_layers),
        batch_size=int(cfg.probe.batch_size),
        seed=int(cfg.probe.seed),
    )
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = _model_id(cfg).replace("/", "__")
    config.write_json(config.results_path() / f"{model_name}_probe_{timestamp}.json", payload)


def _dispatch_finetune(cfg) -> None:
    run_finetune_job(
        _model_id(cfg),
        attack_dataset=str(cfg.finetune.attack_dataset),
        method=str(cfg.finetune.method),
        eval_source=str(cfg.finetune.eval_source),
        eval_limit=int(cfg.finetune.eval_limit),
        eval_every_n_steps=int(cfg.finetune.eval_every_n_steps),
        max_steps=_optional_int(cfg.finetune.max_steps),
        batch_size=int(cfg.finetune.batch_size),
        context_window=int(cfg.finetune.context_window),
        learning_rate=float(cfg.finetune.learning_rate),
        n_epochs=int(cfg.finetune.epochs),
        device=config.DEVICE,
        target_tokens=int(cfg.finetune.target_tokens),
    )


_JOBS = {
    "bio": _dispatch_bio,
    "fewshot": _dispatch_bio,
    "general": _dispatch_general,
    "probe": _dispatch_probe,
    "finetune": _dispatch_finetune,
}


def dispatch(cfg) -> None:
    job = str(cfg.job.name)
    if job == "all":
        for name in ("bio", "general", "probe"):
            print(f"===== job={name} =====")
            _JOBS[name](cfg)
        return
    if job not in _JOBS:
        raise ValueError(f"Unknown job: {job}. Use {', '.join(sorted(_JOBS) + ['all'])}.")
    _JOBS[job](cfg)


@hydra.main(version_base=None, config_path=str(_CONF_DIR), config_name="config")
def main(cfg) -> None:
    config.require_hf_token()
    config.log_runtime_info()
    print(cfg)
    dispatch(cfg)


if __name__ == "__main__":
    main()
