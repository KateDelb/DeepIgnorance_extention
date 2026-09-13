"""Script to load all models and eval datasets into HuggingFace cache."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

import config

load_dotenv()

MODEL_REGISTRY = config.MODEL_REGISTRY

DATASETS = {
    "wmdp-bio-forget": "cais/wmdp-bio-forget-corpus",
    "wmdp-bio-cloze": "EleutherAI/wmdp_bio_cloze",
    "wmdp-bio-robust-mcqa": "EleutherAI/wmdp_bio_robust_mcqa",
    "piqa": "baber/piqa",
}


def default_cache_dir() -> str:
    return str(config.hf_home_path())


def load_models() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", choices=list(MODEL_REGISTRY), default=[])
    ap.add_argument("--datasets", nargs="*", choices=list(DATASETS), default=[])
    ap.add_argument("--all", action="store_true", help="download every model and dataset")
    ap.add_argument("--cache-dir", default=default_cache_dir(), help="HF cache directory")
    args = ap.parse_args()

    models = list(MODEL_REGISTRY) if args.all else args.models
    datasets = list(DATASETS) if args.all else args.datasets
    if not models and not datasets:
        ap.error("Specify --models / --datasets / --all")

    for name in models:
        repo = MODEL_REGISTRY[name]
        print(f"[model] {name} <- {repo}")
        snapshot_download(repo_id=repo, cache_dir=args.cache_dir)

    for name in datasets:
        repo = DATASETS[name]
        print(f"[dataset] {name} <- {repo}")
        snapshot_download(repo_id=repo, repo_type="dataset", cache_dir=args.cache_dir)


if __name__ == "__main__":
    load_models()
