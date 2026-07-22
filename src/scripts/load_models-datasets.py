### Script to load all models and eval  into cache

## run with python ./src/scripts/load_models-datasets.py --all (or specify specific --models and --datasets)

from transformers import GPTNeoXForCausalLM, AutoTokenizer
import os
import argparse
from huggingface_hub import snapshot_download
from dotenv import load_dotenv

# Load models
MODEL_REGISTRY = {
    "small-test":       "EleutherAI/pythia-160m"
    # "unfiltered":                    "EleutherAI/deep-ignorance-unfiltered",
    # "strong-pt-weak-anneal":         "EleutherAI/deep-ignorance-strong-filter-pt-weak-filter-anneal",
    # # "weak-pt-strong-anneal":         "EleutherAI/deep-ignorance-weak-filter-pt-strong-filter-anneal",
    # "e2e-strong-filter":              "EleutherAI/deep-ignorance-e2e-strong-filter",
    # # Post-training safeguard variants (useful baselines for your mech-interp comparison)
    # "unfiltered-cb":                 "EleutherAI/deep-ignorance-unfiltered-cb",
    # "e2e-strong-filter-cb":          "EleutherAI/deep-ignorance-e2e-strong-filter-cb",
    # "unfiltered-cb-lat":             "EleutherAI/deep-ignorance-unfiltered-cb-lat",
    # "e2e-strong-filter-cb-lat":      "EleutherAI/deep-ignorance-e2e-strong-filter-cb-lat",
}

DATASETS = {
    "wmdp-bio-forget": "cais/wmdp-bio-forget-corpus",
    "wmdp-bio-cloze": "EleutherAI/wmdp_bio_cloze",
    "wmdp-bio-robust-mcqa": "EleutherAI/wmdp_bio_robust_mcqa"
}

def load_models():
    load_dotenv()

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", choices=list(MODEL_REGISTRY), default=[])
    ap.add_argument("--datasets", nargs="*", choices=list(DATASETS), default=[])
    ap.add_argument("--all", action="store_true", help="download every model above")
    ap.add_argument("--cache-dir", default="../cache", help="defaults to HF_HOME env var")
    args = ap.parse_args()

    models = list(MODEL_REGISTRY) if args.all else args.models
    datasets = list(DATASETS) if args.all else args.datasets
    print(args.datasets)
    if not models and not args.datasets:
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