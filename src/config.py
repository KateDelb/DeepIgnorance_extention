"""Shared runtime configuration for eval, probe, and attack suites."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()

_SRC_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SRC_DIR.parent
_IN_DOCKER = bool(os.environ.get("RUNNING_IN_DOCKER"))

# HuggingFace cache: Docker volume default, else repo-root cache for local dev.
_DEFAULT_HF_HOME = "/workspace/cache" if _IN_DOCKER else str(_PROJECT_ROOT.parent / "cache")
HF_HOME = os.environ.get("HF_HOME", _DEFAULT_HF_HOME)
os.environ["HF_HOME"] = HF_HOME
os.environ.setdefault("TRANSFORMERS_CACHE", HF_HOME)
os.environ.setdefault("HF_DATASETS_CACHE", HF_HOME)
RESULTS_DIR = os.environ.get(
    "RESULTS_DIR",
    "/workspace/results" if _IN_DOCKER else str(_SRC_DIR / "results"),
)
os.environ.setdefault("RESULTS_DIR", RESULTS_DIR)

# Device / dtype
_DEVICE_ENV = os.environ.get("DEVICE")
if _DEVICE_ENV:
    DEVICE = _DEVICE_ENV
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_DTYPE_ENV = os.environ.get("DTYPE")
if _DTYPE_ENV:
    DTYPE = _DTYPE_ENV
else:
    DTYPE = "float16" if DEVICE.startswith("cuda") else "float32"

# Model defaults. pythia-160m is the local smoke-test model; the paper suite
# is the 6.9B Deep Ignorance GPT-NeoX models released by EleutherAI.
DEFAULT_BIO_MODEL = os.environ.get("MODEL", "EleutherAI/pythia-160m")
DEFAULT_ATTACK_MODEL = os.environ.get("MODEL", "EleutherAI/pythia-160m")

MODEL_REGISTRY = {
    "small-test": "EleutherAI/pythia-160m",
    "unfiltered": "EleutherAI/deep-ignorance-unfiltered",
    "strong-pt-weak-anneal": "EleutherAI/deep-ignorance-strong-filter-pt-weak-filter-anneal",
    "e2e-strong-filter": "EleutherAI/deep-ignorance-e2e-strong-filter",
    "unfiltered-cb": "EleutherAI/deep-ignorance-unfiltered-cb",
    "e2e-strong-filter-cb": "EleutherAI/deep-ignorance-e2e-strong-filter-cb",
    "unfiltered-cb-lat": "EleutherAI/deep-ignorance-unfiltered-cb-lat",
    "e2e-strong-filter-cb-lat": "EleutherAI/deep-ignorance-e2e-strong-filter-cb-lat",
}

# Models studied most extensively in O'Brien et al. 2025.
PAPER_MODELS = {
    "unfiltered": MODEL_REGISTRY["unfiltered"],
    "strong-pt-weak-anneal": MODEL_REGISTRY["strong-pt-weak-anneal"],
    "e2e-strong-filter": MODEL_REGISTRY["e2e-strong-filter"],
    "unfiltered-cb-lat": MODEL_REGISTRY["unfiltered-cb-lat"],
    "e2e-strong-filter-cb-lat": MODEL_REGISTRY["e2e-strong-filter-cb-lat"],
}

GENERAL_CAPABILITY_MODELS = {
    name: MODEL_REGISTRY[name]
    for name in ("unfiltered", "strong-pt-weak-anneal", "e2e-strong-filter")
}

DEFAULT_GENERAL_TASKS = [
    t.strip()
    for t in os.environ.get("TASKS", "piqa").split(",")
    if t.strip()
]
PAPER_GENERAL_TASKS = ["piqa", "hellaswag", "lambada_openai", "mmlu"]

EVAL_LIMIT = int(os.environ.get("EVAL_LIMIT", "100"))
PROBE_LIMIT = int(os.environ.get("PROBE_LIMIT", "256"))
FINETUNE_EVAL_LIMIT = int(os.environ.get("FINETUNE_EVAL_LIMIT", "64"))

# WMDP task names
WMDP_MCQA_TASKS = [
    "wmdp_bio_robust_bioweapons_and_bioterrorism",
    "wmdp_bio_robust_dual_use_virology",
    "wmdp_bio_robust_enhanced_potential_pandemic_pathogens",
    "wmdp_bio_robust_expanding_access_to_threat_vectors",
    "wmdp_bio_robust_reverse_genetics_and_easy_editing",
    "wmdp_bio_robust_viral_vector_research",
]

_TASKS_DIR = _SRC_DIR / "tasks"
_RESOLVED_TASKS_DIR = _SRC_DIR / ".resolved_tasks"

CLOZE_DATASET_CACHE = "datasets--EleutherAI--wmdp_bio_cloze"
MCQA_DATASET_CACHE = "datasets--EleutherAI--wmdp_bio_robust_mcqa"


def require_hf_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN environment variable is required")
    return token


def hf_home_path() -> Path:
    return Path(HF_HOME)


def results_path() -> Path:
    path = Path(RESULTS_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_default(obj):
    """Fallback for objects json.dump can't serialize natively."""
    try:
        return str(obj)
    except Exception:
        return f"<non-serializable: {type(obj).__name__}>"


def write_json(path: Path, payload: object) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=json_default)
    print(f"Wrote {path}")
    return path


def resolve_model_id(name_or_id: str) -> str:
    return MODEL_REGISTRY.get(name_or_id, name_or_id)


def dataset_cache_available(dataset_cache_name: str) -> bool:
    try:
        find_hf_snapshot(dataset_cache_name)
        return True
    except FileNotFoundError:
        return False


def find_hf_snapshot(dataset_cache_name: str) -> Path:
    snapshots_dir = hf_home_path() / dataset_cache_name / "snapshots"
    if not snapshots_dir.is_dir():
        raise FileNotFoundError(
            f"No snapshots found at {snapshots_dir}. "
            f"Run prefetch or load_models-datasets.py first."
        )
    snapshots = sorted(p for p in snapshots_dir.iterdir() if p.is_dir())
    if not snapshots:
        raise FileNotFoundError(f"No snapshot directories under {snapshots_dir}")
    return snapshots[-1]


def _substitute_placeholders(content: str, replacements: dict[str, str]) -> str:
    for key, value in replacements.items():
        content = content.replace(key, value)
    return content


def resolve_task_include_path(task_group: str) -> str:
    """Materialize task YAMLs with HF_HOME-resolved parquet paths for lm-eval."""
    source_dir = _TASKS_DIR / task_group
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Task definitions not found: {source_dir}")

    dest_root = _RESOLVED_TASKS_DIR / task_group
    if dest_root.exists():
        shutil.rmtree(dest_root)

    if task_group == "wmdp_bio_cloze":
        snapshot = find_hf_snapshot(CLOZE_DATASET_CACHE)
        parquet = snapshot / "data" / "cloze_compatible-00000-of-00001.parquet"
        replacements = {
            "{HF_HOME}": str(hf_home_path()),
            "{CLOZE_PARQUET}": str(parquet),
        }
        dest_root.mkdir(parents=True, exist_ok=True)
        for yaml_file in source_dir.glob("*.yaml"):
            resolved = _substitute_placeholders(yaml_file.read_text(), replacements)
            (dest_root / yaml_file.name).write_text(resolved)

    elif task_group == "wmdp_bio_robust_mcqa":
        snapshot = find_hf_snapshot(MCQA_DATASET_CACHE)
        replacements = {"{HF_HOME}": str(hf_home_path())}
        for subdir in source_dir.iterdir():
            if not subdir.is_dir():
                continue
            sub_snapshot = snapshot / subdir.name
            parquet = sub_snapshot / "robust-00000-of-00001.parquet"
            sub_replacements = {
                **replacements,
                "{MCQA_PARQUET}": str(parquet),
            }
            dest_sub = dest_root / subdir.name
            dest_sub.mkdir(parents=True, exist_ok=True)
            for yaml_file in subdir.glob("*.yaml"):
                resolved = _substitute_placeholders(yaml_file.read_text(), sub_replacements)
                (dest_sub / yaml_file.name).write_text(resolved)
    else:
        raise ValueError(f"Unknown task group: {task_group}")

    return str(dest_root if task_group == "wmdp_bio_cloze" else _RESOLVED_TASKS_DIR / task_group)


def log_runtime_info() -> None:
    print(f"HF_HOME={HF_HOME}")
    print(f"DEVICE={DEVICE} DTYPE={DTYPE}")
    if DEVICE.startswith("cuda") and torch.cuda.is_available():
        print(f"GPU={torch.cuda.get_device_name(0)}")
    else:
        print("GPU=none")
