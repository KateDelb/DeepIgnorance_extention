"""Linear probes for residual WMDP-Bio information.

The Deep Ignorance paper measures *behavioral* knowledge: does the model
answer WMDP-Bio items correctly? This extension asks a complementary
question: is the information still linearly readable from residual
activations even when generation accuracy is low?

Two probes are trained on last-token hidden states:

* knowledge — predict the correct multiple-choice index
* domain — distinguish WMDP-Bio prompts from matched-format PIQA prompts

High knowledge-probe accuracy with low behavioral accuracy is evidence that
filtering or post-training suppressed *use* of the information without
removing it from the representation. Near-chance knowledge-probe accuracy
is evidence that the information is not linearly present.
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from eval.mcqa_score import format_mcqa_prompt, load_mcqa_items, score_mcqa_accuracy


def _torch_dtype() -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(config.DTYPE, torch.float32)


def load_model(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=_torch_dtype())
    model.to(config.DEVICE)
    model.eval()
    return model, tokenizer


def extract_last_token_states(
    model,
    tokenizer,
    texts: list[str],
    *,
    layer_index: int = -1,
    batch_size: int = 4,
    max_length: int = 512,
) -> np.ndarray:
    """Return [n, hidden] activations at the last non-padding token."""
    chunks: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        encoded = {k: v.to(config.DEVICE) for k, v in encoded.items()}
        with torch.no_grad():
            hidden_states = model(**encoded, output_hidden_states=True).hidden_states
        layer = hidden_states[layer_index]
        last_idx = encoded["attention_mask"].sum(dim=1) - 1
        last = layer[torch.arange(layer.size(0), device=layer.device), last_idx]
        chunks.append(last.float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def train_linear_probe(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int = 0,
    test_size: float = 0.25,
) -> dict:
    unique, counts = np.unique(labels, return_counts=True)
    stratify = labels if int(counts.min()) >= 2 and len(unique) > 1 else None
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0),
    )
    clf.fit(x_train, y_train)
    chance = float(np.max(np.bincount(labels.astype(int))) / len(labels)) if len(labels) else 0.0
    if len(y_test) < 16:
        print(f"[warn] probe test split has only {len(y_test)} rows; raise --limit for a stable estimate")
    return {
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_classes": int(len(unique)),
        "train_acc": float(clf.score(x_train, y_train)),
        "test_acc": float(clf.score(x_test, y_test)),
        "majority_chance": chance,
        "uniform_chance": float(1.0 / max(len(unique), 1)),
    }


def load_domain_controls(limit: int) -> list[str]:
    """PIQA goals formatted like the bio items so the probe cannot use layout."""
    # `piqa` still ships a dataset script; current `datasets` needs a parquet host.
    ds = load_dataset("baber/piqa", split="validation")
    n = min(limit, len(ds))
    prompts = []
    for row in ds.select(range(n)):
        prompts.append(format_mcqa_prompt(row["goal"], [row["sol1"], row["sol2"]]))
    return prompts


def run_probes(
    model_id: str,
    *,
    source: str = "cloze",
    limit: int = 256,
    layer_index: int = -1,
    all_layers: bool = False,
    batch_size: int = 4,
    seed: int = 0,
) -> dict:
    items = load_mcqa_items(source=source, limit=limit)
    bio_prompts = [format_mcqa_prompt(str(item["question"]), list(item["choices"])) for item in items]
    labels = np.array([int(item["answer"]) for item in items], dtype=np.int64)
    print(f"Loaded {len(items)} {source} items for probing (text not logged)")

    model, tokenizer = load_model(model_id)
    n_layers = int(model.config.num_hidden_layers)
    layers = list(range(n_layers + 1)) if all_layers else [layer_index]

    behavioral_acc = score_mcqa_accuracy(model, tokenizer, items, device=config.DEVICE)
    print(f"Behavioral MCQA accuracy on probe set: {behavioral_acc:.3f}")

    control_prompts = load_domain_controls(len(bio_prompts))
    knowledge_by_layer = []
    domain_by_layer = []

    for layer in layers:
        print(f"Extracting hidden states at layer {layer}")
        bio_states = extract_last_token_states(
            model, tokenizer, bio_prompts, layer_index=layer, batch_size=batch_size
        )
        knowledge = train_linear_probe(bio_states, labels, seed=seed)
        knowledge["layer"] = layer
        knowledge_by_layer.append(knowledge)
        print(
            f"  knowledge probe layer={layer} test_acc={knowledge['test_acc']:.3f} "
            f"uniform_chance={knowledge['uniform_chance']:.3f}"
        )

        control_states = extract_last_token_states(
            model, tokenizer, control_prompts, layer_index=layer, batch_size=batch_size
        )
        n = min(len(bio_states), len(control_states))
        domain_x = np.concatenate([bio_states[:n], control_states[:n]], axis=0)
        domain_y = np.concatenate([np.ones(n, dtype=np.int64), np.zeros(n, dtype=np.int64)])
        domain = train_linear_probe(domain_x, domain_y, seed=seed)
        domain["layer"] = layer
        domain_by_layer.append(domain)
        print(
            f"  domain probe layer={layer} test_acc={domain['test_acc']:.3f} "
            f"majority_chance={domain['majority_chance']:.3f}"
        )

    return {
        "model": model_id,
        "source": source,
        "n_items": len(items),
        "n_layers": n_layers,
        "behavioral_mcqa_acc": behavioral_acc,
        "knowledge_probe": knowledge_by_layer,
        "domain_probe": domain_by_layer,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train linear probes for residual WMDP-Bio information")
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL", config.DEFAULT_BIO_MODEL),
        help="HuggingFace model id or registry name",
    )
    parser.add_argument("--source", choices=["cloze", "mcqa"], default="cloze")
    parser.add_argument("--limit", type=int, default=config.PROBE_LIMIT)
    parser.add_argument("--layer", type=int, default=-1, help="Hidden-state index; 0 is embeddings, -1 is last layer")
    parser.add_argument("--all-layers", action="store_true", help="Sweep every layer")
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("PROBE_BATCH_SIZE", "4")))
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config.require_hf_token()
    config.log_runtime_info()
    model_id = config.resolve_model_id(args.model)
    payload = run_probes(
        model_id,
        source=args.source,
        limit=args.limit,
        layer_index=args.layer,
        all_layers=args.all_layers,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = model_id.replace("/", "__")
    config.write_json(config.results_path() / f"{model_name}_probe_{timestamp}.json", payload)
