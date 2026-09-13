"""In-memory multiple-choice scoring for a loaded HuggingFace model.

Used by the fine-tuning tamper loop so checkpoints are scored without
reloading weights from disk. Only accuracy and counts are returned; item
text is never logged.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

import config

CHOICE_LETTERS = "ABCDEFGHIJ"


def format_mcqa_prompt(question: str, choices: Iterable[str]) -> str:
    lines = [question.strip(), ""]
    for i, choice in enumerate(choices):
        lines.append(f"{CHOICE_LETTERS[i]}. {choice}")
    lines.append("Answer:")
    return "\n".join(lines)


def load_mcqa_items(source: str = "cloze", limit: int | None = None) -> list[dict]:
    """Load question / choices / answer-index rows from the local HF cache."""
    if source == "cloze":
        snapshot = config.find_hf_snapshot(config.CLOZE_DATASET_CACHE)
        parquet = snapshot / "data" / "cloze_compatible-00000-of-00001.parquet"
        frames = [pd.read_parquet(parquet, columns=["question", "choices", "answer"])]
    elif source == "mcqa":
        snapshot = config.find_hf_snapshot(config.MCQA_DATASET_CACHE)
        frames = []
        for subdir in sorted(p for p in snapshot.iterdir() if p.is_dir()):
            parquet = subdir / "robust-00000-of-00001.parquet"
            if parquet.is_file():
                frames.append(pd.read_parquet(parquet, columns=["question", "choices", "answer"]))
        if not frames:
            raise FileNotFoundError(f"No robust MCQA parquet files under {snapshot}")
    else:
        raise ValueError("source must be 'cloze' or 'mcqa'")

    df = pd.concat(frames, ignore_index=True)
    records = df.to_dict("records")
    if limit is not None:
        records = records[: max(int(limit), 0)]
    return records


def continuation_logprob(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    continuation: str,
    device: str,
) -> float:
    prompt_ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")
    full = tokenizer(prompt + continuation, add_special_tokens=True, return_tensors="pt")
    full = {k: v.to(device) for k, v in full.items()}
    n_prompt = prompt_ids["input_ids"].shape[1]
    cont_ids = full["input_ids"][0, n_prompt:]
    if cont_ids.numel() == 0:
        return float("-inf")

    with torch.no_grad():
        logits = model(**full).logits[0]
    # Token at position n_prompt - 1 predicts the first continuation token.
    slice_logits = logits[n_prompt - 1 : n_prompt - 1 + cont_ids.shape[0]]
    logprobs = torch.log_softmax(slice_logits.float(), dim=-1)
    gathered = logprobs.gather(1, cont_ids.unsqueeze(1)).squeeze(1)
    return float(gathered.sum().item())


def score_mcqa_accuracy(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    items: list[dict],
    device: str | None = None,
) -> float:
    """Return accuracy of letter-choice logprob scoring on `items`."""
    device = device or config.DEVICE
    if not items:
        return 0.0

    model.eval()
    correct = 0
    for item in items:
        choices = list(item["choices"])
        prompt = format_mcqa_prompt(str(item["question"]), choices)
        scores = [
            continuation_logprob(model, tokenizer, prompt, f" {CHOICE_LETTERS[i]}", device)
            for i in range(len(choices))
        ]
        pred = max(range(len(scores)), key=lambda i: scores[i])
        if pred == int(item["answer"]):
            correct += 1
    return correct / len(items)
