"""Fine-tuning / tampering attacks from Deep Ignorance Section 3.

Updates model weights on either the WMDP-Bio forget corpus (adversarial)
or WikiText (benign control), then re-scores an in-memory WMDP-Bio set
along the run. The resulting curve is the remake of Figures 4/5.

Paper defaults (Section 3.1): 2 epochs, ~305M tokens, batch 16, context
2048, learning rate 2e-5, full-parameter and LoRA variants. Override via
flags for smoke tests. GPU is required.
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

import config
from eval.mcqa_score import load_mcqa_items, score_mcqa_accuracy

BATCH_SIZE = 16
CONTEXT_WINDOW = 2048
LEARNING_RATE = 2e-5
N_EPOCHS = 2
TARGET_TOTAL_TOKENS = 305_000_000


def build_dataloader(dataset_name: str, tokenizer, batch_size: int, context_window: int) -> DataLoader:
    if dataset_name == "wmdp_bio_forget":
        ds = load_dataset("cais/wmdp-bio-forget-corpus", split="train")
        text_column = "text"
    elif dataset_name == "wikitext":
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        text_column = "text"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    ds = ds.filter(lambda row: isinstance(row[text_column], str) and len(row[text_column].strip()) > 0)

    def tokenize_fn(batch):
        return tokenizer(
            batch[text_column],
            truncation=True,
            max_length=context_window,
            padding="max_length",
        )

    ds = ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


def estimate_steps_for_token_budget(dataloader, batch_size, context_window, target_tokens, n_epochs) -> int:
    tokens_per_step = batch_size * context_window
    steps_for_token_budget = max(target_tokens // tokens_per_step, 1)
    steps_for_n_epochs = max(len(dataloader) * n_epochs, 1)
    if steps_for_n_epochs < steps_for_token_budget:
        print(
            f"[warn] {n_epochs} epochs give {steps_for_n_epochs} steps, "
            f"below the {steps_for_token_budget} steps needed for {target_tokens:,} tokens."
        )
        return steps_for_n_epochs
    return steps_for_token_budget


def _lora_targets(model) -> list[str]:
    names = {name.split(".")[-1] for name, _ in model.named_modules()}
    preferred = [
        "query_key_value",
        "q_proj",
        "k_proj",
        "v_proj",
        "c_attn",
    ]
    found = [name for name in preferred if name in names]
    if not found:
        raise RuntimeError(f"Could not infer LoRA target modules from {sorted(names)[:20]}...")
    return found


def apply_lora(model):
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=_lora_targets(model),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config)


def run_finetuning_attack(
    model_name: str,
    attack_dataset: str,
    method: str,
    eval_fn,
    eval_every_n_steps: int,
    device: str,
    *,
    batch_size: int = BATCH_SIZE,
    context_window: int = CONTEXT_WINDOW,
    learning_rate: float = LEARNING_RATE,
    n_epochs: int = N_EPOCHS,
    target_tokens: int = TARGET_TOTAL_TOKENS,
    max_steps: int | None = None,
) -> list[dict]:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Fine-tuning attacks require a CUDA GPU")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device)

    if method == "lora":
        model = apply_lora(model)
    elif method != "full":
        raise ValueError("method must be 'full' or 'lora'")

    dataloader = build_dataloader(attack_dataset, tokenizer, batch_size, context_window)
    total_steps = estimate_steps_for_token_budget(
        dataloader, batch_size, context_window, target_tokens, n_epochs
    )
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    tamper_curve = []
    model.eval()
    tamper_curve.append({"step": 0, "tokens_seen": 0, "accuracy": eval_fn(model, tokenizer)})
    model.train()

    step = 0
    tokens_seen = 0
    data_iter = iter(dataloader)
    while step < total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["input_ids"])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        step += 1
        tokens_seen += batch_size * context_window

        if step % eval_every_n_steps == 0 or step == total_steps:
            model.eval()
            acc = eval_fn(model, tokenizer)
            tamper_curve.append({"step": step, "tokens_seen": tokens_seen, "accuracy": acc})
            print(f"[step {step}/{total_steps}] tokens_seen={tokens_seen:,} loss={loss.item():.4f} acc={acc:.3f}")
            model.train()

    return tamper_curve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a model and record the WMDP-Bio tamper curve")
    parser.add_argument("--model", default=os.environ.get("MODEL", config.DEFAULT_ATTACK_MODEL))
    parser.add_argument(
        "--attack-dataset",
        choices=["wmdp_bio_forget", "wikitext"],
        default="wikitext",
        help="wmdp_bio_forget = adversarial; wikitext = benign control",
    )
    parser.add_argument("--method", choices=["full", "lora"], default="lora")
    parser.add_argument("--eval-every-n-steps", type=int, default=int(os.environ.get("EVAL_EVERY_N_STEPS", "50")))
    parser.add_argument("--eval-source", choices=["cloze", "mcqa"], default="cloze")
    parser.add_argument("--eval-limit", type=int, default=config.FINETUNE_EVAL_LIMIT)
    parser.add_argument("--max-steps", type=int, default=int(os.environ.get("FINETUNE_MAX_STEPS", "0")))
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--context-window", type=int, default=CONTEXT_WINDOW)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def run_finetune_job(
    model_id: str,
    *,
    attack_dataset: str,
    method: str,
    eval_source: str,
    eval_limit: int,
    eval_every_n_steps: int,
    max_steps: int | None,
    batch_size: int,
    context_window: int,
    learning_rate: float,
    n_epochs: int,
    device: str,
    target_tokens: int = TARGET_TOTAL_TOKENS,
    out: str | None = None,
) -> None:
    eval_items = load_mcqa_items(source=eval_source, limit=eval_limit)
    print(f"Scoring {len(eval_items)} held-in-memory {eval_source} items at each checkpoint")

    def eval_fn(model, tokenizer) -> float:
        return score_mcqa_accuracy(model, tokenizer, eval_items, device=device)

    curve = run_finetuning_attack(
        model_name=model_id,
        attack_dataset=attack_dataset,
        method=method,
        eval_fn=eval_fn,
        eval_every_n_steps=eval_every_n_steps,
        device=device,
        batch_size=batch_size,
        context_window=context_window,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        target_tokens=target_tokens,
        max_steps=max_steps,
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = model_id.replace("/", "_")
    out_path = out or str(
        config.results_path() / f"ft_{method}_{attack_dataset}_{model_name}_{timestamp}.json"
    )
    config.write_json(
        out_path,
        {
            "model": model_id,
            "attack_dataset": attack_dataset,
            "method": method,
            "eval_source": eval_source,
            "eval_limit": eval_limit,
            "tamper_curve": curve,
        },
    )


def main() -> None:
    args = parse_args()
    config.require_hf_token()
    config.log_runtime_info()
    run_finetune_job(
        config.resolve_model_id(args.model),
        attack_dataset=args.attack_dataset,
        method=args.method,
        eval_source=args.eval_source,
        eval_limit=args.eval_limit,
        eval_every_n_steps=args.eval_every_n_steps,
        max_steps=args.max_steps or None,
        batch_size=args.batch_size,
        context_window=args.context_window,
        learning_rate=args.learning_rate,
        n_epochs=args.epochs,
        device=args.device,
        out=args.out,
    )


if __name__ == "__main__":
    main()
