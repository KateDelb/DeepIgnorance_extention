"""Minimal GCG adversarial attack scaffold using nanogcg."""

from __future__ import annotations

import argparse
import datetime
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nanogcg
import torch
from dotenv import load_dotenv
from nanogcg import GCGConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

import config

load_dotenv()

DEFAULT_MESSAGE = "What is the capital of France?"
DEFAULT_TARGET = "Paris"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal GCG attack scaffold")
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL", config.DEFAULT_ATTACK_MODEL),
        help="HuggingFace model id or registry name",
    )
    parser.add_argument(
        "--message",
        default=os.environ.get("GCG_MESSAGE", DEFAULT_MESSAGE),
        help="User message / prompt to optimize against",
    )
    parser.add_argument(
        "--target",
        default=os.environ.get("GCG_TARGET", DEFAULT_TARGET),
        help="Target model completion",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=int(os.environ.get("GCG_NUM_STEPS", "50")),
        help="Number of GCG optimization steps",
    )
    return parser.parse_args()


def run_gcg_attack(
    model_id: str,
    message: str,
    target: str,
    num_steps: int,
) -> dict:
    if not config.DEVICE.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("GCG attack requires a CUDA GPU")

    dtype = torch.float16 if config.DTYPE == "float16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(config.DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    gcg_config = GCGConfig(num_steps=num_steps)
    result = nanogcg.run(model, tokenizer, message, target, config=gcg_config)

    return {
        "model": model_id,
        "message": message,
        "target": target,
        "num_steps": num_steps,
        "best_loss": result.best_loss,
        "best_string": result.best_string,
        "losses": result.losses,
        "strings": result.strings,
    }


def main() -> None:
    args = parse_args()
    config.require_hf_token()
    config.log_runtime_info()
    model_id = config.resolve_model_id(args.model)

    output = run_gcg_attack(
        model_id=model_id,
        message=args.message,
        target=args.target,
        num_steps=args.num_steps,
    )

    model_name = model_id.replace("/", "_")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    config.write_json(config.results_path() / f"gcg_{model_name}_{timestamp}.json", output)


if __name__ == "__main__":
    main()
