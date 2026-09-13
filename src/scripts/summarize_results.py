"""Summarize saved eval / probe / attack JSON files into a compact table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config


def _acc_from_lm_eval(payload: dict) -> dict[str, float]:
    results = payload.get("results") or {}
    scores: dict[str, float] = {}
    for task, metrics in results.items():
        if not isinstance(metrics, dict):
            continue
        for key in ("acc,none", "acc", "acc_norm,none"):
            if key in metrics:
                scores[task] = float(metrics[key])
                break
    return scores


def _summarize_file(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    stem = path.stem
    row: dict = {"file": path.name}

    if "tamper_curve" in payload:
        curve = payload["tamper_curve"]
        row.update(
            {
                "kind": "finetune",
                "model": payload.get("model"),
                "dataset": payload.get("attack_dataset"),
                "method": payload.get("method"),
                "start_acc": curve[0]["accuracy"] if curve else None,
                "end_acc": curve[-1]["accuracy"] if curve else None,
            }
        )
        return row

    if "knowledge_probe" in payload:
        knowledge = payload.get("knowledge_probe") or []
        domain = payload.get("domain_probe") or []
        last_k = knowledge[-1] if knowledge else {}
        last_d = domain[-1] if domain else {}
        row.update(
            {
                "kind": "probe",
                "model": payload.get("model"),
                "behavioral_acc": payload.get("behavioral_mcqa_acc"),
                "knowledge_test_acc": last_k.get("test_acc"),
                "domain_test_acc": last_d.get("test_acc"),
            }
        )
        return row

    if "best_loss" in payload:
        row.update(
            {
                "kind": "gcg",
                "model": payload.get("model"),
                "best_loss": payload.get("best_loss"),
            }
        )
        return row

    scores = _acc_from_lm_eval(payload)
    if scores:
        row.update({"kind": "eval", "file": path.name, **scores})
        if "cloze" in stem:
            row["suite"] = "cloze"
        elif "mcqa" in stem:
            row["suite"] = "mcqa"
        else:
            row["suite"] = "general"
        return row

    return None


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("No result files found.")
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    widths = {key: max(len(key), *(len(_fmt(row.get(key))) for row in rows)) for key in keys}
    header = "  ".join(key.ljust(widths[key]) for key in keys)
    print(header)
    print("  ".join("-" * widths[key] for key in keys))
    for row in rows:
        print("  ".join(_fmt(row.get(key)).ljust(widths[key]) for key in keys))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize saved Deep Ignorance remake results")
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Directory of JSON results (default: configured RESULTS_DIR)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir) if args.results_dir else config.results_path()
    rows = []
    for path in sorted(results_dir.glob("*.json")):
        row = _summarize_file(path)
        if row:
            rows.append(row)
    print_table(rows)


if __name__ == "__main__":
    main()
