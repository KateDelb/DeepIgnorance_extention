"""Plot saved remake results. Kind is inferred from the filename and payload.

    python scripts/visualize_results.py
    python scripts/visualize_results.py src/results/smoke_probe_pythia160m.json
    python scripts/visualize_results.py --kind probe --out src/results/plots
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

import config

KINDS = ("bio", "general", "probe", "finetune", "gcg")

_BIO_NAME_TOKENS = ("cloze", "mcqa", "wmdp", "fewshot", "_fs")
_GENERAL_NAME_TOKENS = ("piqa", "hellaswag", "lambada", "mmlu")
_WMDP_TASK_TOKENS = ("wmdp", "cloze", "mcqa")


def _load_json(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def short_model_name(model: str | None, fallback: str = "") -> str:
    if not model:
        return fallback
    name = model.replace("EleutherAI/", "").replace("deep-ignorance-", "")
    return name.replace("__", "/").strip("/") or fallback


def infer_kind(path: Path, payload: dict) -> str | None:
    """Prefer payload shape; fall back to tokens in the results filename."""
    stem = path.stem.lower()
    if "tamper_curve" in payload or stem.startswith("ft_") or "tamper" in stem:
        return "finetune"
    if "knowledge_probe" in payload or "probe" in stem:
        return "probe"
    if "best_loss" in payload or stem.startswith("gcg_"):
        return "gcg"
    if any(token in stem for token in _BIO_NAME_TOKENS):
        return "bio"
    if any(token in stem for token in _GENERAL_NAME_TOKENS):
        return "general"

    tasks = payload.get("results")
    if isinstance(tasks, dict) and tasks:
        names = " ".join(tasks)
        if any(token in names for token in _WMDP_TASK_TOKENS):
            return "bio"
        return "general"
    return None


def discover_result_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    files = [p for p in root.rglob("*.json") if ".hydra" not in p.parts]
    return sorted(files)


def collect_records(paths: list[Path], kind_filter: str | None = None) -> dict[str, list[tuple[Path, dict]]]:
    grouped: dict[str, list[tuple[Path, dict]]] = defaultdict(list)
    for path in paths:
        payload = _load_json(path)
        if payload is None:
            continue
        kind = infer_kind(path, payload)
        if kind is None:
            continue
        if kind_filter and kind != kind_filter:
            continue
        grouped[kind].append((path, payload))
    return grouped


def _model_from_path(path: Path, payload: dict) -> str:
    if payload.get("model"):
        return short_model_name(str(payload["model"]))
    stem = path.stem
    for suffix in ("_probe_", "_cloze_", "_mcqa_", "_piqa"):
        if suffix in stem:
            return short_model_name(stem.split(suffix)[0])
    return short_model_name(stem.rsplit("_", 2)[0], fallback=stem)


def _acc_with_stderr(payload: dict) -> dict[str, tuple[float, float | None]]:
    results = payload.get("results") or {}
    scores: dict[str, tuple[float, float | None]] = {}
    for task, metrics in results.items():
        if not isinstance(metrics, dict):
            continue
        acc = None
        for key in ("acc,none", "acc", "acc_norm,none"):
            if key in metrics:
                acc = float(metrics[key])
                break
        if acc is None:
            continue
        stderr = metrics.get("acc_stderr,none", metrics.get("acc_stderr"))
        scores[task] = (acc, float(stderr) if stderr is not None else None)
    return scores


def _filter_family_label(name: str) -> str:
    n = name.lower().replace("_", "-")
    if "pythia" in n or n in {"small-test", "small_test"}:
        return "No filter\n(pythia-160m smoke)"
    extras = []
    if "cb-lat" in n:
        extras.append("CB+LAT")
    elif n.endswith("-cb") or "-cb-" in n:
        extras.append("CB")
    extra = f"\n+ {' + '.join(extras)}" if extras else ""
    if n.startswith("unfiltered"):
        return f"Unfiltered{extra}"
    if "e2e-strong-filter" in n:
        return f"End-to-end\nstrong filter{extra}"
    if "strong-pt-weak" in n or "strong-filter-pt-weak" in n:
        return "Strong filter PT\n/ weak anneal"
    return name


def _probe_label(path: Path, payload: dict) -> str:
    return _filter_family_label(_model_from_path(path, payload))


def _best_probe_per_model(records: list[tuple[Path, dict]]) -> list[tuple[Path, dict]]:
    """Keep one file per model: prefer layer sweeps, then more items, then newer files."""
    best: dict[str, tuple[tuple, Path, dict]] = {}
    for path, payload in records:
        key = str(payload.get("model") or path.stem)
        score = (
            len(payload.get("knowledge_probe") or []),
            int(payload.get("n_items") or 0),
            path.stat().st_mtime,
        )
        if key not in best or score > best[key][0]:
            best[key] = (score, path, payload)
    return [(path, payload) for _, path, payload in best.values()]


def _resolved_layer(row: dict, n_layers: int, index: int) -> int:
    layer = row.get("layer", index)
    if layer is None:
        return index
    layer = int(layer)
    if layer < 0:
        return n_layers + 1 + layer
    return layer


def _plot_probe_how(records: list[tuple[Path, dict]], ax) -> None:
    labels, behavioral, knowledge, domain = [], [], [], []
    k_chance, d_chance = 0.25, 0.5
    for path, payload in records:
        labels.append(_probe_label(path, payload))
        k_rows = payload.get("knowledge_probe") or [{}]
        d_rows = payload.get("domain_probe") or [{}]
        behavioral.append(float(payload.get("behavioral_mcqa_acc") or 0.0))
        knowledge.append(float(k_rows[-1].get("test_acc") or 0.0))
        domain.append(float(d_rows[-1].get("test_acc") or 0.0))
        k_chance = float(k_rows[-1].get("uniform_chance") or k_chance)
        d_chance = float(d_rows[-1].get("uniform_chance") or d_chance)

    x = range(len(labels))
    width = 0.24
    ax.bar([i - width for i in x], behavioral, width, color="#4c72b0", label="Uses the answer (behavioral MCQA)")
    ax.bar(list(x), knowledge, width, color="#dd8452", label="Answer is linearly readable (knowledge probe)")
    ax.bar([i + width for i in x], domain, width, color="#55a868", label="Bio vs non-bio is linearly readable (domain probe)")
    ax.axhline(k_chance, color="#dd8452", linestyle=":", linewidth=1, label=f"Knowledge chance ({k_chance:.2f})")
    ax.axhline(d_chance, color="#55a868", linestyle="--", linewidth=1, label=f"Domain chance ({d_chance:.2f})")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.08)
    ax.set_title("How the information shows up")
    ax.legend(loc="upper left", fontsize=8, frameon=False)


def _plot_probe_where(records: list[tuple[Path, dict]], ax) -> None:
    drew = False
    for path, payload in records:
        n_layers = int(payload.get("n_layers") or 0)
        label = _probe_label(path, payload).replace("\n", " ")
        knowledge = payload.get("knowledge_probe") or []
        domain = payload.get("domain_probe") or []
        if knowledge:
            xs = [_resolved_layer(row, n_layers, i) for i, row in enumerate(knowledge)]
            ax.plot(xs, [row.get("test_acc") for row in knowledge], marker="o", label=f"{label} — knowledge")
            drew = True
        if domain:
            xs = [_resolved_layer(row, n_layers, i) for i, row in enumerate(domain)]
            ax.plot(xs, [row.get("test_acc") for row in domain], marker="s", linestyle="--", label=f"{label} — domain")
            drew = True
        if payload.get("behavioral_mcqa_acc") is not None:
            ax.axhline(float(payload["behavioral_mcqa_acc"]), linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Residual stream layer (0 = embeddings, last = final block)")
    ax.set_ylabel("Linear probe test accuracy")
    ax.set_ylim(0, 1.08)
    max_layer = max((int(p.get("n_layers") or 0) for _, p in records), default=0)
    if max_layer:
        ax.set_xlim(-0.5, max_layer + 0.5)
    if drew and any(len(p.get("knowledge_probe") or []) > 1 for _, p in records):
        ax.set_title("Where it sits in the network")
    else:
        ax.set_title("Where it sits (last layer only — re-run with probe.all_layers=true for a depth curve)")
    ax.legend(fontsize=8, frameon=False)


_PROBE_CAPTION = (
    "Knowledge probe: a linear classifier reads the correct WMDP-Bio choice from the last-token residual. "
    "Domain probe: it only tells WMDP-Bio prompts from format-matched PIQA. "
    "Behavioral: the model itself picks the right choice. "
    "Knowledge ≫ behavioral means the fact is still in the residual stream but not used. "
    "Knowledge ≈ chance means it is not linearly present. "
    "Compare Unfiltered vs strong-filter columns to see whether pretraining filters remove the representation, not just the answers."
)


def plot_probe(records: list[tuple[Path, dict]], ax) -> str:
    records = _best_probe_per_model(records)
    fig = ax.figure
    ax.remove()
    gs = fig.add_gridspec(2, 1, height_ratios=[1.05, 1], hspace=0.42)
    ax_how = fig.add_subplot(gs[0])
    ax_where = fig.add_subplot(gs[1])
    _plot_probe_how(records, ax_how)
    _plot_probe_where(records, ax_where)
    fig.suptitle("WMDP-Bio information: used vs still linearly readable", fontsize=12, y=0.98)
    fig.text(0.5, 0.01, textwrap.fill(_PROBE_CAPTION, width=110), ha="center", va="bottom", fontsize=8)
    return "probe"


def plot_eval_bars(records: list[tuple[Path, dict]], ax, title: str) -> str:
    series: dict[str, dict[str, tuple[float, float | None]]] = {}
    tasks: set[str] = set()
    for path, payload in records:
        model = _model_from_path(path, payload)
        scores = _acc_with_stderr(payload)
        if not scores:
            continue
        series[model] = scores
        tasks.update(scores)

    task_list = sorted(tasks)
    models = sorted(series)
    if not task_list or not models:
        ax.set_title(f"{title} (no accuracy fields)")
        return "eval_empty"

    width = 0.8 / max(len(models), 1)
    x = range(len(task_list))
    for i, model in enumerate(models):
        ys = [series[model][task][0] if task in series[model] else 0.0 for task in task_list]
        yerr = [
            series[model][task][1] if task in series[model] and series[model][task][1] is not None else float("nan")
            for task in task_list
        ]
        err = yerr if any(v == v for v in yerr) else None
        ax.bar([t + (i - (len(models) - 1) / 2) * width for t in x], ys, width, yerr=err, label=model)
    ax.set_xticks(list(x))
    ax.set_xticklabels(task_list, rotation=25, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend()
    return "eval"


def plot_finetune(records: list[tuple[Path, dict]], ax) -> str:
    for path, payload in records:
        curve = payload.get("tamper_curve") or []
        if not curve:
            continue
        model = _model_from_path(path, payload)
        method = payload.get("method", "")
        dataset = payload.get("attack_dataset", "")
        xs = [row.get("tokens_seen", row.get("step", i)) for i, row in enumerate(curve)]
        ys = [row.get("accuracy") for row in curve]
        ax.plot(xs, ys, marker="o", label=f"{model} {method} {dataset}".strip())
    ax.set_xlabel("Tokens seen")
    ax.set_ylabel("WMDP-Bio accuracy")
    ax.set_title("Fine-tuning tamper curve")
    ax.set_ylim(0, 1.05)
    ax.legend()
    return "finetune"


def plot_gcg(records: list[tuple[Path, dict]], ax) -> str:
    for path, payload in records:
        losses = payload.get("losses") or []
        model = _model_from_path(path, payload)
        if losses:
            ax.plot(range(len(losses)), losses, label=model)
        elif payload.get("best_loss") is not None:
            ax.axhline(float(payload["best_loss"]), label=f"{model} best_loss")
    ax.set_xlabel("GCG step")
    ax.set_ylabel("Loss")
    ax.set_title("GCG optimization")
    ax.legend()
    return "gcg"


_PLOTTERS = {
    "probe": lambda recs, ax: plot_probe(recs, ax),
    "bio": lambda recs, ax: plot_eval_bars(recs, ax, "WMDP-Bio accuracy"),
    "general": lambda recs, ax: plot_eval_bars(recs, ax, "General capability accuracy"),
    "finetune": plot_finetune,
    "gcg": plot_gcg,
}


def render_kind(kind: str, records: list[tuple[Path, dict]], out_path: Path) -> Path:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 8.5) if kind == "probe" else (9, 5))
    _PLOTTERS[kind](records, ax)
    if kind == "probe":
        fig.subplots_adjust(left=0.1, right=0.98, top=0.9, bottom=0.18, hspace=0.45)
    else:
        fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_results(
    source: Path | None = None,
    out: Path | None = None,
    kind: str | None = None,
) -> list[Path]:
    source = Path(source) if source else config.results_path()
    files = discover_result_files(source)
    grouped = collect_records(files, kind_filter=kind)
    if not grouped:
        print(f"No plottable result files under {source}")
        return []

    out_dir = Path(out) if out else (source if source.is_dir() else source.parent) / "plots"
    written = []
    for plot_kind, records in grouped.items():
        out_path = out_dir / f"{plot_kind}.png"
        written.append(render_kind(plot_kind, records, out_path))
        print(f"{plot_kind}: {len(records)} file(s) -> {out_path}")
    print(f"Wrote {len(written)} figure(s) to {out_dir}")
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Deep Ignorance remake results by experiment kind")
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="JSON file or results directory (default: RESULTS_DIR)",
    )
    parser.add_argument(
        "--kind",
        choices=KINDS,
        default=None,
        help="Only plot this kind. Default: infer from each file and plot every kind found.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Directory for PNG files (default: <results>/plots)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_results(
        source=Path(args.path) if args.path else None,
        out=Path(args.out) if args.out else None,
        kind=args.kind,
    )


if __name__ == "__main__":
    main()
