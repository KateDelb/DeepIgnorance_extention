# Deep Ignorance remake


Replication of [Deep Ignorance: Filtering Pretraining Data Builds Tamper-Resistant Safeguards into Open-Weight LLMs](https://huggingface.co/papers/2508.06601) (O'Brien et al., 2025), plus a linear-probe extension.

The paper asks whether filtering hazardous biosecurity data out of pretraining, optionally combined with circuit breaking and latent adversarial training, prevents models from encoding that knowledge and keeps the safeguard tamper-resistant. This repo does **not** retrain the 6.9B GPT-NeoX models. It evaluates the [released EleutherAI suite](https://huggingface.co/collections/EleutherAI/deep-ignorance) and adds a probe that tests whether WMDP-Bio information is still linearly readable from residual activations.

Local smoke tests default to `EleutherAI/pythia-160m`. Paper comparisons should use the 6.9B Deep Ignorance checkpoints on a GPU host (RunPod or equivalent).

## What is implemented

1. **Prefetch** — download models and eval datasets into `HF_HOME` (`scripts/load_models-datasets.py`).
2. **WMDP-Bio evaluation** — cloze and robust MCQA subsets via lm-eval, using EleutherAI parquet task files.
3. **General capability** — PIQA, HellaSwag, LAMBADA, and MMLU (paper set) or any lm-eval tasks.
4. **Input-space few-shot** — 16-shot WMDP-Bio, matching Section 2.4.
5. **GCG scaffold** — `nanogcg` grey-box suffix search with a caller-supplied message and target.
6. **Fine-tuning / tampering** — full-parameter or LoRA updates on the WMDP-Bio forget corpus or WikiText, with an in-memory WMDP-Bio accuracy curve (Section 3 / Figures 4–5).
7. **rExtension: linear probes** — knowledge probe (corect choice index) and domain probe (WMDP-Bio vs PIQA) on last-token hidden states.

## Repository layout

```
deep-ignorance-extension/
  conf/                     # Hydra job / model / experiment groups
  src/run.py                # Hydra launcher
  src/config.py             # HF cache, tokens, parquet paths
  src/eval/                 # bio, general, in-memory MCQA scoring
  src/probe/linear_probe.py # extension
  src/adversarial_attack/   # GCG, few-shot, fine-tuning
  src/tasks/                # lm-eval YAML for WMDP-Bio parquet
  src/scripts/              # prefetch + result summary
  docker/                   # base / eval / attack images and runners
```

Working directory for Python entry points is `src/`. `import config` and `from eval...` rely on that.

## Local setup

```bash
cd deep-ignorance-extension
cp .env.example .env   # Should contain a HuggingFace token for loading models and datasets (HF_TOKEN)
uv sync
```

`.env` should contain a HuggingFace token with access to the gated models and datasets. Keep `HF_TOKEN` in `.env` only — never in Hydra YAML. Optional path/device overrides:

```
HF_HOME=/path/to/cache
DEVICE=cuda
DTYPE=float16
```

Default cache is the repo-root `DeepIgnorance/cache` directory.

Prefetch (from `src/`):

```bash
python scripts/load_models-datasets.py --models small-test --datasets wmdp-bio-cloze wmdp-bio-robust-mcqa
# or the full paper suite:
python scripts/load_models-datasets.py --all
```

## Running the remake

All commands below are from `src/`. The Hydra launcher is the supported way to compose jobs. Individual argparse scripts remain as thin aliases.

To run the full smoke suite on `EleutherAI/pythia-160m` and write plots:

```bash
python scripts/run_pythia160m.py
python scripts/run_pythia160m.py eval.limit=16 probe.limit=16
```

That runs `job=bio`, `general`, `fewshot`, and `probe` with `experiment=smoke`, plus `job=finetune` when CUDA is available, then calls `visualize_results.py`.

```bash
python -m run job=bio model=small_test
python -m run job=fewshot model=e2e_strong_filter
python -m run experiment=paper_eval job=probe model=unfiltered
python -m run experiment=paper_tamper model=e2e_strong_filter job=finetune
python -m run -m job=bio,probe model=unfiltered,e2e_strong_filter,unfiltered_cb_lat
```

Inspect a composed config without running anything: `python -m run --cfg job job=fewshot model=e2e_strong_filter`.

| Group | Choices |
|---|---|
| `job` | `bio`, `general`, `fewshot`, `probe`, `finetune`, `all` |
| `model` | `small_test`, `unfiltered`, `strong_pt_weak_anneal`, `e2e_strong_filter`, `unfiltered_cb`, `e2e_strong_filter_cb`, `unfiltered_cb_lat`, `e2e_strong_filter_cb_lat` |
| `experiment` | `smoke` (default), `paper_eval`, `paper_tamper` |

Override any leaf with Hydra syntax, e.g. `eval.limit=32` or `probe.all_layers=true`. Resolved YAML is written under `RESULTS_DIR` (`src/results/` locally).

### WMDP-Bio

```bash
python -m run job=bio model=small_test
python -m run experiment=paper_eval job=bio model=e2e_strong_filter
# argparse alias:
python eval/bio_eval.py --model EleutherAI/pythia-160m
```

Robust MCQA runs automatically when that parquet cache is present (`eval.mcqa=auto`). Set `eval.mcqa=false` to skip it. `job=fewshot` is the paper's 16-shot setting.

### General capability

```bash
python -m run job=general model=small_test
python -m run experiment=paper_eval job=general model=unfiltered
# argparse alias:
python eval/general_capability_eval.py --model EleutherAI/pythia-160m --tasks piqa --limit 100
```

### Linear probe (extension)

```bash
python -m run job=probe model=small_test
python -m run experiment=paper_eval job=probe model=e2e_strong_filter probe.all_layers=true
# argparse alias:
python probe/linear_probe.py --model EleutherAI/pythia-160m --limit 64
```

The knowledge probe predicts the correct multiple-choice index from the last-token residual. The domain probe classifies WMDP-Bio vs format-matched PIQA prompts. Compare:

- high behavioral accuracy → the model still *uses* the information
- high probe accuracy, low behavioral accuracy → the information is still linearly present
- near-chance probe accuracy → the information is not linearly readable

### Fine-tuning attack

Requires CUDA. Smoke tests cap the run with `finetune.max_steps`; the paper recipe is `experiment=paper_tamper`.

```bash
python -m run experiment=paper_tamper job=finetune model=e2e_strong_filter
python -m run job=finetune model=e2e_strong_filter finetune.attack_dataset=wikitext finetune.max_steps=20
# argparse alias:
python adversarial_attack/finetuning_attacks.py \
  --model e2e-strong-filter \
  --attack-dataset wmdp_bio_forget \
  --method lora \
  --max-steps 20
```

`finetune.attack_dataset=wikitext` is the benign control. `finetune.method=full` is the full-parameter variant. Paper defaults (2 epochs, ~305M tokens, batch 16, context 2048, lr `2e-5`) live in `conf/experiment/paper_tamper.yaml`.

### GCG scaffold

Requires CUDA. Defaults to a non-bio message/target.

```bash
python adversarial_attack/gcg_attack.py --model EleutherAI/pythia-160m --num-steps 50
```

### Summarize and plot results

```bash
python scripts/summarize_results.py
python scripts/visualize_results.py
python scripts/visualize_results.py results/smoke_probe_pythia160m.json
python scripts/visualize_results.py --kind probe
```

The visualizer infers experiment kind from the filename (`probe`, `cloze`/`mcqa`, `piqa`/`mmlu`, `ft_`, `gcg_`) and the JSON shape, then writes one PNG per kind under `src/results/plots/`:

- **probe** — behavioral vs knowledge/domain probe accuracy (or accuracy vs layer if `--all-layers`)
- **bio** — WMDP-Bio cloze / MCQA accuracy bars
- **general** — PIQA / HellaSwag / LAMBADA / MMLU bars
- **finetune** — tamper curve (accuracy vs tokens)
- **gcg** — GCG loss vs step

JSON files are written under `src/results/` locally, or `/workspace/results` in Docker.

## Docker / RunPod

From `deep-ignorance-extension/` (needs `.env` with `HF_TOKEN`):

```bash
make build    # build the app image
make up       # start the container in the background
make down     # stop and remove it
```

`src/` and `conf/` are mounted, so code edits show up without a rebuild. Cache is `../cache`, results are `src/results`. Exec into the running container (WORKDIR `/app/src`):

```bash
docker compose exec app bash
docker compose exec app python scripts/run_pythia160m.py
```

One-shot images still work if you prefer `docker build` / `docker run`:

```bash
cd deep-ignorance-extension
docker build -f docker/Dockerfile.base -t deep-ignorance-base:latest .
docker build -f docker/Dockerfile.eval -t deep-ignorance-eval:latest .
docker build -f docker/Dockerfile.attack -t deep-ignorance-attack:latest .
```

Mount the shared cache and results volumes. From this directory the cache lives one level up (`../cache`).

```bash
# Prefetch
docker run --rm --gpus all --env-file .env \
  -v "$PWD/../cache:/workspace/cache" \
  -v "$PWD/src/results:/workspace/results" \
  deep-ignorance-base:latest /app/docker/prefetch.sh

# Bio + general + probe
docker run --rm --gpus all --env-file .env \
  -v "$PWD/../cache:/workspace/cache" \
  -v "$PWD/src/results:/workspace/results" \
  deep-ignorance-eval:latest --suite all --model small_test

# Few-shot WMDP-Bio
docker run --rm --gpus all --env-file .env \
  -v "$PWD/../cache:/workspace/cache" \
  -v "$PWD/src/results:/workspace/results" \
  deep-ignorance-eval:latest --suite fewshot --model e2e_strong_filter

# Paper eval overrides
docker run --rm --gpus all --env-file .env \
  -v "$PWD/../cache:/workspace/cache" \
  -v "$PWD/src/results:/workspace/results" \
  deep-ignorance-eval:latest --suite bio --model unfiltered --experiment paper_eval

# GCG
docker run --rm --gpus all --env-file .env \
  -v "$PWD/../cache:/workspace/cache" \
  -v "$PWD/src/results:/workspace/results" \
  deep-ignorance-attack:latest --model EleutherAI/pythia-160m --num-steps 50

# Fine-tune / tamper curve
docker run --rm --gpus all --env-file .env \
  -v "$PWD/../cache:/workspace/cache" \
  -v "$PWD/src/results:/workspace/results" \
  --entrypoint /app/docker/run_finetune.sh \
  deep-ignorance-base:latest \
  --model e2e_strong_filter --attack-dataset wikitext --method lora --max-steps 20
```

Eval/finetune wrappers also accept extra Hydra `key=value` overrides after the flags.

`docker compose` equivalents (`prefetch`, `eval`, `probe`, `attack`, `finetune`) are in `docker-compose.yml`. Build the base image first so `--build` on the eval/attack services can see `deep-ignorance-base:latest`, or build those images with `docker build` as above.

On a RunPod box, clone this repo, copy `.env`, and run the same commands. `WORKDIR` inside the image is `/app/src`.

## Suggested paper comparison

Evaluate at least:

| Registry name | HuggingFace id |
|---|---|
| `unfiltered` | `EleutherAI/deep-ignorance-unfiltered` |
| `strong-pt-weak-anneal` | `EleutherAI/deep-ignorance-strong-filter-pt-weak-filter-anneal` |
| `e2e-strong-filter` | `EleutherAI/deep-ignorance-e2e-strong-filter` |
| `unfiltered-cb-lat` | `EleutherAI/deep-ignorance-unfiltered-cb-lat` |
| `e2e-strong-filter-cb-lat` | `EleutherAI/deep-ignorance-e2e-strong-filter-cb-lat` |

For each model: 0-shot cloze + robust MCQA, 16-shot cloze, general tasks with `experiment=paper_eval`, last-layer (and optionally all-layer) probes, and LoRA + full fine-tunes via `experiment=paper_tamper`.

## Notes

- Eval JSON no longer stores per-sample documents (`log_samples=False`).
- Probe and attack logs report counts and accuracies only; item text is not printed.
- `cais/wmdp-bio-forget-corpus` is gated. Request access before the adversarial fine-tune.

