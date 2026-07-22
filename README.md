# How well does filtering forget?

This is a replication of the Deep Ignorance paper (O'Brien, 2025) with an extension ...................

1. Loading
 Models and eval dataset loaded into cache for later use (loaded from /src/scripts/load into /src/cache)

2. Evaluation
Evals are run with the lm_eval_harnass by EleutherAI. All tasks are built into the harnass or parquet files provided by EleutherAI (https://huggingface.co/collections/EleutherAI/deep-ignorance).

Models are evaluated on subsets of the WMDP bio evals (cloze and robust mcqa subsets)
as well as tested for general capability on the MMLU, MMLU-No-Bio, PIQA, LAMBADA and HellaSwag datasets.

3.  

## Docker setup

WRKDIR is always ~/Documents/code/AISafety_projects/DeepIgnorance/deep-ignorance-extension/src/

## .env setup

Should contain a HuggingFace token for loading models and datasets (HF_TOKEN)