# How well does filtering forget?

This is a replication of the Deep Ignorance paper (O'Brien, 2025) with an mechinterp extension to see to what extend the information has been removed in the weights (simple linear probe).

1. Loading
 Models and eval dataset loaded into cache for later use (loaded from /src/scripts/load into /src/cache)

2. Evaluation
Evals are run with the lm_eval_harnass by EleutherAI. All tasks are built into the lm_eval_harnass, besides WMDP bio cloze and mcqa subset evals, which are provided as parquet files provided by EleutherAI (https://huggingface.co/collections/EleutherAI/deep-ignorance).

Models are evaluated on subsets of the WMDP bio evals (cloze and robust mcqa subsets)
as well as tested for general capability on the MMLU, MMLU-No-Bio, PIQA, LAMBADA and HellaSwag datasets.

3. Adversarial attacks
#TODO 

4. Extention: Confirm WMDP bio information is still present in the model by training a linear probe on it. 
#TODO

## Docker setup

WRKDIR is always ~/Documents/code/AISafety_projects/DeepIgnorance/deep-ignorance-extension/src/
#TODO

## .env setup

Should contain a HuggingFace token for loading models and datasets (HF_TOKEN)
