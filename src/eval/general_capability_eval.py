import lm_eval
from lm_eval.models.huggingface import HFLM

import json
import datetime
from dotenv import load_dotenv

TASKS = [""]
MODELS = {
    # "small-test":       "EleutherAI/pythia-160m"
    "unfiltered": "EleutherAI/deep-ignorance-unfiltered",
    "strong-pt-weak-anneal": "EleutherAI/deep-ignorance-strong-filter-pt-weak-filter-anneal",
    "e2e-strong-filter": "EleutherAI/deep-ignorance-e2e-strong-filter",
}


def eval_general_capability(model, tasks):
    # Load cached model
    lm = HFLM(pretrained=model, dtype="float32", device="cpu")

    results = lm_eval.simple_evaluate(
        model=lm,
        model_args="pretrained=gpt2",
        tasks=tasks,
        limit=100,
    )
    return results


def json_default(obj):
    """Fallback for objects json.dump can't serialize natively."""
    try:
        return str(obj)
    except Exception:
        return f"<non-serializable: {type(obj).__name__}>"


if __name__ == "__main__":
    load_dotenv()

    for model in MODELS:
        model = MODELS[model]
        model_name = model.replace("/", "_")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        tasks = ["piqa"]
        tasks_string = "-".join(tasks)

        print(
            f"######################### TASK {tasks} on {model} #########################"
        )
        results_general_cap = eval_general_capability(model, tasks)
        with open(f"results/{model_name}_{tasks_string}_{timestamp}.json", "w") as f:
            json.dump(results_general_cap, f, indent=2, default=json_default)
