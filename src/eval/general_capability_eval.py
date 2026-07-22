import lm_eval
from lm_eval.models.huggingface import HFLM

import json
from json import json_de
import os
import datetime
from dotenv import load_dotenv

TASKS = [""]

def eval_general_capability(model):
    # Load cached model
    lm = HFLM(pretrained=model, dtype="float32", device="cpu")

    results = lm_eval.simple_evaluate(
    model=lm,
    model_args="pretrained=gpt2",
    tasks=["piqa"],
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

    model = "EleutherAI/pythia-160m"
    model_name = model.replace("/", "__")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    results_general_cap = eval_general_capability(model)
    print(results_general_cap)
    with open(f"results/{model_name}_piqa_{timestamp}.json", "w") as f:
        json.dump(results_general_cap, f, indent=2, default=json_default)