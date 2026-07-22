import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager

import json
import os
import datetime
from dotenv import load_dotenv

def eval_wmdp_cloze(model):
    # Load cached model
    lm = HFLM(pretrained=model, dtype="float32", device="cpu")

    # Build task dictionary
    task_manager = TaskManager(
        include_path="./cache/datasets--EleutherAI--wmdp_bio_cloze"
    )
    task_dict_cloze = task_manager.load(["wmdp_bio_cloze"])

    # run evals
    results_cloze = lm_eval.evaluate(
        lm=lm, task_dict=task_dict_cloze
    )  # NOTE: A valid .yaml file for this task is necessary in /cache
    return results_cloze


def eval_wmdp_robust_mcqa(model):
    # Load cached model
    lm = HFLM(pretrained=model, dtype="float32", device="cpu")

    # Build task dictionary
    task_manager = TaskManager(
        include_path="./cache/datasets--EleutherAI--wmdp_bio_robust_mcqa"
    )
    task_dict_mcqa = task_manager.load(
        [
            "wmdp_bio_robust_bioweapons_and_bioterrorism", "wmdp_bio_robust_dual_use_virology", "wmdp_bio_robust_enhanced_potential_pandemic_pathogens", "wmdp_bio_robust_expanding_access_to_threat_vectors", "reverse_genetics_and_easy_editing, viral_vector_research"
        ]
    )

    # run evals
    results_mcqa = lm_eval.evaluate(
        lm=lm, task_dict=task_dict_mcqa
    )  # NOTE: A valid .yaml file for this task is necessary in /cache
    return results_mcqa


if __name__ == "__main__":
    load_dotenv()
    print(repr(os.environ.get("HF_TOKEN")))

    model = "EleutherAI/pythia-160m"
    model_name = model.replace("/", "__")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs("results", exist_ok=True)
    results_cloze = eval_wmdp_cloze(model)
    with open(f"results/{model_name}_cloze_{timestamp}.json", "w") as f:
        json.dump(results_cloze, f, indent=2)
    
    # results_mcqa = eval_wmdp_robust_mcqa(model)
    # with open(f"results/{model_name}_mcqa_{timestamp}.json", "w") as f:
    #     json.dump(results_mcqa, f, indent=2)

