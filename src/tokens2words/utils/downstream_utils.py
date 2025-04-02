from lm_eval.evaluator import simple_evaluate
from lm_eval import tasks
from lm_eval.models.huggingface import HFLM
import numpy as np
import os
import json
from collections import defaultdict

import logging

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


task_to_limited_samples = defaultdict(lambda: 100)
task_to_limited_samples["minerva_math"] = 5
task_to_limited_samples["arabicmmlu"] = 5
task_to_limited_samples["xquad_ar"] = 200

generation_tasks = []


def evaluate_model(model, tokenizer, limited=False, task_names=None, output_path=None):
    """
    Evaluate model on multiple benchmarks using lm-evaluation-harness.
    """
    task_configs = {
        "xquad_ar": {
            "num_fewshot": 1,
            "gen_kwargs": "max_tokens=32",
        },
        "xnli_ar": {
            "num_fewshot": 3,
            "gen_kwargs": "max_tokens=32",
        },
        "xstorycloze_ar": {
            "num_fewshot": 0,
        },
        "arabicmmlu": {
            "num_fewshot": 3,
        },
        "tinyMMLU": {
            "num_fewshot": 3,
        },
        "tinyArc": {
            "num_fewshot": 25,
            "gen_kwargs": "max_tokens=100",
        },
        "squadv2": {
            "num_fewshot": 1,
            "gen_kwargs": "max_tokens=32",
        },
    }

    if task_names is not None:
        task_configs = {task_name: task_configs[task_name] for task_name in task_names}

    adapted_model = HFLM(
        model,
        tokenizer=tokenizer,
    )

    # Evaluate the model on each task
    results = {}
    for task_name, config in task_configs.items():
        # Evaluate the model on the task
        curr_output_path = os.path.join(output_path, task_name) if output_path else None
        os.makedirs(curr_output_path, exist_ok=True)
        result = simple_evaluate(
            model=adapted_model,
            tasks=[task_name],
            limit=task_to_limited_samples[task_name] if limited else None,
            batch_size=1,
            num_fewshot=config.get("num_fewshot", None),
            gen_kwargs=config.get("gen_kwargs", None),
            log_samples=True if curr_output_path else False,
        )

        results[task_name] = result['results'][task_name]

        logger.info(f"Downstream eval on {task_name}: {results[task_name]}")

        if curr_output_path:
            preds = (sum((result['samples'][k] for k in sorted(result['samples'])), [])
                                 if task_name not in result['samples'] else result['samples'][task_name])
            final_preds = list()
            for pred in preds:
                pred.pop("metrics")
                pred = {k: v for k, v in pred.items() if k in ["doc_id", "doc", "target", "arguments", "resps", "filtered_resps", "exact"]}
                final_preds.append(pred)
            with open(os.path.join(curr_output_path, "preds.json"), "w") as fp:
                json.dump(final_preds, fp, indent=4, ensure_ascii=False)

    return results
