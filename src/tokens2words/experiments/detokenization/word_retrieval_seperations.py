import os
import argparse

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...utils.enums import MultiTokenKind
from ...word_retriever import AnalysisWordRetriever
from ...utils.file_utils import save_df_to_dir

from utils import MODELS_MAP, dtype_arg, load_dataset_for_analysis


def process_dataset_for_models(models_info, dataset):
    for model_name, model_path in models_info.items():
        print(f"Processing model: {model_name}")

        model = AutoModelForCausalLM.from_pretrained(model_path).to(device).to(dtype)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        word_retriever = AnalysisWordRetriever(model, tokenizer, MultiTokenKind.Split, add_context=True,
                                       model_name=model_name, device=device, dataset=dataset)

        for add_context in [True]:
            print(f"Processing model: {model_name} with context: {add_context}")
            results = word_retriever.retrieve_words_in_dataset(number_of_examples_to_retrieve=4)
            results_df = pd.DataFrame(results)
            print(results_df)

            save_df_to_dir(
                results_df=results_df,
                base_dir="Data",
                sub_dirs=["outputs", "retrieval", "multi_tokens"],
                file_name_format=f"{model_name}_results_with_context_separation.csv" if add_context else f"{model_name}_results_without_context_separation.csv",
                add_context=True,
                model_name=model_name,
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", nargs="+", type=str, default="Llama-3-8B")
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--hf_token", type=str, default=None,)
    parser.add_argument("--dtype", type=dtype_arg, default=torch.bfloat16)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    models_info = {
        model_name: MODELS_MAP[model_name] for model_name in args.model_name
    } if MODELS_MAP != "all" else MODELS_MAP
    dtype = args.dtype
    dataset = load_dataset_for_analysis(args.dataset)

    process_dataset_for_models(models_info, dataset)
