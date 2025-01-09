"""
python -m tokens2words.run_patchscopes --exp_name llama3.1-8b_wiki40b_hebrew_words_prompt=beivrit_x,x,x,x, --dataset_name wiki40b --dataset_language he --dataset_max_samples 1000 --patchscopes_max_words 10000 --patchscopes_prompt "בעברית: X, X, X, X," --words_filter_numeric --words_filter_en
python -m tokens2words.run_patchscopes --exp_name llama3.1-8b_hebrew_twitter_top_1k_words_prompt_beivrit_x,x,x,x, --words_list experiments/top_1k_hebrew_words_twitter.txt --patchscopes_prompt "בעברית: X, X, X, X,"
python -m tokens2words.run_patchscopes --exp_name llama3.1-8b_hebrew_top_5k_words_prompt_beivrit_x,x,x,x, --words_list experiments/top_5k_hebrew_words_without_nikud.txt --patchscopes_prompt "בעברית: X, X, X, X,"
python -m tokens2words.run_patchscopes --exp_name llama3.1-8b_arabic_top_5k_words_prompt_belarabia_x,x,x,x, --words_list experiments/top_5k_arabic_words.txt --patchscopes_prompt "بالعربية: X, X, X, X,"
"""

import argparse
import os
import json
import random

import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from accelerate.utils import set_seed
from collections import defaultdict

from .word_retriever import PatchscopesRetriever
from .utils.file_utils import parse_string_list_from_file
from .utils.data_utils import load_lm_dataset, extract_new_words_from_dataset

import logging

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_patchscopes_on_list(
        model, tokenizer, words_list,
        extraction_prompt="X", patchscopes_prompt="X, X, X, X,", prompt_target="X",
        patchscopes_n_new_tokens=5,
):
    patchscopes_retriever = PatchscopesRetriever(model, tokenizer, extraction_prompt, patchscopes_prompt, prompt_target)

    model.eval()
    outputs = defaultdict(dict)
    for word in tqdm(words_list, total=len(words_list), desc="Running patchscopes...", miniters=10,):
        patchscopes_description_by_layers, _ = patchscopes_retriever.get_hidden_states_and_retrieve_word(
            word, num_tokens_to_generate=patchscopes_n_new_tokens)

        outputs[word] = {
            layer_i: patchscopes_result
            for layer_i, patchscopes_result in enumerate(patchscopes_description_by_layers)}

    return outputs


def main(args):
    set_seed(args.seed)

    output_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Loading model...")
    mixed_precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    accelerator = Accelerator(mixed_precision=mixed_precision)
    model = AutoModelForCausalLM.from_pretrained(args.model_name,
                                                 torch_dtype=torch.bfloat16 if mixed_precision == "bf16" else torch.float16)
    model = accelerator.prepare(model)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if args.words_list:
        logger.info(f"Loading words list from file: {args.words_list}")
        words_list = parse_string_list_from_file(args.words_list, args.words_list_delimiter)
        # remove duplicates but keep original word order
        words_list = list(dict.fromkeys(words_list))
    elif args.dataset_name:
        logger.info(f"Building words list from dataset: {args.dataset_name}")
        dataset = load_lm_dataset(args.dataset_name, args.dataset_language)[args.dataset_split]
        words_list, word_freqs = extract_new_words_from_dataset(dataset, tokenizer, max_samples=args.dataset_max_samples)

    if args.words_filter_numeric:
        words_list = [s for s in words_list if not any(c.isdigit() for c in s)]
    if args.words_filter_en:
        words_list = [s for s in words_list if not any('a' <= c <= 'z' or 'A' <= c <= 'Z' for c in s)]

    if args.patchscopes_max_words:
        # words_list = random.sample(words_list, args.patchscopes_max_words)
        words_list = words_list[:args.patchscopes_max_words]

    patchscopes_results = run_patchscopes_on_list(
        model, tokenizer, words_list,
        args.extraction_prompt,
        args.patchscopes_prompt,
        args.patchscopes_prompt_target,
        args.patchscopes_generate_n_tokens,
    )
    logger.info("Done running patchscopes!")
    
    results_df = pd.DataFrame.from_records(patchscopes_results)

    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(os.path.join(output_dir, f"patchscopes_results.csv"))
    results_df.to_parquet(os.path.join(output_dir, f"patchscopes_results.parquet"))

    config = vars(args)
    config["total_words"] = len(results_df)
    with open(os.path.join(output_dir, f"config.json"), "w") as config_file:
        json.dump(config, config_file, indent=4)

    logger.info(f"Results saved to: {output_dir}")

    return


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute patchscope results for a given list of words, or for words extracted from a dataset.")
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--extraction_prompt", type=str, default="X")
    parser.add_argument("--patchscopes_prompt", type=str, default="X, X, X, X,")
    parser.add_argument("--patchscopes_prompt_target", type=str, default="X")
    parser.add_argument("--patchscopes_generate_n_tokens", type=int, default=20)
    parser.add_argument("--patchscopes_max_words", type=int, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_language", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="validation")
    parser.add_argument("--dataset_max_samples", type=int, default=None)
    parser.add_argument("--words_list", type=str, default=None)
    parser.add_argument("--words_list_delimiter", type=str, default=None)
    parser.add_argument("--words_filter_en", action="store_true", default=False)
    parser.add_argument("--words_filter_numeric", action="store_true", default=False)
    parser.add_argument("--output_dir", type=str, default="./experiments/")

    args = parser.parse_args()

    assert args.dataset_name is not None or args.words_list is not None, \
        "Please pass either a dataset name (--dataset_name) " \
        "or a path to a file containing a list of words (--words_list)"

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
