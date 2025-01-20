"""
python -m tokens2words.analysis.identified_in_patchscopes --results_dir experiments/llama3.1-8b_hebrew_top_5k_words_prompt_beivrit_x,x,x,x,/ --words_list experiments/top_5k_hebrew_words_without_nikud.txt
python -m tokens2words.analysis.identified_in_patchscopes --results_dir experiments/llama3.1-8b_arabic_top_5k_words_prompt_belarabia_x,x,x,x,/ --words_list experiments/top_5k_arabic_words.txt
"""

import argparse
import os
import json

import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from ..utils.file_utils import parse_string_list_from_file

# from word2word import Word2word
# import nltk
# # nltk.download('wordnet')
# from nltk.corpus import wordnet as wn

import logging

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def is_synonym():
    pass


def main(args):
    results_dir = args.results_dir

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

    logger.info("Loading patchscopes results...")
    results_df = pd.read_parquet(os.path.join(results_dir, 'patchscopes_results.parquet'))

    words_list = parse_string_list_from_file(args.words_list)
    words_list = list(dict.fromkeys(words_list))

    found_words = results_df.index.intersection(words_list)
    missing_words = list(set(words_list) - set(found_words))
    results_df = results_df.loc[found_words]

    identified_df = results_df.apply(lambda row: row.str.strip().str.startswith(row.name), axis=1)
    # identified_wo_1_prefix_letter_df = results_df.apply(lambda row: row.str.strip().str.startswith(row.name[1:]), axis=1)
    identified_wo_al_prefix_df = results_df.apply(lambda row: row.str.strip().str.startswith(row.name[2:]) if row.name.startswith('ال') else row.str.strip().str.startswith(row.name), axis=1)
    # weakly_identified_df = results_df.apply(lambda row: row.str.strip().str.contains(row.name), axis=1)

    compute_success_rate = lambda df: df.sum(axis=1).clip(0, 1).mean()
    get_mean_tokenization_len = lambda words: (pd.Series(tokenizer(words)['input_ids']).str.len()-1).mean().item()
    compute_success_rate(identified_df.iloc[:1000])
    mean_tokenization_len = get_mean_tokenization_len(words_list_wo_missing[:1000])

    import pdb; pdb.set_trace()

    # he2en = Word2word("he", "en")
    # en2he = Word2word("en", "he")
    # synsets1 = wn.synsets(word1)

    # logger.info(f"Results saved to: {output_dir}")

    return


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze patchscope results: identification rates, weaker identification, etc.")
    parser.add_argument("--results_dir", type=str, default="./experiments/")
    parser.add_argument("--words_list", type=str, default="./experiments/top_5k_hebrew_words.txt")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
