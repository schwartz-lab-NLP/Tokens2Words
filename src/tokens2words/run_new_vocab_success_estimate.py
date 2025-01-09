"""

"""

import argparse
import os
import gc
import json
import pdb

import pandas as pd
import numpy as np
from tabulate import tabulate
from tqdm import tqdm
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tokenizers import AddedToken
from accelerate import Accelerator
from accelerate.utils import set_seed
from collections import defaultdict
from copy import deepcopy
import types

from .word_retriever import PatchscopesRetriever
from .representation_translator import LinearRepresentationTranslators, ProcrustesRepresentationTranslators
from .vocab_modifier import DetokenizationVocabularyExpander, HeuristicDetokenizationVocabularyExpander
from .utils.file_utils import parse_string_list_from_file
from .utils.data_utils import load_lm_dataset, extract_new_words_from_dataset, tokenize_and_prepare_dataset
from .utils.eval_utils import eval_next_word_prediction, count_tokens_in_dataset
from .run_vocab_expansion_eval import prepare_translators, prepare_patchscopes_retriever

import logging

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def prepare_new_words(
        args, tokenizer):
    
    new_words = parse_string_list_from_file(args.words_list, args.words_list_delimiter)
    new_words = [w for w in new_words if not tokenizer.vocab.get(w, False)]
    if args.max_words is not None:
        new_words = new_words[:args.max_words]
    baseline_tokenization = {w: tokenizer.encode(w, add_special_tokens=False, return_tensors="pt")[0]
                             for w in new_words}

    return new_words, baseline_tokenization


def test_text_generation(model, tokenizer, prompt="Once upon a time", num_tokens=20):

    def _generate_greedy(input_ids):
        # Greedy decoding
        greedy_output = model.generate(input_ids, max_length=input_ids.shape[1] + num_tokens, do_sample=False, temperature=None, top_p=None)
        greedy_decoded = tokenizer.decode(greedy_output[0], skip_special_tokens=True)
        logger.info(f"Greedy Decoding --- {greedy_decoded}\nToken IDs: {greedy_output[0].tolist()}")

    def _generate_top_p(input_ids):
        # Top-p sampling with temperature
        top_p_output = model.generate(input_ids, max_length=input_ids.shape[1] + num_tokens, do_sample=True, top_p=0.9, temperature=0.7)
        top_p_decoded = tokenizer.decode(top_p_output[0], skip_special_tokens=True)
        logger.info(f"Top-p Sampling with Temperature --- {top_p_decoded}\nToken IDs: {top_p_output[0].tolist()}")

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    _generate_greedy(input_ids)
    _generate_top_p(input_ids)


def main(args):
    set_seed(args.seed)

    output_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    base_tokenizer = deepcopy(tokenizer)
    logger.info("Preparing list of words to estimate expansion success for...")
    new_words, orig_tokenization = prepare_new_words(args, tokenizer)
    logger.info(f"Found {len(new_words)} new words: {new_words[:100]} and so on...")

    logger.info("Loading model...")
    mixed_precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    accelerator = Accelerator(mixed_precision=mixed_precision)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16 if mixed_precision == "bf16" else torch.float16)
    model = accelerator.prepare(model)
    model.eval()

    logger.info("Preparing projections to embedding and lm_head spaces...")
    translators, save_translators = prepare_translators(args, model, tokenizer)
    os.makedirs(output_dir, exist_ok=True)
    if save_translators:
        if args.translators_path is not None:
            os.makedirs(os.path.dirname(args.translators_path), exist_ok=True)
            torch.save(translators, args.translators_path)
        else:
            os.makedirs(output_dir, exist_ok=True)
            torch.save(translators, os.path.join(output_dir, f"translators.pt"))

    # TODO allow to use either patchscopes or heuristic
    if args.use_patchscopes:
        logger.info("Running patchscopes on new words...")
        patchscopes_retriever, patchscopes_results = prepare_patchscopes_retriever(args, model, base_tokenizer)

        vocab_modifier = DetokenizationVocabularyExpander(
            model, tokenizer,
            patchscopes_retriever, patchscopes_results,
            translators,
            args.detokenization_decision_rule,
            args.detokenization_decision_rule_E,
            args.detokenization_max_valid_layer,
            add_to_core_vocab=args.add_new_words_to_core_vocab,
            add_space_before_lowercase_word=args.add_space_before_lowercase_words,
        )
    else:
        vocab_modifier = HeuristicDetokenizationVocabularyExpander(
            model, tokenizer,
            translators,
            args.detokenization_layer,
            args.detokenization_layer_embedding,
            add_to_core_vocab=args.add_new_words_to_core_vocab,
            add_space_before_lowercase_word=args.add_space_before_lowercase_words,
        )

    # load / train calibrators
    if args.calibrate_new_entries:
        if args.overwrite_calibration:
            calibrators = None
        else:
            calibrators = vocab_modifier.load_calibrators(save_dir=args.calibration_save_dir)
        if calibrators is None:
            logger.info("Fitting calibration on new entries...")
            logger.info("First, adding new words to model vocabulary...")
            model, tokenizer = vocab_modifier.add_words_to_vocab(new_words)

            logger.info("Next, training calibration...")
            calibration_dataset = load_lm_dataset(args.calibration_dataset, language=args.calibration_dataset_language)
            calibration_dataset = calibration_dataset[args.calibration_dataset_split]
            calibrators = vocab_modifier.train_calibrators(calibration_dataset, save_dir=args.calibration_save_dir, overwrite_cache=args.overwrite_calibration, max_samples=args.calibration_max_samples, lr=args.calibration_lr, lr_schedule=args.calibration_lr_schedule, num_epochs=args.calibration_num_epochs, batch_size=args.calibration_batch_size, max_length=args.eval_max_length, n_warmup_steps=args.calibration_n_warmup_steps, clip_grad_norm=args.calibration_clip_grad_norm, target_loss_weight=args.calibration_target_loss_weight, subsequent_loss_weight=args.calibration_subsequent_loss_weight, mixed_precision=mixed_precision)
        else:
            logger.info(f"Loaded calibrators from: {args.calibration_save_dir}")
        vocab_modifier.set_calibrators(calibrators)

        if args.run_text_generation_test:
            model, tokenizer = vocab_modifier.undo_vocabulary_changes()
            logger.info("Testing model generates sane text: before any changes...")
            test_text_generation(model, tokenizer, prompt="Once upon a time")
            test_text_generation(model, tokenizer, prompt="היה היה פעם")
            logger.info("With added words, before calibration...")
            model, tokenizer = vocab_modifier.add_words_to_vocab(new_words)
            test_text_generation(model, tokenizer, prompt="Once upon a time")
            test_text_generation(model, tokenizer, prompt="היה היה פעם")
            vocab_modifier.calibrators['new_tokens_end'] = max(vocab_modifier.new_token_ids) + 1
            model = vocab_modifier.apply_calibrators_to_new_entries()
            logger.info("And after calibration...")
            test_text_generation(model, tokenizer, prompt="Once upon a time")
            test_text_generation(model, tokenizer, prompt="היה היה פעם")
            test_text_generation(model, tokenizer, prompt="היה הייתה")

            import pdb; pdb.set_trace()

        # when estimating performance, we will add 1 new token at a time
        calibrators["new_tokens_end"] = calibrators["new_tokens_start"] + 1
        model, tokenizer = vocab_modifier.undo_vocabulary_changes()

    gc.collect()
    torch.cuda.empty_cache()

    eval_dataset = load_lm_dataset(args.eval_dataset, language=args.eval_dataset_language)
    eval_dataset = eval_dataset[args.eval_dataset_split]

    # set containers to hold metrics per word
    background_metrics = defaultdict(dict)
    target_metrics = defaultdict(dict)
    overall_metrics = defaultdict(dict)

    for new_word in tqdm(new_words, total=len(new_words), desc="Estimating vocabulary expansion performance on new words...", unit="word"):
        # add word to vocabulary
        word_added = vocab_modifier.add_word_to_vocab(new_word, finalize=True)
        if not word_added:
            continue
        if args.calibrate_new_entries:
            vocab_modifier.apply_calibrators_to_new_entries()

        model, tokenizer = vocab_modifier.get_model_and_tokenizer()

        # get evaluation data for word
        new_token_id = vocab_modifier.new_token_ids[0]
        curr_eval_dataset = eval_dataset
        curr_eval_dataset = curr_eval_dataset.filter(lambda x: new_word in x[args.eval_dataset_text_col])
        if args.eval_max_samples is not None:
            eval_idx = range(min(10*args.eval_max_samples, len(curr_eval_dataset)))
            if args.eval_shuffle_samples:
                eval_idx = np.random.choice(len(curr_eval_dataset), min(10*args.eval_max_samples, len(curr_eval_dataset)))
            curr_eval_dataset = curr_eval_dataset.select(eval_idx)
        curr_eval_dataset = tokenize_and_prepare_dataset(curr_eval_dataset, tokenizer, accelerator, max_length=args.eval_max_length, text_col_name=args.eval_dataset_text_col,)
        curr_eval_dataset = curr_eval_dataset.filter(lambda x: new_token_id in x['input_ids'])

        # compute metrics
        if len(curr_eval_dataset) > 0:
            new_tokens_to_replaced_token_seqs = vocab_modifier.get_new_tokens_to_replaced_token_seqs_map()
            seq_lens = {len(v) for v in new_tokens_to_replaced_token_seqs.values()}
            replaced_token_seqs_by_len = {curr_seq_len: [seq for seq in new_tokens_to_replaced_token_seqs.values() if len(seq) == curr_seq_len] for curr_seq_len in seq_lens}
            new_token_to_original_first_token = {k: v[0] for k, v in new_tokens_to_replaced_token_seqs.items()}
            background_metrics[new_word], target_metrics[new_word], overall_metrics[new_word] = eval_next_word_prediction(
                model, tokenizer, curr_eval_dataset, accelerator,
                batch_size=args.eval_batch_size, new_token_ids=vocab_modifier.new_token_ids,
                replaced_token_seqs_by_len=replaced_token_seqs_by_len,
                new_token_to_original_first_token=new_token_to_original_first_token,
                max_length=args.eval_max_length,
                eval_max_samples=args.eval_max_samples,
                eval_shuffle_samples=args.eval_shuffle_samples,
                drop_last=False,
                reduction="mean",
            )
        # reset model and tokenizer
        model, tokenizer = vocab_modifier.undo_vocabulary_changes()

    if args.use_patchscopes:
        updated_patchscopes_results = vocab_modifier.get_patchscopes_results()
        if patchscopes_results is None or len(updated_patchscopes_results) > len(patchscopes_results):
            logger.info("Saving updated patchscopes cache to file...")
            patchscopes_results = updated_patchscopes_results
            if args.patchscopes_results_cache is not None:
                try:
                    os.makedirs(os.path.dirname(args.patchscopes_results_cache), exist_ok=True)
                    patchscopes_results.to_parquet(args.patchscopes_results_cache)
                except:
                    patchscopes_results.to_parquet(
                        os.path.join(output_dir, "patchscopes_results.parquet"))
            else:
                patchscopes_results.to_parquet(
                    os.path.join(output_dir, "patchscopes_results.parquet"))

    pd.DataFrame(background_metrics).T.to_csv(os.path.join(output_dir, "background_metrics.csv"))
    pd.DataFrame(target_metrics).T.to_csv(os.path.join(output_dir, "target_metrics.csv"))
    pd.DataFrame(overall_metrics).T.to_csv(os.path.join(output_dir, "overall_metrics.csv"))
    config = vars(args)
    with open(os.path.join(output_dir, f"config.json"), "w") as config_file:
        json.dump(config, config_file, indent=4)

    logger.info(f"Results saved to: {output_dir}")

    return


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate word-level vocabulary expansion success.")
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--output_dir", type=str, default="./experiments/")

    parser.add_argument("--run_text_generation_test", action="store_true", default=False)

    parser.add_argument("--use_patchscopes", action="store_true", default=False)
    parser.add_argument("--overwrite_calibration", action="store_true", default=False)
    parser.add_argument("--add_new_words_to_core_vocab", action="store_true", default=False)
    parser.add_argument("--add_space_before_lowercase_words", action="store_true", default=False)
    parser.add_argument("--detokenization_layer", type=int, default=4)
    parser.add_argument("--detokenization_layer_embedding", type=int, default=4)
    parser.add_argument("--detokenization_decision_rule", type=str, default="first_id_layer")
    parser.add_argument("--detokenization_decision_rule_E", type=str, default=None)
    parser.add_argument("--detokenization_max_valid_layer", type=int, default=None)
    parser.add_argument("--early_exit_layer", type=int, default=None)
    parser.add_argument("--extraction_prompt", type=str, default="X")
    parser.add_argument("--prompt_target", type=str, default="X")
    parser.add_argument("--extraction_batch_size", type=int, default=128)
    parser.add_argument("--patchscopes_prompt", type=str, default="X, X, X, X,")
    parser.add_argument("--patchscopes_results_cache", type=str, default=None)
    parser.add_argument("--patchscopes_generate_n_tokens", type=int, default=20)
    parser.add_argument("--patchscopes_max_words", type=int, default=None)

    parser.add_argument("--translators_path", type=str, default=None)
    parser.add_argument("--translators_fit_intercept", action="store_true", default=False)
    parser.add_argument("--translators_do_residual", action="store_true", default=False)
    parser.add_argument("--translators_learn_mlp", action="store_true", default=False)
    parser.add_argument("--translators_use_procrustes", action="store_true", default=False)
    parser.add_argument("--translators_procrustes_normalize", action="store_true", default=False)
    parser.add_argument("--translators_procrustes_layers", nargs="+", type=int, default=None)
    parser.add_argument("--translators_learn_on_space_prefixed_words_only", action="store_true", default=False)
    parser.add_argument("--translators_fit_min_word_len", type=int, default=None)

    parser.add_argument("--calibrate_new_entries", action="store_true", default=False)
    parser.add_argument("--calibration_save_dir", type=str, default=None)
    parser.add_argument("--calibration_dataset", type=str, default=None)
    parser.add_argument("--calibration_dataset_split", type=str, default=None)
    parser.add_argument("--calibration_dataset_language", type=str, default=None)
    parser.add_argument("--calibration_batch_size", type=int, default=4)
    parser.add_argument("--calibration_lr", type=float, default=0.0001)
    parser.add_argument("--calibration_clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--calibration_target_loss_weight", type=float, default=0.15)
    parser.add_argument("--calibration_subsequent_loss_weight", type=float, default=0.15)
    parser.add_argument("--calibration_lr_schedule", type=str, default="linear")
    parser.add_argument("--calibration_n_warmup_steps", type=float, default=0.03)
    parser.add_argument("--calibration_num_epochs", type=int, default=1)
    parser.add_argument("--calibration_max_samples", type=int, default=None)

    parser.add_argument("--eval_dataset", type=str, default="wikitext")
    parser.add_argument("--eval_dataset_language", type=str, default=None)
    parser.add_argument("--eval_max_samples", type=int, default=None)
    parser.add_argument("--eval_shuffle_samples", action="store_true", default=True)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--eval_max_length", type=int, default=256)
    parser.add_argument("--eval_dataset_split", type=str, default="test")
    parser.add_argument("--eval_dataset_text_col", type=str, default="text")

    parser.add_argument("--words_list", type=str, default=None)
    parser.add_argument("--words_list_delimiter", type=str, default=None)
    parser.add_argument("--max_words", type=int, default=None)

    args = parser.parse_args()

    assert  args.words_list is not None, \
        "Please pass the path to a file containing a list of words (--words_list)"

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
