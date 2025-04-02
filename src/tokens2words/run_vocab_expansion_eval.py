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
import re
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
from .vocab_modifier import PatchscopesLimitedHeuristicDetokenizationVocabularyExpander
from .utils.file_utils import parse_string_list_from_file
from .utils.data_utils import load_lm_dataset, extract_new_words_from_dataset, tokenize_and_prepare_dataset
from .utils.eval_utils import eval_next_word_prediction, count_tokens_in_dataset
from .utils.downstream_utils import evaluate_model as downstream_eval
from .utils.calibration_utils import get_language_tokens

import logging

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_word_filter(args):

    def word_filter(word, token_count):
        is_valid = True
        if args.words_filter_max_n_tokens and not (token_count <= args.words_filter_max_n_tokens):
            is_valid = False
        if args.words_filter_non_en and not all('a' <= char <= 'z' or 'A' <= char <= 'Z' for char in word):
            is_valid = False
        if args.words_filter_numeric and any(char.isdigit() for char in word):
            is_valid = False
        return is_valid

    return word_filter


def prepare_new_words(
        args, tokenizer):

    _word_filter = get_word_filter(args)

    def _get_token_length(word):
        return len(tokenizer.tokenize(word))

    if not args.words_list:
        new_words = list()
    else:
        new_words = parse_string_list_from_file(args.words_list, args.words_list_delimiter)
        new_words = [w for w in new_words if not tokenizer.vocab.get(w, False) and _word_filter(w, _get_token_length(w))]

    if args.words_dataset:
        words_dataset = load_lm_dataset(args.words_dataset, language=args.words_dataset_language)
        if args.words_dataset_overlap_split is not None:
            words_overlap_dataset = words_dataset[args.words_dataset_overlap_split]
            new_words_from_overlap_data, new_words_from_doverlap_data_freqs = extract_new_words_from_dataset(
                words_overlap_dataset, tokenizer, args.words_dataset_text_col, filter_func=_word_filter)

        words_dataset = words_dataset[args.words_dataset_split]

        new_words_from_data, new_words_from_data_freqs = extract_new_words_from_dataset(
            words_dataset, tokenizer, args.words_dataset_text_col, filter_func=_word_filter)

        if args.words_dataset_overlap_split is not None:
            new_words_from_data = list(set(new_words_from_data).intersection(new_words_from_overlap_data))

        if args.words_filter_min_freq is not None:
            new_words_from_data = [word for word in new_words_from_data if new_words_from_data_freqs[word] >= args.words_filter_min_freq]

        # Estimate new tokens rates
        topline_tokenizer = deepcopy(tokenizer)
        n_new_words = topline_tokenizer.add_tokens(new_words_from_data)
        baseline_vocab_total_tokens = count_tokens_in_dataset(words_dataset, tokenizer, args.words_dataset_text_col)
        max_vocab_total_tokens = count_tokens_in_dataset(words_dataset, topline_tokenizer, args.words_dataset_text_col)
        logger.info(f"Baseline tokenizer - total tokens: {baseline_vocab_total_tokens}")
        logger.info(f"Topline expanded tokenizer - total tokens: {max_vocab_total_tokens} - new words: {n_new_words}")

    new_words += new_words_from_data

    if args.max_words is not None:
        new_words = new_words[:args.max_words]

    baseline_tokenization = {w: tokenizer.encode(w, add_special_tokens=False, return_tensors="pt")[0]
                             for w in new_words}

    return new_words, baseline_tokenization


def prepare_patchscopes_retriever(args, model, tokenizer):
    patchscopes_retriever = PatchscopesRetriever(
        model, tokenizer,
        args.extraction_prompt,
        args.patchscopes_prompt,
        args.prompt_target,
        num_tokens_to_generate=args.patchscopes_generate_n_tokens,
    )

    patchscopes_results = None
    try:
        if args.patchscopes_results_cache is not None:
            patchscopes_results = pd.read_parquet(args.patchscopes_results_cache)
    except:
        pass

    return patchscopes_retriever, patchscopes_results


def prepare_translators(args, model, tokenizer):
    save_translators = True
    translators = None
    if args.translators_path:
        try:
            translators = torch.load(args.translators_path, map_location=torch.device('cpu'), weights_only=False)
            save_translators = False
            return translators, save_translators
        except:
            pass

    if args.translators_use_procrustes:
        translators = ProcrustesRepresentationTranslators()
        translators.fit_on_tokens(
            model, tokenizer,
            prompt=args.extraction_prompt,
            prompt_target=args.prompt_target,
            translation_layers=args.translators_procrustes_layers,
            normalize=args.translators_procrustes_normalize,
            normalize_embeddings=args.translators_procrustes_normalize_embeddings,
            post_normalize_mode=args.translators_post_normalize_mode,
            batch_size=args.extraction_batch_size,
            layer_batch_size=args.translators_layer_batch_size,
            space_prefixed_only=args.translators_learn_on_space_prefixed_words_only,
            min_word_len=args.translators_fit_min_word_len,
        )
    elif args.translators_learn_mlp:
        translators = MLPRepresentationTranslators()
        translators.fit_on_tokens(
            model, tokenizer,
            prompt=args.extraction_prompt,
            prompt_target=args.prompt_target,
            batch_size=args.extraction_batch_size,
        )
    elif args.translators_learn_linear:
        translators = LinearRegressionRepresentationTranslators()
        translators.fit_on_tokens(
            model, tokenizer,
            prompt=args.extraction_prompt,
            prompt_target=args.prompt_target,
            translation_layers=args.translators_procrustes_layers,
            normalize=args.translators_procrustes_normalize,
            normalize_embeddings=args.translators_procrustes_normalize_embeddings,
            post_normalize_mode=args.translators_post_normalize_mode,
            batch_size=args.extraction_batch_size,
            layer_batch_size=args.translators_layer_batch_size,
            space_prefixed_only=args.translators_learn_on_space_prefixed_words_only,
            min_word_len=args.translators_fit_min_word_len,
        )

    elif args.translators_use_rms:
        translators = RMSRepresentationTranslators()

    if translators is None:
        save_translators = False
    return translators, save_translators


def test_text_generation(model, tokenizer, prompt="Once upon a time", num_tokens=20):
    def _generate_greedy(input_ids):
        # Greedy decoding
        greedy_output = model.generate(input_ids, max_length=input_ids.shape[1] + num_tokens, do_sample=False,
                                       temperature=None, top_p=None)
        greedy_decoded = tokenizer.decode(greedy_output[0], skip_special_tokens=True)
        logger.info(f"Greedy Decoding --- {greedy_decoded}\nToken IDs: {greedy_output[0].tolist()}")

    def _generate_top_p(input_ids):
        # Top-p sampling with temperature
        top_p_output = model.generate(input_ids, max_length=input_ids.shape[1] + num_tokens, do_sample=True, top_p=0.9,
                                      temperature=0.7)
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
    model = AutoModelForCausalLM.from_pretrained(args.model_name,
                                                 torch_dtype=torch.bfloat16 if mixed_precision == "bf16" else torch.float16)
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

    if args.use_patchscopes:
        logger.info("Running patchscopes on new words...")
        patchscopes_retriever, patchscopes_results = prepare_patchscopes_retriever(args, model, base_tokenizer)

        vocab_modifier = DetokenizationVocabularyExpander(
            model, tokenizer,
            patchscopes_retriever, patchscopes_results,
            args.patchscopes_force_starts_with_word,
            translators,
            args.detokenization_decision_rule,
            args.detokenization_decision_rule_E,
            args.detokenization_max_valid_layer,
            add_to_core_vocab=args.add_new_words_to_core_vocab,
            add_space_before_lowercase_word=args.add_space_before_lowercase_words,
        )
    elif args.use_heuristic and args.heuristic_use_patchscopes_filter:
        logger.info("Running patchscopes on new words...")
        patchscopes_retriever, patchscopes_results = prepare_patchscopes_retriever(args, model, base_tokenizer)

        vocab_modifier = PatchscopesLimitedHeuristicDetokenizationVocabularyExpander(
            model, tokenizer,
            patchscopes_retriever, patchscopes_results,
            args.patchscopes_force_starts_with_word,
            translators,
            args.detokenization_layer,
            args.detokenization_layer_embedding,
            add_to_core_vocab=args.add_new_words_to_core_vocab,
            add_space_before_lowercase_word=args.add_space_before_lowercase_words,
            use_mean_hidden_states=args.heuristic_use_mean_hidden_states,
            use_mean_embeddings=args.heuristic_use_mean_embeddings,
            extract_hidden_states_as_dict=True,
        )
    elif args.use_heuristic:
        vocab_modifier = HeuristicDetokenizationVocabularyExpander(
            model, tokenizer,
            translators,
            args.detokenization_layer,
            args.detokenization_layer_embedding,
            add_to_core_vocab=args.add_new_words_to_core_vocab,
            add_space_before_lowercase_word=args.add_space_before_lowercase_words,
            use_mean_hidden_states=args.heuristic_use_mean_hidden_states,
            use_mean_embeddings=args.heuristic_use_mean_embeddings,
            extract_hidden_states_as_dict=True,
        )
    else:
        raise ValueError("To add new words to the vocabulary you need to set either --use_patchscopes or --use_heuristic. These define which layer to extract representations from.")

    if args.run_text_generation_test:
        logger.info("Testing model generates sane text: before any changes...")
        test_text_generation(model, tokenizer, prompt="Once upon a time")

    logger.info("Adding new words to model vocabulary...")
    # if args.use_patchscopes:
    #     model, tokenizer = vocab_modifier.add_words_to_vocab_batch(new_words)
    # else:
    model, tokenizer = vocab_modifier.add_words_to_vocab(new_words)

    if args.use_patchscopes:
        logger.info("Saving updated patchscopes cache to file...")
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

    if args.run_text_generation_test:
        logger.info("Testing model generates sane text: with added words, before calibration...")
        test_text_generation(model, tokenizer, prompt="Once upon a time")

    # compute some metrics
    new_token_ids = deepcopy(vocab_modifier.new_token_ids)
    new_tokens_to_replaced_token_seqs = vocab_modifier.get_new_tokens_to_replaced_token_seqs_map()
    seq_lens = {len(v) for v in new_tokens_to_replaced_token_seqs.values()}
    replaced_token_seqs_by_len = {
        curr_seq_len: [seq for seq in new_tokens_to_replaced_token_seqs.values() if len(seq) == curr_seq_len]
        for curr_seq_len in seq_lens}
    new_token_to_original_first_token = {k: v[0] for k, v in new_tokens_to_replaced_token_seqs.items()}

    other_metrics = dict()
    other_metrics["n_new_words"] = len(vocab_modifier.new_words)
    other_metrics["patchscopes_success_rate"] = len(vocab_modifier.new_words) / (len(vocab_modifier.new_words) + len(vocab_modifier.failed_words))
    other_metrics["n_attempted_words"] = len(new_words)
    # load / train calibrators
    calibrators = None
    if args.calibrate_new_entries:
        if args.calibrate_existing_language_tokens is not None:
            existing_token_ids = get_language_tokens(base_tokenizer, args.calibrate_existing_language_tokens)
            vocab_modifier.set_existing_tokens_to_calibrate(existing_token_ids)

        if args.calibration_similarity_language_tokens is not None:
            similarity_reg_token_ids = \
                get_language_tokens(base_tokenizer, args.calibration_similarity_language_tokens)
        else:
            similarity_reg_token_ids = None

        if not args.overwrite_calibration:
            calibrators = vocab_modifier.load_calibrators(save_dir=args.calibration_save_dir)
        if calibrators is None:
            logger.info("Fitting calibration on new entries...")
            calibration_dataset = load_lm_dataset(args.calibration_dataset, language=args.calibration_dataset_language)
            calibration_dataset = calibration_dataset[args.calibration_dataset_split]
            calibration_model, calibrators = vocab_modifier.train_calibrators(
               calibration_dataset, save_dir=args.calibration_save_dir,
               overwrite_cache=args.overwrite_calibration,
               max_samples=args.calibration_max_samples,
               lr=args.calibration_lr,
               lr_schedule=args.calibration_lr_schedule,
               num_epochs=args.calibration_num_epochs,
               batch_size=args.calibration_batch_size,
               max_length=args.eval_max_length,
               n_warmup_steps=args.calibration_n_warmup_steps,
               clip_grad_norm=args.calibration_clip_grad_norm,
               target_loss_weight=args.calibration_target_loss_weight,
               subsequent_loss_weight=args.calibration_subsequent_loss_weight,
               mixed_precision=mixed_precision,
               learn_gate_activation=args.calibration_learn_gate_activation,
               learn_per_token_bias=args.calibration_learn_per_token_bias,
               new_tokens_to_orig_first_map=new_token_to_original_first_token,
               soft_allow_orig_first_tokens=args.calibration_soft_allow_original_token,
               existing_tokens_for_similarity_loss=similarity_reg_token_ids,
            )

        else:
            logger.info(f"Loaded calibrators from: {args.calibration_save_dir}")

        vocab_modifier.set_calibrators(calibrators)
        model = vocab_modifier.apply_calibrators_to_new_entries()

        if args.run_text_generation_test:
            # vocab_modifier.calibrators['new_tokens_end'] = max(vocab_modifier.new_token_ids) + 1
            logger.info("And after calibration...")
            test_text_generation(model, tokenizer, prompt="Once upon a time")

    eval_dataset = load_lm_dataset(args.eval_dataset, language=args.eval_dataset_language)
    eval_dataset = eval_dataset[args.eval_dataset_split]

    # set containers to hold metrics per word
    background_metrics = defaultdict(dict)
    target_metrics = defaultdict(dict)
    overall_metrics = defaultdict(dict)
    downstream_metrics = defaultdict(dict)

    model, tokenizer = vocab_modifier.get_model_and_tokenizer()

    # tokenize eval dataset
    lm_dataset = tokenize_and_prepare_dataset(
        eval_dataset, tokenizer, accelerator, max_length=args.eval_max_length, text_col_name=args.eval_dataset_text_col, )

    baseline_lm_dataset = tokenize_and_prepare_dataset(
        eval_dataset, base_tokenizer, accelerator, max_length=args.eval_max_length, text_col_name=args.eval_dataset_text_col, )

    baseline_vocab_total_tokens = count_tokens_in_dataset(eval_dataset, base_tokenizer, args.eval_dataset_text_col)
    new_vocab_total_tokens = count_tokens_in_dataset(eval_dataset, tokenizer, args.eval_dataset_text_col)
    logger.info(f"Baseline tokenizer - total tokens: {baseline_vocab_total_tokens}")
    logger.info(f"Expanded tokenizer - total tokens: {new_vocab_total_tokens}")
    other_metrics["total_tokens"] = {"expanded": new_vocab_total_tokens, "baseline": baseline_vocab_total_tokens}
    other_metrics["tokens_saved"] = 1 - other_metrics["total_tokens"]["expanded"] / other_metrics["total_tokens"]["baseline"]

    logger.info(f"==== Some numbers ====\n{other_metrics}")

    # compute metrics - expanded
    background_metrics["expanded"], target_metrics["expanded"], overall_metrics["expanded"] = \
        eval_next_word_prediction(
            model, tokenizer, lm_dataset, accelerator,
            batch_size=args.eval_batch_size, new_token_ids=new_token_ids,
            replaced_token_seqs_by_len=replaced_token_seqs_by_len,
            new_token_to_original_first_token=new_token_to_original_first_token,
            max_length=args.eval_max_length,
            eval_max_samples=args.eval_max_samples,
            eval_shuffle_samples=args.eval_shuffle_samples,
            drop_last=False,
            reduction="mean",
    )

    if args.eval_downstream:
        downstream_log_dir = os.path.join(output_dir, "downstream_preds", "expanded") if args.save_downstream_outputs else None
        downstream_metrics["expanded"] = downstream_eval(model, tokenizer, limited=True, output_path=downstream_log_dir)

    # compute metrics - baseline
    model, tokenizer = vocab_modifier.undo_vocabulary_changes()
    background_metrics["baseline"], target_metrics["baseline"], overall_metrics["baseline"] = \
        eval_next_word_prediction(
            model, base_tokenizer, baseline_lm_dataset, accelerator,
            batch_size=args.eval_batch_size, new_token_ids=new_token_ids,
            replaced_token_seqs_by_len=replaced_token_seqs_by_len,
            new_token_to_original_first_token=None,
            max_length=args.eval_max_length,
            eval_max_samples=args.eval_max_samples,
            eval_shuffle_samples=args.eval_shuffle_samples,
            drop_last=False,
            reduction="mean",
        )

    if args.eval_baseline_downstream:
        downstream_log_dir = os.path.join(output_dir, "downstream_preds", "baseline") if args.save_downstream_outputs else None
        downstream_metrics["baseline"] = downstream_eval(model, base_tokenizer, limited=True, output_path=downstream_log_dir)

    background_df = pd.DataFrame.from_dict(background_metrics)
    target_df = pd.DataFrame.from_dict(target_metrics)
    overall_df = pd.DataFrame.from_dict(overall_metrics)
    other_df = pd.DataFrame.from_dict(other_metrics)
    logger.info(f"==== Background Results ====\n{tabulate(background_df, headers='keys', tablefmt='psql')}")
    logger.info(f"==== Target Results ====\n{tabulate(target_df, headers='keys', tablefmt='psql')}")
    logger.info(f"==== Overall Results ====\n{tabulate(overall_df, headers='keys', tablefmt='psql')}")
    logger.info(f"==== Other Results ====\n{other_metrics}")

    background_df.to_json(os.path.join(output_dir, "metrics_background.json"), indent=4)
    target_df.to_json(os.path.join(output_dir, "metrics_target.json"), indent=4)
    overall_df.to_json(os.path.join(output_dir, "metrics_overall.json"), indent=4)
    other_df.to_json(os.path.join(output_dir, "metrics_other.json"), indent=4)

    if args.eval_downstream or args.eval_baseline_downstream:
        final_downstream_metrics = defaultdict(dict)
        for eval_type, eval_results in downstream_metrics.items():
            for task_name, task_results in eval_results.items():
                final_downstream_metrics[task_name][eval_type] = task_results
        with open(os.path.join(output_dir, f"metrics_downstream.json"), "w") as f:
            json.dump(final_downstream_metrics, f, indent=4, ensure_ascii=False)

    if args.eval_downstream:
        logger.info(f"==== Downstream Results - Expanded ====\n{tabulate(pd.DataFrame.from_dict(downstream_metrics['expanded']), headers='keys', tablefmt='psql')}")

    if args.eval_baseline_downstream:
        logger.info(f"==== Downstream Results - Baseline ====\n{tabulate(pd.DataFrame.from_dict(downstream_metrics['baseline']), headers='keys', tablefmt='psql')}")

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
    parser.add_argument("--use_heuristic", action="store_true", default=False)
    parser.add_argument("--overwrite_calibration", action="store_true", default=False)
    parser.add_argument("--add_new_words_to_core_vocab", action="store_true", default=False)
    parser.add_argument("--add_space_before_lowercase_words", action="store_true", default=False)
    parser.add_argument("--detokenization_layer", type=int, default=4)
    parser.add_argument("--detokenization_layer_embedding", type=int, default=4)
    parser.add_argument("--detokenization_decision_rule", type=str, default="first_id_layer")
    parser.add_argument("--detokenization_decision_rule_E", type=str, default=None)
    parser.add_argument("--detokenization_max_valid_layer", type=int, default=None)
    parser.add_argument("--heuristic_use_mean_hidden_states", action="store_true", default=False)
    parser.add_argument("--heuristic_use_mean_embeddings", action="store_true", default=False)
    parser.add_argument("--heuristic_use_patchscopes_filter", action="store_true", default=False)
    parser.add_argument("--early_exit_layer", type=int, default=None)
    parser.add_argument("--extraction_prompt", type=str, default="X")
    parser.add_argument("--prompt_target", type=str, default="X")
    parser.add_argument("--extraction_batch_size", type=int, default=128)
    parser.add_argument("--patchscopes_prompt", type=str, default="X, X, X, X,")
    parser.add_argument("--patchscopes_results_cache", type=str, default=None)
    parser.add_argument("--patchscopes_generate_n_tokens", type=int, default=20)
    parser.add_argument("--patchscopes_max_words", type=int, default=None)
    parser.add_argument("--patchscopes_force_starts_with_word", action="store_true", default=False)

    parser.add_argument("--translators_path", type=str, default=None)
    parser.add_argument("--translators_fit_intercept", action="store_true", default=False)
    parser.add_argument("--translators_do_residual", action="store_true", default=False)
    parser.add_argument("--translators_learn_mlp", action="store_true", default=False)
    parser.add_argument("--translators_learn_linear", action="store_true", default=False)
    parser.add_argument("--translators_use_procrustes", action="store_true", default=False)
    parser.add_argument("--translators_post_normalize_mode", type=str, default=None)
    parser.add_argument("--translators_use_rms", action="store_true", default=False)
    parser.add_argument("--translators_procrustes_normalize", action="store_true", default=False)
    parser.add_argument("--translators_procrustes_normalize_embeddings", action="store_true", default=False)
    parser.add_argument("--translators_procrustes_layers", nargs="+", type=int, default=None)
    parser.add_argument("--translators_learn_on_space_prefixed_words_only", action="store_true", default=False)
    parser.add_argument("--translators_fit_min_word_len", type=int, default=None)
    parser.add_argument("--translators_layer_batch_size", type=int, default=2)

    parser.add_argument("--calibrate_new_entries", action="store_true", default=False)
    parser.add_argument("--calibrate_existing_language_tokens", type=str, default=None)
    parser.add_argument("--calibration_similarity_language_tokens", type=str, default=None)
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
    parser.add_argument("--calibration_learn_gate_activation", action="store_true", default=False)
    parser.add_argument("--calibration_learn_per_token_bias", action="store_true", default=False)
    parser.add_argument("--calibration_soft_allow_original_token", action="store_true", default=False)

    parser.add_argument("--eval_dataset", type=str, default="wikitext")
    parser.add_argument("--eval_dataset_language", type=str, default=None)
    parser.add_argument("--eval_max_samples", type=int, default=None)
    parser.add_argument("--eval_shuffle_samples", action="store_true", default=True)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--eval_max_length", type=int, default=256)
    parser.add_argument("--eval_dataset_split", type=str, default="test")
    parser.add_argument("--eval_dataset_text_col", type=str, default="text")

    parser.add_argument("--eval_downstream", action="store_true", default=False)
    parser.add_argument("--eval_baseline_downstream", action="store_true", default=False)
    parser.add_argument("--save_downstream_outputs", action="store_true", default=True)

    parser.add_argument("--words_dataset", type=str, default=None)
    parser.add_argument("--words_dataset_language", type=str, default=None)
    parser.add_argument("--words_dataset_split", type=str, default="test")
    parser.add_argument("--words_dataset_overlap_split", type=str, default=None)
    parser.add_argument("--words_dataset_text_col", type=str, default="text")

    parser.add_argument("--words_filter_min_freq", type=int, default=None)
    parser.add_argument("--words_filter_max_n_tokens", type=int, default=5)
    parser.add_argument("--words_filter_non_en", action="store_true", default=False)
    parser.add_argument("--words_filter_numeric", action="store_true", default=True)

    parser.add_argument("--words_list", type=str, default=None)
    parser.add_argument("--words_list_delimiter", type=str, default=None)
    parser.add_argument("--preprocess_remove_noisy_chars", action="store_true", default=True)
    parser.add_argument("--space_prefix", type=str, default="Ġ")
    parser.add_argument("--max_words", type=int, default=None)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
