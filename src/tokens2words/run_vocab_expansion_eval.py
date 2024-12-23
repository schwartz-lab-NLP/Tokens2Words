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
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from tokenizers import AddedToken
from accelerate import Accelerator
from accelerate.utils import set_seed
from collections import defaultdict
from torch.utils.data import DataLoader
from copy import deepcopy
import types

from .word_retriever import PatchscopesRetriever
from .representation_translator import LinearRepresentationTranslators, ProcrustesRepresentationTranslators, MLPRepresentationTranslators
from .vocab_modifier import DetokenizationVocabularyExpander
from .utils.file_utils import parse_string_list_from_file
from .utils.data_utils import load_lm_dataset, extract_new_words_from_dataset, get_group_texts_func, get_tokenize_func
from .utils.eval_utils import get_last_zero_in_every_seq_mask, get_first_zero_in_every_seq_mask, compute_topk_token_rank
from .utils.eval_utils import count_tokens_in_dataset

import logging

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def eval_lm(
        model, accelerator, tokenizer, baseline_tokenizer, dataset,
        batch_size: int = 4,
        top_ks=[5, 10],
        new_token_ids=None, replaced_token_seqs_by_len=None,
        new_token_to_original_first_token=None, new_token_to_original_token_len=None,
        text_col_name: str = "text",
        max_length: int = 256,
        eval_max_samples: int = None,
):
    model.eval()
    if tokenizer.bos_token is not None and max_length:
        add_start_token = True
        # leave room for <BOS> token to be added:
        max_tokenized_len = max_length - 1
    else:
        add_start_token = False
        max_tokenized_len = max_length

    tokenize_function = get_tokenize_func(tokenizer, text_col_name)
    baseline_tokenize_function = get_tokenize_func(baseline_tokenizer, text_col_name)

    column_names = dataset.column_names

    with accelerator.main_process_first():
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )
        group_texts = get_group_texts_func(block_size=max_tokenized_len)
        lm_dataset = tokenized_dataset.map(
            group_texts,
            batched=True,
        )

        baseline_tokenized_dataset = dataset.map(
            baseline_tokenize_function,
            batched=True,
            remove_columns=column_names,
            load_from_cache_file=False,
            desc="Running baseline tokenizer on dataset",
        )
        baseline_lm_dataset = baseline_tokenized_dataset.map(
            group_texts,
            batched=True,
        )

        baseline_vocab_total_tokens = count_tokens_in_dataset(dataset, baseline_tokenizer, text_col_name)
        new_vocab_total_tokens = count_tokens_in_dataset(dataset, tokenizer, text_col_name)

    logger.info(f"Baseline tokenizer - total tokens: {baseline_vocab_total_tokens}")
    logger.info(f"Expanded tokenizer - total tokens: {new_vocab_total_tokens}")

    if eval_max_samples:
        lm_dataset = lm_dataset.select(range(eval_max_samples))
        baseline_lm_dataset = baseline_lm_dataset.select(range(eval_max_samples))

    data_collator = default_data_collator

    # Create data loaders
    expanded_vocab_dataloader = DataLoader(
        lm_dataset, collate_fn=data_collator, batch_size=batch_size, drop_last=True, shuffle=False,
    )
    baseline_vocab_dataloader = DataLoader(
        baseline_lm_dataset, collate_fn=data_collator, batch_size=batch_size, drop_last=True, shuffle=False,
    )
    baseline_vocab_dataloader = accelerator.prepare(baseline_vocab_dataloader)
    model.eval()

    if new_token_ids is not None:
        new_token_ids = torch.tensor(new_token_ids).to(model.device)
    if replaced_token_seqs_by_len is not None:
        replaced_token_seqs_by_len = {token_length: torch.tensor(skip_token_seqs).to(model.device) for token_length, skip_token_seqs in replaced_token_seqs_by_len.items() if len(skip_token_seqs) > 0}
    if new_token_to_original_first_token is not None:
        # Convert the mapping into a tensor for efficient indexing, create a mapping tensor that defaults to identity
        new_token_to_orig_first_mapping_tensor = torch.arange(len(tokenizer), device=model.device)
        new_token_to_orig_first_mapping_tensor[torch.tensor(list(new_token_to_original_first_token.keys()), device=model.device)] = \
            torch.tensor(list(new_token_to_original_first_token.values()), device=model.device)

    target_metrics = {
        "baseline": defaultdict(list),
        "expanded_E": defaultdict(list),
        "fully_expanded": defaultdict(list),
        # "clipped_expanded": defaultdict(list),
    }

    background_metrics = deepcopy(target_metrics)
    overall_metrics = deepcopy(target_metrics)
    other_metrics = dict()

    # for perplexity
    ce_loss_func = nn.CrossEntropyLoss(reduction="none")

    # TODO consider not aggregating results here, to enable metrics for specific words
    def _compute_metrics(
            model, tokenizer, logits, labels, attention_mask, original_labels=None,
            compute_target_metrics=True, compute_subsequent_metrics=True, compute_perplexity=False,
            return_successful_targets=False,
            debug=False):
        target_results = dict()  # will hold metrics for all the new words we add or their original tokenization
        background_results = dict()  # will hold metrics for all background tokens, i.e., not the ones we add or replace
        overall_results = dict()  # will hold metrics for all tokens
        successful_targets = None  # will hold list of target tokens successfully predicted
        if compute_subsequent_metrics:
            # prepare labels and attentions masks for computing metrics only for the 1st tokens following the new words
            subsequent_labels = labels[:,  1:]
            subsequent_attention_mask = get_last_zero_in_every_seq_mask(attention_mask[..., :-1].contiguous())
            subsequent_attention_mask_bool = subsequent_attention_mask == 1
        attention_mask_bool = attention_mask == 1
        overall_mask_bool = attention_mask_bool

        if compute_target_metrics:
            target_mask = get_first_zero_in_every_seq_mask(attention_mask)
            target_mask_bool = target_mask == 1
            overall_mask_bool = attention_mask_bool | target_mask_bool

        if compute_perplexity:
            background_results["perplexity"] = torch.exp(
                (ce_loss_func(logits.transpose(1, 2), labels) * attention_mask).sum(1)
                / attention_mask.sum(1)
            ).mean().detach().cpu().numpy()

        top1 = logits.argmax(dim=-1)

        if compute_target_metrics:
            target_results["top1_acc"] = ((labels == top1)[target_mask_bool]).detach().cpu().numpy()
            if original_labels is not None:
                target_results["sum_top1_acc"] = (
                    ((original_labels == top1) | (labels == top1))[target_mask_bool]).detach().cpu().numpy()
            if return_successful_targets:
                successful_targets = (labels[(labels == top1) & target_mask_bool]).detach().cpu().numpy()

        background_results["top1_acc"] = ((
                             labels == top1)[attention_mask_bool]).detach().cpu().numpy()
        if compute_subsequent_metrics:
            background_results["subsequent_top1_acc"] = ((subsequent_labels == top1[:, 1:])[subsequent_attention_mask_bool]).detach().cpu().numpy()

        overall_results["top1_acc"] = ((labels == top1))[overall_mask_bool].detach().cpu().numpy()
        if original_labels is not None:
            overall_results["sum_top1_acc"] = (
                ((original_labels == top1) | (labels == top1)))[overall_mask_bool].detach().cpu().numpy()

        for top_k in top_ks:
            topk = logits.topk(top_k, dim=-1).indices
            background_results[f"top{top_k}_acc"] = ((topk == labels.unsqueeze(-1)).any(
                dim=-1)[attention_mask_bool]).detach().cpu().numpy()
            if compute_subsequent_metrics:
                background_results[f"subsequent_top{top_k}_acc"] = ((topk[:, 1:] == subsequent_labels.unsqueeze(-1)).any(
                    dim=-1)[subsequent_attention_mask_bool]).detach().cpu().numpy()
            if compute_target_metrics:
                target_results[f"top{top_k}_acc"] = ((topk == labels.unsqueeze(-1)).any(
                    dim=-1)[target_mask_bool]).detach().cpu().numpy()
                if original_labels is not None:
                    # target_results[f"original_top{top_k}_acc"] = ((topk == original_labels.unsqueeze(-1)).any(
                    #     dim=-1)[target_mask_bool]).detach().cpu().numpy()
                    target_results[f"sum_top{top_k}_acc"] = (
                        ((topk == original_labels.unsqueeze(-1)) | (topk == labels.unsqueeze(-1))).any(
                        dim=-1)[target_mask_bool]).detach().cpu().numpy()

            overall_results[f"top{top_k}_acc"] = ((topk == labels.unsqueeze(-1))[overall_mask_bool].any(
                dim=-1)).detach().cpu().numpy()
            if original_labels is not None:
                overall_results[f"sum_top{top_k}_acc"] = (
                    ((topk == original_labels.unsqueeze(-1)) | (topk == labels.unsqueeze(-1)))[overall_mask_bool].any(
                    dim=-1)).detach().cpu().numpy()

        rank = compute_topk_token_rank(logits, labels)
        background_results["mrr"] = ((1 / rank)[attention_mask_bool]).detach().cpu().numpy()
        if compute_subsequent_metrics:
            background_results["subsequent_mrr"] = ((1 / rank[:, 1:])[subsequent_attention_mask_bool]).detach().cpu().numpy()

        if compute_target_metrics:
            target_results["mrr"] = ((1 / rank)[target_mask_bool]).detach().cpu().numpy()
            if original_labels is not None:
                orig_rank = compute_topk_token_rank(logits, original_labels)
                # target_results["original_mrr"] = ((1 / orig_rank)[target_mask_bool]).detach().cpu().numpy()
                target_results["sum_mrr"] = ((1 / torch.minimum(orig_rank, rank))[target_mask_bool]).detach().cpu().numpy()

        overall_results["mrr"] = ((1 / rank)[overall_mask_bool]).detach().cpu().numpy()
        if original_labels is not None:
            orig_rank = compute_topk_token_rank(logits, original_labels)
            overall_results["sum_mrr"] = ((1 / torch.minimum(orig_rank, rank))[overall_mask_bool]).detach().cpu().numpy()

        if debug:
            import pdb; pdb.set_trace()
        del rank
        return background_results, target_results, overall_results, successful_targets

    def _add_start_token(batch):
        bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * batch["input_ids"].size(dim=0)).to(batch["input_ids"].device)
        batch["input_ids"] = torch.cat([bos_tokens_tensor, batch["input_ids"]], dim=1)
        batch["attention_mask"] = torch.cat(
            [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(batch["attention_mask"].device), batch["attention_mask"]], dim=1)
        return batch

    def _ignore_new_words_in_attention_mask(shift_attention_mask_batch, shift_labels):
        # Ignore token_ids of new vocabulary words in shift_labels and shift_logits
        if new_token_ids is not None:
            ignore_mask = torch.isin(shift_labels, new_token_ids)
            shift_attention_mask_batch = shift_attention_mask_batch * (~ignore_mask).long()

        # Ignore multi-token sequences of that were replaced with a single token
        if replaced_token_seqs_by_len is not None:
            # Create a mask that will be updated where sequences match
            ignore_mask = shift_attention_mask_batch.clone()  # Clone the attention mask to modify it
            # Loop over sequences in skip_token_seqs
            for seq_len, seqs in replaced_token_seqs_by_len.items():
                # Create a sliding window of the same size as the skip_seq and check for matches
                for i in range(shift_labels.size(1) - seq_len + 1):
                    # Check if the sequence matches at position i
                    window = shift_labels[:, i:i + seq_len]
                    curr_mask = torch.all(window.unsqueeze(1) == seqs.unsqueeze(0), dim=-1)
                    if curr_mask.any():
                        # Zero out the ignore mask for the length of the sequence
                        ignore_mask[curr_mask.any(dim=-1), i:i + seq_len] = 0
            # Apply the ignore mask to the attention mask
            shift_attention_mask_batch *= ignore_mask

        return shift_attention_mask_batch, ignore_mask

    # metrics for baseline
    for batch_i, batch in tqdm(enumerate(baseline_vocab_dataloader), total=len(baseline_vocab_dataloader), miniters=10, desc="Evaluating baseline vocabulary..."):
        if add_start_token:
            batch = _add_start_token(batch)

        labels = batch["input_ids"]
        attn_mask = batch["attention_mask"]
        batch.pop("labels")
        with torch.no_grad():
            outputs = model(**batch)
        out_logits = outputs.logits

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

        shift_attention_mask_batch, ignore_mask = _ignore_new_words_in_attention_mask(shift_attention_mask_batch, shift_labels)

        # compute metrics for baseline vocabulary
        if new_token_ids is not None:
            # output vocab is expanded - to compute baseline, need to remove logits for new words
            shift_logits = torch.cat([shift_logits[:, :, :min(new_token_ids)], shift_logits[:, :, max(new_token_ids)+1:]], dim=-1)
        else:
            shift_logits = shift_logits

        background_results, target_results, overall_results, successful_targets = _compute_metrics(model, tokenizer, shift_logits, shift_labels, shift_attention_mask_batch, compute_perplexity=True)
        for metric_name, metric_value in target_results.items():
            target_metrics['baseline'][metric_name].append(metric_value)
        for metric_name, metric_value in background_results.items():
            background_metrics['baseline'][metric_name].append(metric_value)
        for metric_name, metric_value in overall_results.items():
            overall_metrics['baseline'][metric_name].append(metric_value)

    baseline_vocab_dataloader = accelerator.free_memory(baseline_vocab_dataloader)

    gc.collect()
    torch.cuda.empty_cache()
    expanded_vocab_dataloader = accelerator.prepare(expanded_vocab_dataloader)
    # metrics for expanded vocabulary
    for batch_i, batch in tqdm(enumerate(expanded_vocab_dataloader), total=len(expanded_vocab_dataloader),
                               miniters=10, desc="Evaluating expanded vocabulary..."):
        if add_start_token:
            batch = _add_start_token(batch)

        labels = batch["input_ids"]
        attn_mask = batch["attention_mask"]
        batch.pop("labels")
        with torch.no_grad():
            outputs = model(**batch)
        out_logits = outputs.logits

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

        shift_attention_mask_batch, ignore_mask = _ignore_new_words_in_attention_mask(shift_attention_mask_batch, shift_labels)
        original_labels = None if new_token_to_original_first_token is None \
            else new_token_to_orig_first_mapping_tensor[shift_labels]
        background_results, target_results, overall_results, successful_targets = \
            _compute_metrics(model, tokenizer, shift_logits, shift_labels, shift_attention_mask_batch,
                             original_labels=original_labels, return_successful_targets=True, debug=False)
        for metric_name, metric_value in target_results.items():
            target_metrics['fully_expanded'][metric_name].append(metric_value)
        for metric_name, metric_value in background_results.items():
            background_metrics['fully_expanded'][metric_name].append(metric_value)
        for metric_name, metric_value in overall_results.items():
            overall_metrics['fully_expanded'][metric_name].append(metric_value)
        if successful_targets is not None:
            target_metrics['fully_expanded']["successful_targets"].append(successful_targets)

        if new_token_ids is not None:
            # output vocab is expanded - to compute input-only expansion baseline, need to remove logits for new words
            shift_logits = torch.cat(
                [shift_logits[:, :, :min(new_token_ids)], shift_logits[:, :, max(new_token_ids) + 1:]], dim=-1)
        else:
            shift_logits = shift_logits
        background_results, _, overall_results, _ = _compute_metrics(model, tokenizer, shift_logits, shift_labels,
                                                 shift_attention_mask_batch, compute_target_metrics=False)
        _, target_results, _ , _= _compute_metrics(model, tokenizer, shift_logits, original_labels,
                                                 shift_attention_mask_batch, compute_target_metrics=True)
        for metric_name, metric_value in background_results.items():
            background_metrics['expanded_E'][metric_name].append(metric_value)
        for metric_name, metric_value in target_results.items():
            target_metrics['expanded_E'][metric_name].append(metric_value)
        for metric_name, metric_value in overall_results.items():
            overall_metrics['expanded_E'][metric_name].append(metric_value)

    expanded_vocab_dataloader = accelerator.free_memory(expanded_vocab_dataloader)
    gc.collect()
    torch.cuda.empty_cache()

    # handle successfully saved tokens counts
    if "successful_targets" in target_metrics["fully_expanded"]:
        if new_token_to_original_token_len is not None:
            successful_targets = np.concatenate(target_metrics['fully_expanded']["successful_targets"])
            successful_targets_token_counts = np.vectorize(new_token_to_original_token_len.get)(successful_targets)
            other_metrics["successful_pred_saved_tokens"] = successful_targets_token_counts.sum().item() - len(successful_targets_token_counts)
            other_metrics["successful_pred_original_token_len"] = successful_targets_token_counts.sum().item()
        target_metrics['fully_expanded'].pop("successful_targets")

    for eval_type in target_metrics.keys():
        target_metrics[eval_type] = {metric: np.nanmean(np.concatenate([np.atleast_1d(v) for v in results_list])) for metric, results_list in target_metrics[eval_type].items()}
    for eval_type in background_metrics.keys():
        background_metrics[eval_type] = {metric: np.nanmean(np.concatenate([np.atleast_1d(v) for v in results_list])) for metric, results_list in background_metrics[eval_type].items()}
    for eval_type in overall_metrics.keys():
            overall_metrics[eval_type] = {metric: np.nanmean(np.concatenate([np.atleast_1d(v) for v in results_list])) for metric, results_list in overall_metrics[eval_type].items()}

    other_metrics["total_tokens"] = {
            "baseline": baseline_vocab_total_tokens,
            "expanded": new_vocab_total_tokens,
    }
    return background_metrics, target_metrics, other_metrics, overall_metrics


def get_word_filter(args):

    def word_filter(word, token_count):
        is_valid = True
        if args.words_filter_max_n_tokens and not (token_count <= args.words_filter_max_n_tokens):
            is_valid = False
        if args.words_filter_non_en and not all('a' <= char <= 'z' or 'A' <= char <= 'Z' for char in word):
            is_valid = False
        if args.words_filter_numeric and not word.isalpha():
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
        logger.infoo(f"Topline expanded tokenizer - total tokens: {max_vocab_total_tokens} - new words: {n_new_words}")

    new_words += new_words_from_data
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
    if args.translators_path:
        try:
            translators = torch.load(args.translators_path, map_location=torch.device('cpu'))
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
            batch_size=args.extraction_batch_size,
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
    else:
        translators = LinearRepresentationTranslators(do_residual=args.translators_do_residual)
        translators.fit_on_tokens(
            model, tokenizer,
            prompt=args.extraction_prompt,
            prompt_target=args.prompt_target,
            batch_size=args.extraction_batch_size,
            fit_intercept=args.translators_fit_intercept,
            space_prefixed_only=args.translators_learn_on_space_prefixed_words_only,
            min_word_len=args.translators_fit_min_word_len,
        )

    return translators, save_translators


def main(args):
    set_seed(args.seed)

    output_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    baseline_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    logger.info("*** Evaluating expanding input vocabulary ***")
    logger.info("Preparing list of words to add to vocabulary...")
    new_words, orig_tokenization = prepare_new_words(args, tokenizer)
    logger.info(f"Found {len(new_words)} new words: {new_words[:100]}...")
    # dump new words to file
    with open(os.path.join(output_dir, "new_words.txt"), "w") as fp:
        fp.write("\n".join(new_words))

    logger.info("Loading model...")
    mixed_precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    accelerator = Accelerator(mixed_precision=mixed_precision)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16 if mixed_precision == "bf16" else torch.float16)

    model = accelerator.prepare(model)
    logger.info("Running patchscopes on new words...")
    patchscopes_retriever, patchscopes_results = prepare_patchscopes_retriever(args, model, baseline_tokenizer)

    logger.info("Preparing transformations to embedding and lm_head spaces...")
    translators, save_translators = prepare_translators(args, model, tokenizer)
    os.makedirs(output_dir, exist_ok=True)
    if save_translators:
        if args.translators_path is not None:
            os.makedirs(os.path.dirname(args.translators_path), exist_ok=True)
            torch.save(translators, args.translators_path)
        else:
            os.makedirs(output_dir, exist_ok=True)
            torch.save(translators, os.path.join(output_dir, f"translators.pt"))

    logger.info("Adding new words to model vocabulary...")
    model.eval()
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
    model, tokenizer = vocab_modifier.add_words_to_vocab(new_words)

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

    if args.calibrate_new_entries:
        logger.info("Calibrating new LM head entries...")
        calibration_dataset = load_lm_dataset(args.calibration_dataset, language=args.calibration_dataset_language)
        calibration_dataset = calibration_dataset[args.calibration_dataset_split]
        model = vocab_modifier.train_and_apply_calibrators_to_new_entries(calibration_dataset, save_dir=args.calibration_save_dir, max_samples=args.calibration_max_samples, lr=args.calibration_lr, lr_schedule=args.calibration_lr_schedule, num_epochs=args.calibration_num_epochs, batch_size=args.calibration_batch_size, max_length=args.eval_max_length, n_warmup_steps=args.calibration_n_warmup_steps, clip_grad_norm=args.calibration_clip_grad_norm, target_loss_weight=args.calibration_target_loss_weight, subsequent_loss_weight=args.calibration_subsequent_loss_weight)

    logger.info("Done adding words! Patchscopes success rate: "
                f"{len(vocab_modifier.new_words) / (len(vocab_modifier.new_words) + len(vocab_modifier.failed_words))}")

    del patchscopes_results
    gc.collect()
    torch.cuda.empty_cache()

    # compute metrics
    eval_dataset = load_lm_dataset(args.eval_dataset, language=args.eval_dataset_language)
    eval_dataset = eval_dataset[args.eval_dataset_split]

    new_tokens_to_replaced_token_seqs = vocab_modifier.get_new_tokens_to_replaced_token_seqs_map()
    seq_lens = {len(v) for v in new_tokens_to_replaced_token_seqs.values()}
    replaced_token_seqs_by_len = {curr_seq_len: [seq for seq in new_tokens_to_replaced_token_seqs.values() if len(seq) == curr_seq_len] for curr_seq_len in seq_lens}
    new_token_to_original_first_token = {k: v[0] for k, v in new_tokens_to_replaced_token_seqs.items()}
    new_token_to_original_token_len = {k: len(v) for k, v in new_tokens_to_replaced_token_seqs.items()}

    background_metrics, target_metrics, other_metrics, overall_metrics = eval_lm(
        model, accelerator, tokenizer, baseline_tokenizer, eval_dataset,
        batch_size=args.eval_batch_size, top_ks=[5, 10], new_token_ids=vocab_modifier.new_token_ids,
        replaced_token_seqs_by_len=replaced_token_seqs_by_len,
        new_token_to_original_first_token=new_token_to_original_first_token,
        new_token_to_original_token_len=new_token_to_original_token_len,
        max_length=args.eval_max_length,
        eval_max_samples=args.eval_max_samples, text_col_name=args.eval_dataset_text_col,
    )

    other_metrics["n_new_words"] = len(vocab_modifier.new_words)
    other_metrics["patchscopes_success_rate"] = len(vocab_modifier.new_words) / (len(vocab_modifier.new_words) + len(vocab_modifier.failed_words))
    other_metrics["tokens_saved"] = 1 - other_metrics["total_tokens"]["expanded"] / other_metrics["total_tokens"]["baseline"]
    other_metrics["n_attempted_words"] = len(new_words)

    print(other_metrics)
    background_df = pd.DataFrame.from_dict(background_metrics)
    target_df = pd.DataFrame.from_dict(target_metrics)
    overall_df = pd.DataFrame.from_dict(overall_metrics)
    other_df = pd.DataFrame.from_dict(other_metrics)
    print(tabulate(background_df, headers='keys', tablefmt='psql'))
    print(tabulate(target_df, headers='keys', tablefmt='psql'))
    print(tabulate(overall_df, headers='keys', tablefmt='psql'))
    background_df.to_json(os.path.join(output_dir, "metrics_background.json"), indent=4)
    target_df.to_json(os.path.join(output_dir, "metrics_target.json"), indent=4)
    overall_df.to_json(os.path.join(output_dir, "metrics_overall.json"), indent=4)
    other_df.to_json(os.path.join(output_dir, "metrics_other.json"), indent=4)

    config = vars(args)
    with open(os.path.join(output_dir, f"config.json"), "w") as config_file:
        json.dump(config, config_file, indent=4)

    logger.info(f"Results saved to: {output_dir}")

    return


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate vocabulary expansion.")
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--output_dir", type=str, default="./experiments/")

    parser.add_argument("--add_new_words_to_core_vocab", action="store_true", default=False)
    parser.add_argument("--add_space_before_lowercase_words", action="store_true", default=False)
    parser.add_argument("--detokenization_decision_rule", type=str, default="first_id_layer")
    parser.add_argument("--detokenization_decision_rule_E", type=str, default=None)
    parser.add_argument("--detokenization_max_valid_layer", type=int, default=None)
    parser.add_argument("--extraction_batch_size", type=int, default=128)
    parser.add_argument("--extraction_prompt", type=str, default="X")
    parser.add_argument("--patchscopes_prompt", type=str, default="X, X, X, X,")
    parser.add_argument("--prompt_target", type=str, default="X")
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
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--eval_max_length", type=int, default=256)
    parser.add_argument("--eval_dataset_split", type=str, default="test")
    parser.add_argument("--eval_dataset_text_col", type=str, default="text")

    parser.add_argument("--words_dataset", type=str, default=None)
    parser.add_argument("--words_dataset_language", type=str, default=None)
    parser.add_argument("--words_dataset_split", type=str, default="test")
    parser.add_argument("--words_dataset_overlap_split", type=str, default=None)
    parser.add_argument("--words_dataset_text_col", type=str, default="text")
    parser.add_argument("--words_list", type=str, default=None)
    parser.add_argument("--words_list_delimiter", type=str, default=None)
    parser.add_argument("--words_filter_min_freq", type=int, default=None)
    parser.add_argument("--words_filter_max_n_tokens", type=int, default=5)
    parser.add_argument("--words_filter_non_en", action="store_true", default=False)
    parser.add_argument("--words_filter_numeric", action="store_true", default=False)

    args = parser.parse_args()

    assert args.words_dataset is not None or args.words_list is not None, \
        "Please pass either a dataset name (--words_dataset) " \
        "or a path to a file containing a list of words (--words_list)"

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
