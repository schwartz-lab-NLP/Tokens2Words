from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import Iterable, Union, List, Dict, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoTokenizer
from tokenizers import AddedToken
import numpy as np
import pandas as pd
import re
import torch
from torch import nn
from collections import defaultdict
from typing import DefaultDict
import tempfile
import json
from copy import deepcopy

from .representation_translator import RepresentationTranslators
from .word_retriever import PatchscopesRetriever
from .utils.calibration_utils import get_calibration_model, train_calibration_model, merge_calibrators_to_hf_model
from .utils.model_utils import extract_token_i_hidden_states, extract_word_mean_hidden_states, extract_word_mean_embeddings
from .utils.core_vocab_utils import extend_tokenizer


class VocabularyModifier(ABC):
    """
    Abstract class for...  # TODO
    """

    def __init__(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            base_tokenizer: PreTrainedTokenizer = None,
            add_to_core_vocab: bool = False,
            add_space_before_lowercase_words: bool = False,
            space_token: str = "Ġ",
            **kwargs
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.base_tokenizer = base_tokenizer if base_tokenizer is not None else deepcopy(tokenizer)

        self.add_to_core_vocab = add_to_core_vocab
        self.add_space_before_lowercase_words = add_space_before_lowercase_words
        self.space_token = space_token

        self.orig_vocab_size = len(tokenizer)  # if not self.add_to_core_vocab else len(tokenizer._tokenizer.get_vocab(with_added_tokens=False))
        self.num_special_tokens = len(tokenizer._tokenizer.get_added_tokens_decoder())
        self.new_token_ids: List[int] = list()
        self.new_words: List[str] = list()
        self.failed_words: List[str] = list()
        self.entries_cache = {"embedding": dict(), "lm_head": dict()}

        self.calibrators = None
        self.existing_tokens_to_calibrate = None

    @abstractmethod
    def compute_entries_for_word(
            self, word: str
    ) -> (torch.Tensor, torch.Tensor):
        """
        Computes the entries in the embedding and LM head matrices for a given word.

        Args:
            word (str): The word to add to the vocabulary

        Returns:
            embedding entry (torch.Tensor): The transformed representation in embedding space.
            lm_head entry (torch.Tensor): The transformed representation in LM head space.
        """
        pass

    def undo_vocabulary_changes(self):
        self.tokenizer = deepcopy(self.base_tokenizer)
        # self.model.resize_token_embeddings(len(self.base_tokenizer))

        self.orig_vocab_size = len(self.tokenizer)  # if not self.add_to_core_vocab else len(self.tokenizer._tokenizer.get_vocab(with_added_tokens=False))
        self.num_special_tokens = len(self.tokenizer._tokenizer.get_added_tokens_decoder())
        self.new_token_ids = list()
        self.new_words = list()
        self.failed_words = list()
        self.entries_cache = {"embedding": dict(), "lm_head": dict()}
        self.model.resize_token_embeddings(len(self.tokenizer))
        return self.model, self.tokenizer

    def add_word_to_vocab(
            self, word: str,
            finalize: bool = True,
            embedding_entry: torch.Tensor = None, lm_head_entry: torch.Tensor = None,
    ) -> bool:
        if embedding_entry is not None and lm_head_entry is not None:
            # use precomputed entry representations
            pass
        else:
            embedding_entry, lm_head_entry = self.compute_entries_for_word(word)
            if embedding_entry is None or lm_head_entry is None:
                # failed to compute new entries for word
                self.failed_words.append(word)
                return False

        if self.add_space_before_lowercase_words and word[0].islower():
            word = self.space_token + word

        # Add new word to the tokenizer
        num_added_tokens = int(word not in self.tokenizer.get_vocab() and (len(self.tokenizer.tokenize(word)) != 1))

        if num_added_tokens > 0:  # don't add word if it already exists
            self.new_words.append(word)
            if finalize:
                if self.add_to_core_vocab:
                    raise NotImplementedError("In VocabularyModifier, finalize=True is only implemented for adding multiple words to the vocabulary (i.e., in 'add_words_to_vocab', not in 'add_word_to_vocab').")
                    # new_token_idx -= self.num_special_tokens
                self.tokenizer.add_tokens([AddedToken(word)])
                self.model.resize_token_embeddings(len(self.tokenizer) + num_added_tokens, mean_resizing=False)
                new_token_idx = len(self.tokenizer) - 1
                self.new_token_ids.append(new_token_idx)

                with torch.no_grad():
                    self.model.get_input_embeddings().weight[new_token_idx] = embedding_entry
                    self.model.get_output_embeddings().weight[new_token_idx] = lm_head_entry
            else:
                self.entries_cache["embedding"][word] = embedding_entry
                self.entries_cache["lm_head"][word] = lm_head_entry

        return num_added_tokens > 0

    def add_words_to_vocab(
            self, words: Iterable[str],
            precomputed_embedding_entries: Dict[str, torch.Tensor] = None,
            precomputed_lm_head_entries: Dict[str, torch.Tensor] = None,
    ):
        if precomputed_embedding_entries is not None and precomputed_lm_head_entries is not None:
            self.entries_cache["embedding"] = precomputed_embedding_entries
            self.entries_cache["lm_head"] = precomputed_lm_head_entries
        else:
            for word in tqdm(words, total=len(words), desc="Computing entry representations for words...", unit="word"):
                self.add_word_to_vocab(word, finalize=False)

        self.new_token_ids = list(range(self.orig_vocab_size, self.orig_vocab_size+len(self.new_words)))
        if self.add_to_core_vocab:
            space_prefix = self.tokenizer.tokenize(" ")[0][0]

            extended_tokenizer, new_vocab, scaffold_vocab = extend_tokenizer(self.tokenizer, self.new_words, space_prefix=space_prefix, keep_special_token_ids=False)
            self.tokenizer = extended_tokenizer

            self.model.resize_token_embeddings(len(self.tokenizer))

            word_to_new_id_map = new_vocab
        else:
            for word in self.new_words:
                self.tokenizer.add_tokens([AddedToken(word)])

            self.model.resize_token_embeddings(len(self.base_tokenizer) + len(self.new_words))

            word_to_new_id_map = dict(zip(self.new_words, self.new_token_ids))

        for word, new_token_idx in word_to_new_id_map.items():
            if self.add_to_core_vocab and word.startswith(space_prefix) and word not in self.entries_cache["embedding"]:
                word = word.replace(space_prefix, "")
            with torch.no_grad():
                self.model.get_input_embeddings().weight[new_token_idx] = self.entries_cache["embedding"][word]
                self.model.get_output_embeddings().weight[new_token_idx] = self.entries_cache["lm_head"][word]
        self.entries_cache = {"embedding": dict(), "lm_head": dict()}

        return self.model, self.tokenizer

    def get_new_tokens_to_replaced_token_seqs_map(self, remove_prefix_space=True):
        mapping = {token_id: self.base_tokenizer.encode(word, add_special_tokens=False)
                   for word, token_id in zip(self.new_words, self.new_token_ids)}
        if remove_prefix_space:
            mapping = {token_id: encoding[1:] if not bool(self.base_tokenizer.decode(encoding[0])) else encoding
                       for token_id, encoding in mapping.items()}
        return mapping

    def train_and_apply_calibrators_to_new_entries(
            self, dataset, save_dir=None, overwrite_cache=False,
            max_samples=None, lr=1e-4, lr_schedule="linear",
            num_epochs=1, batch_size=4, max_length=256,
            n_warmup_steps=0, clip_grad_norm=1.0,
            target_loss_weight=0.15, subsequent_loss_weight=0.15,
            mixed_precision=None, learn_gate_activation=False,
            existing_tokens_to_calibrate=None,
            new_tokens_to_orig_first_map=None, soft_allow_orig_first_tokens=False,
            existing_tokens_for_similarity_loss=None,
    ):

        self.calibrators = self.train_calibrators(
            dataset, save_dir, overwrite_cache, max_samples,
            lr, lr_schedule, num_epochs, batch_size, max_length,
            n_warmup_steps, clip_grad_norm, target_loss_weight,
            subsequent_loss_weight, mixed_precision, learn_gate_activation,
            existing_tokens_to_calibrate=existing_tokens_to_calibrate if existing_tokens_to_calibrate is not None else self.existing_tokens_to_calibrate,
            new_tokens_to_orig_first_map=new_tokens_to_orig_first_map, soft_allow_orig_first_tokens=soft_allow_orig_first_tokens,
            existing_tokens_for_similarity_loss=existing_tokens_for_similarity_loss,
        )
        self.apply_calibrators_to_new_entries()
        return self.model

    def load_calibrators(self, save_dir):
        calibration_model = get_calibration_model(
            self.model,
            self.orig_vocab_size,
            len(self.new_words),
            existing_tokens_to_calibrate=getattr(self, 'existing_tokens_to_calibrate', None)
        )
        calibrators_loaded = calibration_model.load_calibrators(save_dir, fail_ok=True)
        if calibrators_loaded:
            return calibration_model.get_calibrators()
        return None

    def set_existing_tokens_to_calibrate(self, token_ids):
        self.existing_tokens_to_calibrate = token_ids

    def train_calibrators(self, dataset, save_dir=None, overwrite_cache=False, max_samples=None,
                          lr=1e-4, lr_schedule="linear", num_epochs=1, batch_size=4,
                          max_length=256, n_warmup_steps=0, clip_grad_norm=1.0,
                          target_loss_weight=0.15, subsequent_loss_weight=0.15,
                          mixed_precision=None, learn_gate_activation=False,
                          learn_per_token_bias=False, existing_tokens_to_calibrate=None,
                          new_tokens_to_orig_first_map=None, soft_allow_orig_first_tokens=False,
                          existing_tokens_for_similarity_loss=None,
                          ):
        existing_tokens_to_calibrate = existing_tokens_to_calibrate if existing_tokens_to_calibrate is not None else self.existing_tokens_to_calibrate

        calibration_model = get_calibration_model(
            self.model,
            self.orig_vocab_size,
            len(self.new_words),
            learn_gate_activation=learn_gate_activation,
            existing_tokens_to_calibrate=existing_tokens_to_calibrate,
            new_tokens_to_orig_first_map=new_tokens_to_orig_first_map, soft_allow_orig_first_tokens=soft_allow_orig_first_tokens,
        )

        if (learn_per_token_bias or existing_tokens_to_calibrate is not None) and max_samples is not None:
            max_samples = max_samples // 2

        train_calibrators = True
        if save_dir is not None and not overwrite_cache:
            calibrators_loaded = calibration_model.load_calibrators(save_dir, fail_ok=True)
            train_calibrators = not calibrators_loaded

        if train_calibrators:
            calibration_model = train_calibration_model(
                calibration_model,
                self.tokenizer,
                dataset,
                save_dir,
                max_samples=max_samples,
                lr=lr,
                lr_schedule=lr_schedule,
                num_epochs=num_epochs,
                batch_size=batch_size,
                max_length=max_length,
                n_warmup_steps=n_warmup_steps,
                clip_grad_norm=clip_grad_norm,
                mixed_precision=mixed_precision,
                freeze_existing_tokens=existing_tokens_to_calibrate is not None,
                existing_tokens_for_similarity_loss=existing_tokens_for_similarity_loss,
            )

            # 2-step training when adding per-token bias term, or fine-tuning existing tokens
            if learn_per_token_bias or existing_tokens_to_calibrate is not None:
                if learn_per_token_bias:
                    calibration_model.set_use_bias(True)
                calibration_model = train_calibration_model(
                    calibration_model,
                    self.tokenizer,
                    dataset,
                    save_dir,
                    max_samples=max_samples,
                    lr=lr,
                    lr_schedule=lr_schedule,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    max_length=max_length,
                    n_warmup_steps=n_warmup_steps,
                    clip_grad_norm=clip_grad_norm,
                    mixed_precision=mixed_precision,
                    existing_tokens_for_similarity_loss=existing_tokens_for_similarity_loss,
                )

        return calibration_model, calibration_model.get_calibrators()

    def set_calibrators(self, calibrators):
        self.calibrators = calibrators

    def apply_calibrators_to_new_entries(
            self,
            new_tokens_start=None,
            new_tokens_end=None,
            embedding_calibrator=None,
            lm_head_calibrator=None,
            existing_tokens_to_calibrate=None,
            existing_tokens_embedding_calibrator=None,
            existing_tokens_lm_head_calibrator=None
    ):
        if self.calibrators is not None:
            self.model = merge_calibrators_to_hf_model(self.model, **self.calibrators)
        else:
            assert (new_tokens_start is not None) and \
                   ((embedding_calibrator is not None) or
                    (lm_head_calibrator is not None)), \
                "To apply calibrators, you must either train them first or pass calibrators as parameters"

            self.model = merge_calibrators_to_hf_model(
                self.model,
                new_tokens_start=new_tokens_start,
                new_tokens_end=new_tokens_end,
                embedding_calibrator=embedding_calibrator,
                lm_head_calibrator=lm_head_calibrator,
                existing_tokens_to_calibrate=existing_tokens_to_calibrate if existing_tokens_to_calibrate is not None else self.existing_tokens_to_calibrate,
                existing_tokens_embedding_calibrator=existing_tokens_embedding_calibrator,
                existing_tokens_lm_head_calibrator=existing_tokens_lm_head_calibrator
            )
        return self.model

    def get_model_and_tokenizer(self):
        return self.model, self.tokenizer


class DetokenizationVocabularyExpander(VocabularyModifier):
    def __init__(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            patchscopes_retriever: PatchscopesRetriever,
            patchscopes_results: Union[np.ndarray, pd.DataFrame, Dict[str, Dict[int, str]]] = None,
            patchscopes_force_starts_with_word: bool = True,
            translators: RepresentationTranslators = None,
            detokenization_decision_rule: str = "first_id_layer",
            detokenization_decision_rule_E: str = None,
            max_valid_layer: int = None,
            early_exit_layer: int = None,
            patchscopes_word_batch_size: int = 8,
            **kwargs
    ):
        super().__init__(model, tokenizer, **kwargs)

        self.detokenization_decision_rule = detokenization_decision_rule
        self.detokenization_decision_rule_E = detokenization_decision_rule_E
        self.max_valid_layer = max_valid_layer
        self.early_exit_layer = early_exit_layer

        self.patchscopes_retriever = patchscopes_retriever
        self.patchscopes_results = patchscopes_results
        self.patchscopes_force_starts_with_word = patchscopes_force_starts_with_word
        self.patchscopes_word_batch_size = patchscopes_word_batch_size
        if patchscopes_results is None:
            # create dict that maps new words (str) to a list of their patchscopes output per layer
            self.patchscopes_results: DefaultDict[str, List[str]] = defaultdict(list)

        self.translators = translators

    def _decide_detokenization_end_layer(self, word: str, patchscopes_results: Iterable[str], decision_rule=None):
        decision_rule = self.detokenization_decision_rule if decision_rule is None else decision_rule
        patchscopes_results = np.array(patchscopes_results).astype(str)
        if self.early_exit_layer is not None:
            patchscopes_results = patchscopes_results[:self.early_exit_layer]

        # Check if each layer's result starts with the word
        patchscopes_results = np.char.strip(patchscopes_results)
        starts_with_word = np.char.startswith(patchscopes_results, word)

        # Count occurrences of word in each layer's result
        # Use word boundary \b to match whole words only
        pattern = f"\\b{re.escape(word)}\\b"

        counts = np.array([len(re.findall(pattern, s)) for s in patchscopes_results])
        if self.patchscopes_force_starts_with_word:
            counts[~starts_with_word] = 0
        if np.all(counts == 0):
            return None

        result = None
        if decision_rule in ["first_id_layer", "1st_id_layer"]:
            result = np.argmax(counts > 0).item()
        if decision_rule in ["2nd_id_layer", "3rd_id_layer", "4th_id_layer", "4th_id_layer"]:
            indices = np.where(counts > 0)[0]
            if (decision_rule == "4th_id_layer") and (len(indices) >= 4):
                result = indices[3]
            elif (decision_rule in ["3rd_id_layer", "4th_id_layer"]) and (len(indices) >= 3):
                result = indices[2]
            elif len(indices) >= 2:
                result = indices[1]
            elif len(indices) >= 1:
                result = indices[0]
        if decision_rule == "max_id_layer":
            result = np.argmax(counts).item()
        elif decision_rule == "last_id_layer":
            result = (len(counts) - np.argmax((counts > 0)[::-1]) - 1).item()
        elif decision_rule == "first_layer_with_2_repeats":
            result = (np.argmax(counts >= 2)).item()
        elif decision_rule == "last_layer_with_2_repeats":
            result = (len(counts) - np.argmax((counts >= 2)[::-1]) - 1).item()

        if self.max_valid_layer is not None and result > self.max_valid_layer:
            # default to first id layer
            result = np.argmax(counts > 0).item()

        return result

    def compute_entries_for_word(
            self, word: str
    ) -> (torch.Tensor, torch.Tensor):
        """

        Args:
            word (str):
                ...
        """
        if word not in self.patchscopes_results:
            patchscopes_description_by_layers, last_token_hidden_states = \
                self.patchscopes_retriever.get_hidden_states_and_retrieve_word(word)
            self.patchscopes_results[word] = patchscopes_description_by_layers
        else:
            patchscopes_description_by_layers = self.patchscopes_results[word]
            last_token_hidden_states = self.patchscopes_retriever.extract_hidden_states(word)

        target_layer = target_layer_E = self._decide_detokenization_end_layer(word, patchscopes_description_by_layers)
        if self.detokenization_decision_rule_E is not None:
            target_layer_E = self._decide_detokenization_end_layer(
                word, patchscopes_description_by_layers, self.detokenization_decision_rule_E)

        if target_layer is None:  # detokenization did not occur
            return None, None

        target_as_embedding = last_token_hidden_states[target_layer_E]
        target_as_lm_head = last_token_hidden_states[target_layer]

        if self.translators is not None:
            target_as_embedding = self.translators.to_embedding(target_as_embedding, target_layer_E+1).to(self.model.get_input_embeddings().weight.dtype)
            target_as_lm_head = self.translators.to_lm_head(target_as_lm_head, target_layer+1).to(self.model.get_output_embeddings().weight.dtype)

        return target_as_embedding, target_as_lm_head

    def compute_entries_for_words_batch(self, words: List[str]) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute entries for multiple words in batches"""
        # Filter out words that are already processed
        new_words = [word for word in words if word not in self.patchscopes_results]

        # Process new words in batches
        if new_words:
            batch_results = self.patchscopes_retriever.get_hidden_states_and_retrieve_words_batch(new_words)
            for word, (descriptions, hidden_states) in batch_results.items():
                self.patchscopes_results[word] = descriptions

        # Compute entries for all words
        entries = {}
        for word in words:
            if word in self.patchscopes_results:
                patchscopes_description_by_layers = self.patchscopes_results[word]
                last_token_hidden_states = batch_results[word][1] if word in new_words else \
                    self.patchscopes_retriever.extract_hidden_states(word)

                target_layer = target_layer_E = self._decide_detokenization_end_layer(
                    word, patchscopes_description_by_layers)
                if self.detokenization_decision_rule_E is not None:
                    target_layer_E = self._decide_detokenization_end_layer(
                        word, patchscopes_description_by_layers, self.detokenization_decision_rule_E)

                if target_layer is not None:
                    target_as_embedding = last_token_hidden_states[target_layer_E]
                    target_as_lm_head = last_token_hidden_states[target_layer]

                    if self.translators is not None:
                        target_as_embedding = self.translators.to_embedding(
                            target_as_embedding, target_layer_E + 1
                        ).to(self.model.get_input_embeddings().weight.dtype)
                        target_as_lm_head = self.translators.to_lm_head(
                            target_as_lm_head, target_layer + 1
                        ).to(self.model.get_output_embeddings().weight.dtype)

                    entries[word] = (target_as_embedding, target_as_lm_head)
                else:
                    entries[word] = (None, None)

        return entries

    def add_words_to_vocab_batch(
            self, words: List[str],
            precomputed_embedding_entries: Dict[str, torch.Tensor] = None,
            precomputed_lm_head_entries: Dict[str, torch.Tensor] = None,
    ):
        """Add multiple words to vocabulary using batched processing"""
        if precomputed_embedding_entries is not None and precomputed_lm_head_entries is not None:
            self.entries_cache["embedding"] = precomputed_embedding_entries
            self.entries_cache["lm_head"] = precomputed_lm_head_entries
        else:
            # Process words in batches
            for i in range(0, len(words), self.patchscopes_word_batch_size):
                batch_words = words[i:i + self.patchscopes_word_batch_size]
                batch_entries = self.compute_entries_for_words_batch(batch_words)

                for word, (embedding_entry, lm_head_entry) in batch_entries.items():
                    if embedding_entry is not None and lm_head_entry is not None:
                        if self.add_space_before_lowercase_words and word[0].islower():
                            word = self.space_token + word

                        if word not in self.tokenizer.get_vocab() and (len(self.tokenizer.tokenize(word)) != 1):
                            self.new_words.append(word)
                            self.entries_cache["embedding"][word] = embedding_entry
                            self.entries_cache["lm_head"][word] = lm_head_entry
                    else:
                        self.failed_words.append(word)

        # Continue with the existing logic for finalizing vocabulary changes
        return self.add_words_to_vocab(None,
                                       precomputed_embedding_entries=self.entries_cache["embedding"],
                                       precomputed_lm_head_entries=self.entries_cache["lm_head"])

    def get_patchscopes_results(self):
        return pd.DataFrame.from_records(self.patchscopes_results)


class HeuristicDetokenizationVocabularyExpander(VocabularyModifier):
    def __init__(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            translators: RepresentationTranslators = None,
            detokenization_layer: int = 5,
            embedding_detokenization_layer: int = None,
            extract_hidden_states_as_dict: bool = False,
            use_mean_hidden_states: bool = False,
            use_mean_embeddings: bool = False,
            **kwargs
    ):
        super().__init__(model, tokenizer, **kwargs)

        self.detokenization_layer = detokenization_layer
        self.embedding_detokenization_layer = embedding_detokenization_layer if embedding_detokenization_layer is not None else detokenization_layer
        self.translators = translators
        self.extract_hidden_states_as_dict = extract_hidden_states_as_dict
        self.use_mean_hidden_states = use_mean_hidden_states
        self.use_mean_embeddings = use_mean_embeddings

    def compute_entries_for_word(
            self, word: str
    ) -> (torch.Tensor, torch.Tensor):
        """

        Args:
            word (str):
                ...
        """
        if self.use_mean_hidden_states:
            word_representations = extract_word_mean_hidden_states(
                self.model, self.tokenizer, word,
                return_dict=self.extract_hidden_states_as_dict, verbose=False)
        elif self.use_mean_embeddings:
            word_representations = extract_word_mean_embeddings(
                self.model, self.tokenizer, word,
                return_dict=self.extract_hidden_states_as_dict, verbose=False)
        else:
            word_representations = extract_token_i_hidden_states(
                self.model, self.tokenizer, word, token_idx_to_extract=-1,
                return_dict=self.extract_hidden_states_as_dict, verbose=False)

        target_layer = self.detokenization_layer
        target_layer_E = self.embedding_detokenization_layer

        target_as_embedding = word_representations[target_layer_E]
        target_as_lm_head = word_representations[target_layer]
        
        if self.translators is not None:
            target_as_embedding = self.translators.to_embedding(target_as_embedding, target_layer_E+1).detach().to(self.model.get_input_embeddings().weight.dtype)
            target_as_lm_head = self.translators.to_lm_head(target_as_lm_head, target_layer+1).detach().to(self.model.get_output_embeddings().weight.dtype)

        return target_as_embedding, target_as_lm_head


class PatchscopesLimitedHeuristicDetokenizationVocabularyExpander(VocabularyModifier):
    def __init__(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            patchscopes_retriever: PatchscopesRetriever,
            patchscopes_results: Union[np.ndarray, pd.DataFrame, Dict[str, Dict[int, str]]] = None,
            patchscopes_force_starts_with_word: bool = True,
            translators: RepresentationTranslators = None,
            detokenization_layer: int = 5,
            embedding_detokenization_layer: int = None,
            extract_hidden_states_as_dict: bool = False,
            use_mean_hidden_states: bool = False,
            use_mean_embeddings: bool = False,
            **kwargs
    ):
        super().__init__(model, tokenizer, **kwargs)

        # Patchscopes-related attributes from DetokenizationVocabularyExpander
        self.patchscopes_retriever = patchscopes_retriever
        self.patchscopes_results = patchscopes_results
        self.patchscopes_force_starts_with_word = patchscopes_force_starts_with_word
        if patchscopes_results is None:
            self.patchscopes_results: DefaultDict[str, List[str]] = defaultdict(list)

        # Representation-related attributes from HeuristicDetokenizationVocabularyExpander
        self.detokenization_layer = detokenization_layer
        self.embedding_detokenization_layer = embedding_detokenization_layer if embedding_detokenization_layer is not None else detokenization_layer
        self.translators = translators
        self.extract_hidden_states_as_dict = extract_hidden_states_as_dict
        self.use_mean_hidden_states = use_mean_hidden_states
        self.use_mean_embeddings = use_mean_embeddings

    def _check_word_in_patchscopes(self, word: str) -> bool:
        """Check if the word is retrieved by patchscopes."""
        if word not in self.patchscopes_results:
            patchscopes_description_by_layers, _ = self.patchscopes_retriever.get_hidden_states_and_retrieve_word(word)
            self.patchscopes_results[word] = patchscopes_description_by_layers
        else:
            patchscopes_description_by_layers = self.patchscopes_results[word]

        # Check if word appears in any layer's output
        patchscopes_results = np.array(patchscopes_description_by_layers).astype(str)
        patchscopes_results = np.char.strip(patchscopes_results)

        # Count occurrences of word in each layer's result
        pattern = f"\\b{re.escape(word)}\\b"
        counts = np.array([len(re.findall(pattern, s)) for s in patchscopes_results])

        if self.patchscopes_force_starts_with_word:
            starts_with_word = np.char.startswith(patchscopes_results, word)
            counts[~starts_with_word] = 0

        return np.any(counts > 0)

    def compute_entries_for_word(
            self, word: str
    ) -> (torch.Tensor, torch.Tensor):
        """
        Compute entries for a word, but only if it's found in patchscopes results.
        Uses the same representation computation as HeuristicDetokenizationVocabularyExpander.
        """
        # First check if the word is retrieved by patchscopes
        if not self._check_word_in_patchscopes(word):
            return None, None

        # If word is found in patchscopes, compute representations using the heuristic method
        if self.use_mean_hidden_states:
            word_representations = extract_word_mean_hidden_states(
                self.model, self.tokenizer, word,
                return_dict=self.extract_hidden_states_as_dict, verbose=False)
        elif self.use_mean_embeddings:
            word_representations = extract_word_mean_embeddings(
                self.model, self.tokenizer, word,
                return_dict=self.extract_hidden_states_as_dict, verbose=False)
        else:
            word_representations = extract_token_i_hidden_states(
                self.model, self.tokenizer, word, token_idx_to_extract=-1,
                return_dict=self.extract_hidden_states_as_dict, verbose=False)

        target_layer = self.detokenization_layer
        target_layer_E = self.embedding_detokenization_layer

        target_as_embedding = word_representations[target_layer_E]
        target_as_lm_head = word_representations[target_layer]

        if self.translators is not None:
            target_as_embedding = self.translators.to_embedding(target_as_embedding, target_layer_E + 1).detach().to(
                self.model.get_input_embeddings().weight.dtype)
            target_as_lm_head = self.translators.to_lm_head(target_as_lm_head, target_layer + 1).detach().to(
                self.model.get_output_embeddings().weight.dtype)

        return target_as_embedding, target_as_lm_head

    def get_patchscopes_results(self):
        return pd.DataFrame.from_records(self.patchscopes_results)

