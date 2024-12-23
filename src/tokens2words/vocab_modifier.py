from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import Iterable, Union, List, Dict
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoTokenizer
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


class VocabularyModifier(ABC):
    """
    Abstract class for...  # TODO
    """

    def __init__(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            base_tokenizer: PreTrainedTokenizer = None,
            add_to_core_vocab: bool = True,
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

        self.orig_vocab_size = len(tokenizer) if not self.add_to_core_vocab else len(tokenizer._tokenizer.get_vocab(with_added_tokens=False))
        self.num_special_tokens = len(tokenizer._tokenizer.get_added_tokens_decoder())
        self.new_token_ids: List[int] = list()
        self.new_words: List[str] = list()
        self.failed_words: List[str] = list()
        self.entries_cache = {"embedding": dict(), "lm_head": dict()}

        self.calibrators = None

    @abstractmethod
    def free_memory(self) -> None:
        pass

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

    def undo_vocabulary_changes(self) -> None:
        self.tokenizer = deepcopy(self.base_tokenizer)
        self.model.resize_token_embeddings(len(self.base_tokenizer))

    def add_words_to_core_vocab(self, words: List[str], token_ids: List[int]) -> None:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save tokenizer to the temporary directory
            self.tokenizer.save_pretrained(temp_dir)

            # Load the tokenizer.json file
            tokenizer_json_path = f"{temp_dir}/tokenizer.json"
            with open(tokenizer_json_path, 'r') as f:
                tokenizer_json = json.load(f)

            # Make some modifications to the tokenizer.json (example: add a custom entry)
            for word, token_id in zip(words, token_ids):
                tokenizer_json['model']['vocab'][word] = token_id

            # Save the modified tokenizer.json file
            with open(tokenizer_json_path, 'w') as f:
                json.dump(tokenizer_json, f, indent=2)

            # Reload the modified tokenizer (optional)
            self.tokenizer = AutoTokenizer.from_pretrained(temp_dir)

    def add_word_to_vocab(
            self, word: str, embedding_entry: torch.Tensor = None, lm_head_entry: torch.Tensor = None, finalize: bool = True
    ) -> bool:
        if embedding_entry is not None or lm_head_entry is not None:
            # use precomputed entry representations
            pass
        else:
            embedding_entry, lm_head_entry = self.compute_entries_for_word(word)
            if embedding_entry is None or lm_head_entry is None:
                # failed to compute new entries for word
                self.failed_words.append(word)
                return

        if self.add_space_before_lowercase_words and word[0].islower():
            word = self.space_token + word

        # Add new word to the tokenizer
        num_added_tokens = int(word not in self.tokenizer.get_vocab() and (len(self.tokenizer.tokenize(word)) != 1))

        if num_added_tokens > 0:  # don't add word if it already exists
            self.new_words.append(word)
            if finalize:
                self.tokenizer.add_tokens([word])
                self.model.resize_token_embeddings(len(self.tokenizer) + num_added_tokens, mean_resizing=False)
                new_token_idx = len(self.tokenizer) - 1
                if self.add_to_core_vocab:
                    new_token_idx -= self.num_special_tokens
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
            self.add_words_to_core_vocab(self.new_words, self.new_token_ids)
        else:
            for word in self.new_words:
                self.tokenizer.add_tokens([word])

        self.model.resize_token_embeddings(len(self.base_tokenizer) + len(self.new_words))
        if self.add_to_core_vocab:
            pass  # TODO adjust for direct editing of tokenizer

        for new_token_idx, word in zip(
                self.new_token_ids,
                self.new_words,
            ):
            with torch.no_grad():
                self.model.get_input_embeddings().weight[new_token_idx] = self.entries_cache["embedding"][word]
                self.model.get_output_embeddings().weight[new_token_idx] = self.entries_cache["lm_head"][word]
        self.entries_cache = {"embedding": dict(), "lm_head": dict()}

        return self.model, self.tokenizer

    def get_new_tokens_to_replaced_token_seqs_map(self):
        return {token_id: self.base_tokenizer.encode(word, add_special_tokens=False)
                for word, token_id in zip(self.new_words, self.new_token_ids)}

    def train_and_apply_calibrators_to_new_entries(self, dataset, save_dir=None, max_samples=None, lr=1e-4, lr_schedule="linear", num_epochs=1, batch_size=4, max_length=256, n_warmup_steps=0, clip_grad_norm=1.0, target_loss_weight=0.15, subsequent_loss_weight=0.15):
        self.calibrators = self.train_calibrators(dataset, save_dir, max_samples, lr, lr_schedule, num_epochs, batch_size, max_length, n_warmup_steps, clip_grad_norm, target_loss_weight, subsequent_loss_weight)
        self.apply_calibrators_to_new_entries()
        return self.model

    def train_calibrators(self, dataset, save_dir=None, max_samples=None, lr=1e-4, lr_schedule="linear", num_epochs=1, batch_size=4, max_length=256, n_warmup_steps=0, clip_grad_norm=1.0, target_loss_weight=0.15, subsequent_loss_weight=0.15):
        calibration_model = get_calibration_model(self.model, self.orig_vocab_size, len(self.new_words), target_loss_weight, subsequent_loss_weight)

        train_calibrators = True
        if save_dir is not None:
            calibrators_loaded = calibration_model.load_calibrators(save_dir, fail_ok=True)
            train_calibrators = not calibrators_loaded
        if train_calibrators:
            calibration_model = train_calibration_model(calibration_model, self.tokenizer, dataset, save_dir, max_samples=max_samples, lr=lr, lr_schedule=lr_schedule, num_epochs=num_epochs, batch_size=batch_size, max_length=max_length, n_warmup_steps=n_warmup_steps, clip_grad_norm=clip_grad_norm)

        calibrators = calibration_model.get_calibrators()
        return calibrators

    def apply_calibrators_to_new_entries(self, new_tokens_start=None, new_tokens_end=None, embedding_calibrator=None, lm_head_calibrator=None):
        if self.calibrators is not None:
            self.model = merge_calibrators_to_hf_model(self.model, **calibrators)
        else:
            assert (new_tokens_start is not None) and \
                   ((embedding_calibrator is not None) or (lm_head_calibrator is not None)), \
                   "To apply calibrators, you must either train them first or pass calibrators as parameters"

            self.model = merge_calibrators_to_hf_model(
                self.model,
                new_tokens_start, new_tokens_end,
                embedding_calibrator, lm_head_calibrator,
            )
        return self.model


class DetokenizationVocabularyExpander(VocabularyModifier):
    def __init__(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            patchscopes_retriever: PatchscopesRetriever,
            patchscopes_results: Union[np.ndarray, pd.DataFrame, Dict[str, Dict[int, str]]] = None,
            translators: RepresentationTranslators = None,
            detokenization_decision_rule: str = "first_id_layer",
            detokenization_decision_rule_E: str = None,
            max_valid_layer: int = None,
            early_exit_layer: int = None,
            **kwargs
    ):
        super().__init__(model, tokenizer, **kwargs)

        self.detokenization_decision_rule = detokenization_decision_rule
        self.detokenization_decision_rule_E = detokenization_decision_rule_E
        self.max_valid_layer = max_valid_layer
        self.early_exit_layer = early_exit_layer

        self.patchscopes_retriever = patchscopes_retriever
        self.patchscopes_results = patchscopes_results
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

        target_as_embedding = self.translators.to_embedding(target_as_embedding, target_layer_E+1).to(self.model.get_input_embeddings().weight.dtype)
        target_as_lm_head = self.translators.to_lm_head(target_as_lm_head, target_layer+1).to(self.model.get_output_embeddings().weight.dtype)

        return target_as_embedding, target_as_lm_head

    def get_patchscopes_results(self):
        return pd.DataFrame.from_records(self.patchscopes_results)


class HeuristicDetokenizationVocabularyExpander(VocabularyModifier):
    def __init__(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            translators: RepresentationTranslators = None,
            detokenization_layer: int = 3,
            embedding_detokenization_layer: int = None,
            **kwargs
    ):
        super().__init__(model, tokenizer, **kwargs)

        self.detokenization_layer = detokenization_layer
        self.detokenization_layer = detokenization_layer
        self.embedding_detokenization_layer = embedding_detokenization_layer if embedding_detokenization_layer is not None else detokenization_layer
        self.translators = translators

    def _decide_detokenization_end_layer(self, word: str):
        return self.detokenization_layer

    def compute_entries_for_word(
            self, word: str
    ) -> (torch.Tensor, torch.Tensor):
        """

        Args:
            word (str):
                ...
        """
        # TODO replace patchscopes
        last_token_hidden_states = self.patchscopes_retriever.extract_hidden_states(word)

        target_layer = target_layer_E = self._decide_detokenization_end_layer(word)
        if self.detokenization_decision_rule_E is not None:
            target_layer_E = self._decide_detokenization_end_layer(
                word, patchscopes_description_by_layers, self.detokenization_decision_rule_E)

        target_as_embedding = last_token_hidden_states[target_layer_E]
        target_as_lm_head = last_token_hidden_states[target_layer]

        target_as_embedding = self.translators.to_embedding(target_as_embedding, target_layer_E+1).to(self.model.get_input_embeddings().weight.dtype)
        target_as_lm_head = self.translators.to_lm_head(target_as_lm_head, target_layer+1).to(self.model.get_output_embeddings().weight.dtype)

        return target_as_embedding, target_as_lm_head
