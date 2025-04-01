from typing import List, Dict, Union, Optional, Tuple, Any
from collections import defaultdict
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from tokenizers import AddedToken, Tokenizer
import tempfile
import json
import re
import os
from tqdm import tqdm
from copy import deepcopy

import logging

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extend_tokenizer(tokenizer, tokens_to_add, space_prefix="Ġ", keep_special_token_ids=False):
    vocab_size = len(tokenizer._tokenizer.get_vocab(with_added_tokens=False))

    special_tokens_map = dict()
    new_vocab = dict()
    scaffold_vocab = dict()
    scaffold_merges = list()

    if keep_special_token_ids:
        # Add placeholders for existing special tokens:
        # This prevents the new tokens from taking their ID in the embedding table
        for special_token_id, special_token_str in tokenizer.added_tokens_decoder.items():
            special_tokens_map[special_token_id] = (special_token_id, [])
            new_vocab[str(special_token_str)] = special_token_id

    newest_token_id = max(vocab_size, max(tokenizer.added_tokens_decoder.keys())+1)

    for new_tok in tokens_to_add:
        # get token id for inflection word
        orig_encoding = \
            tokenizer.encode(new_tok.replace(space_prefix, " "), add_special_tokens=False) if space_prefix \
                else tokenizer.encode(new_tok, add_special_tokens=False)

        # add scaffold tokens and merges if necessary
        if len(orig_encoding) == 1:
            logger.warning(f"Token '{new_tok}' passed to extend_tokenizer, but already exists in tokenizer as a single token. This might cause misalignment problems when adding new model embeddings later.")
            continue
        curr_new_token = f"{tokenizer._tokenizer.id_to_token(orig_encoding[0])}"

        for i in range(len(orig_encoding) - 1):
            curr_merge = [f"{curr_new_token}", f"{tokenizer._tokenizer.id_to_token(orig_encoding[i + 1])}"]

            curr_new_token = f"{curr_new_token}{tokenizer._tokenizer.id_to_token(orig_encoding[i + 1])}"
            if curr_new_token in new_vocab:
                continue
            scaffold_merges.append(curr_merge)
            new_vocab[curr_new_token] = newest_token_id
            if i < len(orig_encoding) - 2:
                scaffold_vocab[curr_new_token] = newest_token_id
            newest_token_id += 1

    non_scaffold_vocab = {k: v for k, v in new_vocab.items() if k not in scaffold_vocab}
    new_tokenizer = add_words_to_core_vocab(tokenizer, new_vocab, scaffold_merges)
    scaffold_tokenizer = ScaffoldTokenizer(new_tokenizer, tokenizer, scaffold_vocab)
    return scaffold_tokenizer, non_scaffold_vocab, scaffold_vocab


def add_words_to_core_vocab(tokenizer, new_tokens_to_ids, merges_list):
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save tokenizer to the temporary directory
        tokenizer.save_pretrained(temp_dir)

        # Load the tokenizer.json file
        tokenizer_json_path = f"{temp_dir}/tokenizer.json"
        with open(tokenizer_json_path, 'r') as f:
            tokenizer_json = json.load(f)

        # Make edits to the tokenizer.json
        for word, token_id in new_tokens_to_ids.items():
            tokenizer_json['model']['vocab'][word] = token_id

        tokenizer_json['model']['merges'] = tokenizer_json['model']['merges'] + merges_list
        # Save the edited tokenizer.json as a temp file and reload the modified tokenizer
        with open(tokenizer_json_path, 'w') as f:
            json.dump(tokenizer_json, f, indent=2)
        new_tokenizer = AutoTokenizer.from_pretrained(temp_dir, added_tokens_decoder=tokenizer.added_tokens_decoder.copy())

        return new_tokenizer


class ScaffoldTokenizer(PreTrainedTokenizerFast):
    def __init__(
            self,
            extended_tokenizer: Optional[PreTrainedTokenizerFast] = None,
            base_tokenizer: Optional[PreTrainedTokenizerFast] = None,
            scaffold_vocab: Dict[str, int] = None,
            **kwargs
    ):
        # If base_tokenizer is provided, extract its tokenizer_object
        tokenizer_object = extended_tokenizer.backend_tokenizer

        # Call parent constructor
        super().__init__(
            tokenizer_object=tokenizer_object,
            added_tokens_decoder=base_tokenizer.added_tokens_decoder.copy(),
            **kwargs
        )

        # Copy all attributes from the base tokenizer to the current instance
        for attr, value in extended_tokenizer.__dict__.items():
            setattr(self, attr, value)

        # Initialize scaffold-specific attributes
        self.scaffold_vocab = scaffold_vocab or dict()
        self.scaffold_id_map: Dict[int, List[int]] = {}
        self.scaffold_token_ids: Set[int] = set()
        self._build_scaffold_mappings()

    def _build_scaffold_mappings(self):
        """Build efficient mappings for scaffold token IDs to their shortest non-scaffold decomposition."""
        vocab = self.get_vocab()

        # Clear existing mappings
        self.scaffold_id_map.clear()
        self.scaffold_token_ids.clear()

        # Helper function to get all possible tokenizations
        def get_tokenizations(text: str) -> List[List[int]]:
            n = len(text)
            dp = [[] for _ in range(n + 1)]  # dp[i] stores all valid tokenizations up to position i
            dp[0] = [[]]  # empty sequence for empty string

            for i in range(1, n + 1):
                for j in range(i):
                    substr = text[j:i]
                    if substr in vocab:
                        token_id = vocab[substr]
                        # Only consider this tokenization if the token is not a scaffold
                        # (unless it's a single character token that we can't break down further)
                        if (substr not in self.scaffold_vocab) or (len(substr) == 1):
                            for prev_tokens in dp[j]:
                                dp[i].append(prev_tokens + [token_id])

            return dp[n]

        # Build mappings for each scaffold token
        for scaffold, scaffold_id in self.scaffold_vocab.items():
            if scaffold in vocab:
                self.scaffold_token_ids.add(scaffold_id)

                # Get all possible non-scaffold tokenizations
                tokenizations = get_tokenizations(scaffold)

                if tokenizations:
                    # Choose the shortest valid tokenization
                    shortest = min(tokenizations, key=len)
                    if shortest:  # Only store if we found a valid decomposition
                        self.scaffold_id_map[scaffold_id] = shortest
                else:
                    # If no valid tokenization found (e.g., for single character scaffolds),
                    # we keep the original token
                    self.scaffold_id_map[scaffold_id] = [scaffold_id]

    def _replace_scaffold_tokens(self, token_ids: List[int]) -> List[int]:
        """Replace scaffold tokens using pre-computed mapping."""
        result = []
        for token_id in token_ids:
            if token_id in self.special_tokens_encoding_map:
                result.append(self.special_tokens_encoding_map[token_id])
            elif token_id in self.scaffold_token_ids:
                result.extend(self.scaffold_id_map[token_id])
            else:
                result.append(token_id)
        return result

    def _convert_tokens_to_ids(self, tokens):
        """Override to apply scaffold token replacement."""
        base_ids = super()._convert_tokens_to_ids(tokens)
        return self._replace_scaffold_tokens(base_ids)

