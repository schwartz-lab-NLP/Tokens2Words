import os
from tqdm import tqdm
import math
from abc import ABC, abstractmethod
from typing import Iterable, Union, List
from transformers import PreTrainedModel, PreTrainedTokenizer
import torch
from torch import nn
from torch.utils.data import Dataset, IterableDataset
import numpy as np

from .utils.model_utils import learn_linear_map, extract_vocab_hidden_states
from .utils.model_utils import learn_mlp, learn_ffn
from .utils.procrustes.orthogonal import orthogonal as orthogonal_procrustes


class RepresentationTranslators(ABC, nn.Module):
    """
    Abstract class for mapping intermediate model representations to the model's embedding and lm_head spaces.
    """

    def __init__(self):
        super(RepresentationTranslators, self).__init__()
        self.embedding_maps = nn.ModuleDict()
        self.lm_head_maps = nn.ModuleDict()

    @abstractmethod
    def fit_on_dataset(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            dataset: Union[List[str], Dataset, IterableDataset, "datasets.Dataset"],
    ) -> None:
        """
        Learns transformations that map representations from every layer to the embedding/lm_head spaces,
        by fitting a transformation from intermediate model representations of vocabulary tokens, as computed in
        the dataset's examples, to the corresponding token rows in the embedding and lm_head matrices.

        Args:
            model (PreTrainedModel):
                Model to extract embeddings and intermediate representations from.
            tokenizer (PreTrainedTokenizer):
                Tokenizer to use when extracting embeddings.
            dataset (Union[List[str], Dataset, IterableDataset, "datasets.Dataset"]):
                Data to train transformations on.
        """
        pass

    @abstractmethod
    def fit_on_tokens(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            token_ids: Iterable[int] = None,
            prompt: str = "{target}",
            prompt_target: str = "{target}",
    ) -> None:
        """
        Learns transformations that map representations from every layer to the embedding/lm_head spaces,
        by fitting a transformation from intermediate model representations of single tokens from the vocabulary,
        as extracted using the given prompt, to the respective rows in the embedding and lm_head matrices.

        Args:
            model (PreTrainedModel):
                Model to extract embeddings and intermediate representations from.
            tokenizer (PreTrainedTokenizer):
                Tokenizer to use when extracting embeddings.
            tokens (Iterable[int]):
                The list of token_ids to use as training data.
                If not given, will train over the entire vocabulary.
            prompt (str):
                The prompt to use when extracting token representations, where the parameter prompt_target
                will be replaced with the target token.
            prompt_target (str):
                The placeholder for the target token in the prompt for extracting hidden states.

        """
        pass

    @abstractmethod
    def to_embedding(self, representations: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Transforms the given representations to the embedding space.

        Args:
            representations (torch.Tensor): The intermediate representations to transform.

        Returns:
            torch.Tensor: The transformed representations in embedding space.
        """
        pass

    @abstractmethod
    def to_lm_head(self, representations: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Transforms the given representations to the lm_head space.

        Args:
            representations (torch.Tensor): The intermediate representations to transform.

        Returns:
            torch.Tensor: The transformed representations in lm_head space.
        """
        pass

    def _filter_tokens(
            self, tokenizer, token_ids_to_extract, alpha_only: bool = True, min_word_len: int = None, space_prefixed_only: bool = False):
        tokens_to_extract = np.array([tokenizer.decode(tok_id) for tok_id in token_ids_to_extract])
        tokens_filter = np.ones_like(tokens_to_extract, dtype=bool)
        if alpha_only:
            tokens_filter &= np.char.isalpha(tokens_to_extract)
        if min_word_len is not None:
            tokens_filter &= np.char.str_len(tokens_to_extract) > min_word_len
        if space_prefixed_only:
            tokens_str_rep = np.array(tokenizer.convert_ids_to_tokens(token_ids_to_extract))
            tokens_filter &= np.char.startswith(tokens_str_rep, "▁") | np.char.startswith(tokens_str_rep, "Ġ")
        # # remove tokens which are tokenized into 2 or more tokens when found at the start of sentence
        # tokens_filter &= np.array(
        #     [len(tokenizer.encode(token, add_special_tokens=False)) for token in tokens_to_extract]) == 1
        token_ids_to_extract = [token_id for token_id in np.array(token_ids_to_extract)[tokens_filter]]
        return token_ids_to_extract, tokens_filter


class LinearRepresentationTranslators(RepresentationTranslators):
    """
    Transforms intermediate model representations to the embedding and lm_head spaces
    using linear maps.
    """

    def __init__(self, do_residual=False):
        super(LinearRepresentationTranslators, self).__init__()
        self.do_residual = do_residual

    def fit_on_tokens(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            token_ids: Iterable[int] = None,
            prompt: str = "{target}",
            prompt_target: str = "{target}",
            batch_size: int = 128,
            layer_batch_size: int = 8,
            min_word_len: int = None,
            alpha_only: bool = False,
            space_prefixed_only: bool = False,
            fit_intercept: bool = False,
    ) -> None:
        """
        Learns transformations that map representations from every layer to the embedding/lm_head spaces,
        by fitting a transformation from intermediate model representations of single tokens from the vocabulary,
        as extracted using the given prompt, to the respective rows in the embedding and lm_head matrices.

        Args:
            model (PreTrainedModel):
                Model to extract embeddings and intermediate representations from.
            tokenizer (PreTrainedTokenizer):
                Tokenizer to use when extracting embeddings.
            token_ids (Iterable[int]):
                The list of token_ids to use as training data.
                If not given, will train over the entire vocabulary.
            prompt (str):
                The prompt to use when extracting token representations, where the parameter prompt_target
                will be replaced with the target token.
            prompt_target (str):
                The placeholder for the target token in the prompt for extracting hidden states.
            batch_size (int):
                Batch size to use when extracting token representations.
            layer_batch_size (int):
                Number of layers to compute translators for in parallel.
        """

        input_embeddings = model.get_input_embeddings().weight.detach().cpu()
        lm_head_weights = model.get_output_embeddings().weight.detach().cpu()
        # some models have embeddings for special tokens that aren't included in the "regular" vocabulary
        input_embeddings = input_embeddings[:tokenizer.vocab_size]
        lm_head_weights = lm_head_weights[:tokenizer.vocab_size]

        if token_ids is not None and len(token_ids) != tokenizer.vocab_size:
            input_embeddings = input_embeddings[token_ids]
            lm_head_weights = lm_head_weights[token_ids]
        else:
            token_ids, tokens_filter = \
                self._filter_tokens(tokenizer, np.arange(tokenizer.vocab_size), alpha_only, min_word_len)
            input_embeddings = input_embeddings[tokens_filter]
            lm_head_weights = lm_head_weights[tokens_filter]

        n_layers = model.config.num_hidden_layers
        layer_batch_size = n_layers if layer_batch_size is None else layer_batch_size
        for start_layer_i in tqdm(range(1, n_layers+1, layer_batch_size), desc="Fitting maps to embedding and lm_head spaces", unit="Layer batch"):
            layers_to_learn = None if layer_batch_size is None else \
                list(range(start_layer_i, min(start_layer_i + layer_batch_size, n_layers+1)))
            all_hidden_states = extract_vocab_hidden_states(model, tokenizer, token_ids, prompt, prompt_target, batch_size, layers_to_learn)
            for layer in tqdm(layers_to_learn, total=len(layers_to_learn), unit="layers", desc="Fitting maps..."):
                hidden_states = all_hidden_states[layer]
                if self.do_residual:
                    self.embedding_maps[str(layer)] = learn_linear_map(hidden_states, input_embeddings-hidden_states, fit_intercept)
                    self.lm_head_maps[str(layer)] = learn_linear_map(hidden_states, lm_head_weights-hidden_states, fit_intercept)
                else:
                    self.embedding_maps[str(layer)] = learn_linear_map(hidden_states, input_embeddings, fit_intercept)
                    self.lm_head_maps[str(layer)] = learn_linear_map(hidden_states, lm_head_weights, fit_intercept)

                all_hidden_states[layer] = None

    def fit_on_dataset(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            dataset: Union[List[str], Dataset, IterableDataset, "datasets.Dataset"],
    ) -> None:
        """
        Learns transformations that map representations from every layer to the embedding/lm_head spaces,
        by fitting a transformation from intermediate model representations of vocabulary tokens, as computed in
        the dataset's examples, to the corresponding token rows in the embedding and lm_head matrices.

        Args:
            model (PreTrainedModel):
                Model to extract embeddings and intermediate representations from.
            tokenizer (PreTrainedTokenizer):
                Tokenizer to use when extracting embeddings.
            dataset (Union[List[str], Dataset, IterableDataset, "datasets.Dataset"]):
                Data to train transformations on.
        """
        raise NotImplementedError("Fine-tuning linear translators on a dataset is not implemented.")

    def to_embedding(self, representations: torch.Tensor, layer_index: int, **kwargs) -> torch.Tensor:
        """
        Transforms the given representations to the embedding space.

        Args:
            representations (torch.Tensor): The intermediate representations to transform.
            layer_index (int): The index of the model layer the representations were extracted from.

        Returns:
            torch.Tensor: The transformed representations in embedding space.
        """
        if self.embedding_maps is None or str(layer_index) not in self.embedding_maps:
            raise ValueError("The mapping has not been trained yet. Call fit first.")
        if not isinstance(representations, torch.Tensor):
            raise TypeError("Representations must be torch.Tensor.")

        with torch.no_grad():
            result = self.embedding_maps[str(layer_index)](representations)
            try:
                if self.do_residual:
                    result = result + representations
            except:
                pass
        return result

    def to_lm_head(self, representations: torch.Tensor, layer_index: int, **kwargs) -> torch.Tensor:
        """
        Transforms the given representations to the lm_head space.

        Args:
            representations (torch.Tensor): The intermediate representations to transform.
            layer_index (int): The index of the model layer the representations were extracted from.

        Returns:
            torch.Tensor: The transformed representations in lm_head space.
        """
        if self.lm_head_maps is None or str(layer_index) not in self.lm_head_maps:
            raise ValueError("The mapping has not been trained yet. Call fit first.")
        if not isinstance(representations, torch.Tensor):
            raise TypeError("Representations must be torch.Tensor.")

        with torch.no_grad():
            result = self.lm_head_maps[str(layer_index)](representations)
            try:
                if self.do_residual:
                    result = result + representations
            except:
                pass
        return result


class ProcrustesLayer(nn.Module):
    def __init__(self, W, b_out=None, b_in=None, alpha_out=None, alpha_in=None, normalize=False):
        super().__init__()
        self.W = nn.Parameter(torch.tensor(W, dtype=torch.float32))

        self.b_out = None if b_out is None else nn.Parameter(torch.tensor(b_out, dtype=torch.float32))
        self.b_in = None if b_in is None else nn.Parameter(torch.tensor(b_in, dtype=torch.float32))

        self.alpha_out = None if alpha_out is None else nn.Parameter(alpha_out.clone().detach().to(torch.float32))
        self.alpha_in = None if alpha_in is None else nn.Parameter(alpha_out.clone().detach().to(torch.float32))

        self.normalize = normalize

    def forward(self, h):
        x = h

        if self.b_in is not None:
            x = x - self.b_in

        if self.normalize:
            # x = (x - x.mean()) / x_sigma
            # x = x / x_sigma
            # x = x / x.std()
            x_rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
            x = x / x_rms

            if self.alpha_in is not None:
                x *= self.alpha_in

        # apply procrustes
        y = torch.matmul(x, self.W.t())

        if self.normalize:  # undo normalization
            if self.alpha_out is not None:
                y *= self.alpha_out
                # y *= self.alpha_out / x_rms

        if self.b_out is not None:
            y = y + self.b_out

        return y


class ProcrustesRepresentationTranslators(RepresentationTranslators):
    """
    Transforms intermediate model representations to the embedding and lm_head spaces
    using procrustes matrices.
    """

    def fit_on_tokens(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            token_ids: Iterable[int] = None,
            prompt: str = "{target}",
            prompt_target: str = "{target}",
            translation_layers: Union[int, List[int]] = None,
            normalize: bool = False,
            normalize_embeddings: bool = False,
            layer_batch_size: int = 8,
            batch_size: int = 128,
            min_word_len: int = None,
            alpha_only: bool = False,
            space_prefixed_only: bool = False,
    ) -> None:

        input_embeddings = model.get_input_embeddings().weight.detach().cpu()
        lm_head_weights = model.get_output_embeddings().weight.detach().cpu()
        # some models have embeddings for special tokens that aren't included in the "regular" vocabulary
        input_embeddings = input_embeddings[:tokenizer.vocab_size]
        lm_head_weights = lm_head_weights[:tokenizer.vocab_size]

        if token_ids is not None and len(token_ids) != tokenizer.vocab_size:
            input_embeddings = input_embeddings[token_ids]
            lm_head_weights = lm_head_weights[token_ids]
        elif (min_word_len is not None) or alpha_only or space_prefixed_only:
            token_ids, tokens_filter = \
                self._filter_tokens(tokenizer, np.arange(tokenizer.vocab_size), alpha_only, min_word_len)
            input_embeddings = input_embeddings[tokens_filter]
            lm_head_weights = lm_head_weights[tokens_filter]

        n_layers = model.config.num_hidden_layers
        layer_batch_size = n_layers if layer_batch_size is None else layer_batch_size
        if isinstance(translation_layers, int):
            translation_layers = [translation_layers]
        elif translation_layers is None:
            translation_layers = list(range(1, n_layers+1))
        n_batches = int(math.ceil(len(translation_layers) / layer_batch_size))
        for batch_i in tqdm(range(n_batches),
                                  desc="Fitting maps to embedding and lm_head spaces", unit="Layer batch"):
            layers_to_learn = translation_layers[batch_i*layer_batch_size:(batch_i+1)*layer_batch_size]
            all_hidden_states = extract_vocab_hidden_states(model, tokenizer, token_ids, prompt, prompt_target,
                                                            batch_size, layers_to_learn)
            for layer in tqdm(layers_to_learn, total=len(layers_to_learn), unit="layers", desc="Fitting maps..."):
                hidden_states = all_hidden_states[layer]

                # Learn orthogonal procrustes mappings
                h_bias = hidden_states.mean(dim=0)
                u_bias = lm_head_weights.mean(dim=0)
                e_bias = input_embeddings.mean(dim=0)

                curr_h = hidden_states - h_bias
                curr_u = lm_head_weights - u_bias
                curr_e = input_embeddings - e_bias

                if normalize:
                    h_rms = torch.sqrt(torch.mean(curr_h ** 2, dim=-1, keepdim=True))
                    u_rms = torch.sqrt(torch.mean(curr_u ** 2, dim=-1, keepdim=True))
                    # u_rms_mean = u_rms.mean()
                    # e_rms = torch.sqrt(torch.mean(curr_e ** 2, dim=-1, keepdim=True))
                    u_sd = lm_head_weights.std(dim=-1).mean()
                    # e_sd = input_embeddings.std(dim=-1).mean()
                    # u_rms_gamma = model.model.norm.weight.detach().cpu()
                    u_rms_gamma = 1


                    # # SD normalization
                    # curr_h_norm = curr_h / curr_h.std(-1).unsqueeze(1)
                    # curr_u_norm = curr_u / curr_u.std(-1).unsqueeze(1)
                    # curr_e_norm = curr_e / curr_e.std(-1).unsqueeze(1)

                    # RMS normalization
                    curr_h_norm = curr_h / h_rms
                    curr_u_norm = curr_u
                    curr_e_norm = curr_e
                    # curr_u_norm = curr_u / u_rms
                    # curr_e_norm = curr_e / e_rms

                if normalize:
                    M_u = orthogonal_procrustes((u_rms_gamma*curr_h_norm).cpu().to(torch.float32).numpy(), (curr_u_norm).cpu().to(torch.float32).numpy(), lapack_driver="gesdd", scale=False, translate=False)
                else:
                    M_u = orthogonal_procrustes((curr_h).cpu().to(torch.float32).numpy(), (curr_u).cpu().to(torch.float32).numpy(), lapack_driver="gesdd", scale=False, translate=False)
                M_u = torch.tensor(M_u.t.T)

                if normalize and normalize_embeddings:
                    M_e = orthogonal_procrustes((curr_h_norm).cpu().to(torch.float32).numpy(), (curr_e_norm).cpu().to(torch.float32).numpy(), lapack_driver="gesdd", scale=False, translate=False)
                else:
                    M_e = orthogonal_procrustes((curr_h).cpu().to(torch.float32).numpy(), (curr_e).cpu().to(torch.float32).numpy(), lapack_driver="gesdd", scale=False, translate=False)
                M_e = torch.tensor(M_e.t.T)

                # Set learned parameters into torch modules
                if normalize:
                    # RMS normalization
                    self.lm_head_maps[str(layer)] = ProcrustesLayer(M_u, alpha_out=u_sd, normalize=True)
                    self.embedding_maps[str(layer)] = ProcrustesLayer(M_e)
                    # # SD normalization
                    # self.lm_head_maps[str(layer)] = ProcrustesLayer(M_u, alpha_out=u_sd, normalize=True)
                    # self.embedding_maps[str(layer)] = ProcrustesLayer(M_e, alpha_out=e_sd, normalize=normalize_embeddings)
                else:
                    self.lm_head_maps[str(layer)] = ProcrustesLayer(M_u)
                    self.embedding_maps[str(layer)] = ProcrustesLayer(M_e)
                # self.lm_head_maps[layer] = ProcrustesLayer(M_u, u_bias)
                # self.embedding_maps[layer] = ProcrustesLayer(M_e, e_bias)

        if len(translation_layers) == 1:
            self.lm_head_maps["all"] = self.lm_head_maps[str(translation_layers[0])]
            self.embedding_maps["all"] = self.embedding_maps[str(translation_layers[0])]

        return

    def fit_on_dataset(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            dataset: Union[List[str], Dataset, IterableDataset, "datasets.Dataset"],
    ) -> None:
        """
        Learns transformations that map representations from every layer to the embedding/lm_head spaces,
        by fitting a transformation from intermediate model representations of vocabulary tokens, as computed in
        the dataset's examples, to the corresponding token rows in the embedding and lm_head matrices.

        Args:
            model (PreTrainedModel):
                Model to extract embeddings and intermediate representations from.
            tokenizer (PreTrainedTokenizer):
                Tokenizer to use when extracting embeddings.
            dataset (Union[List[str], Dataset, IterableDataset, "datasets.Dataset"]):
                Data to train transformations on.
        """
        raise NotImplementedError("Fine-tuning linear translators on a dataset is not implemented.")

    def to_embedding(self, representations: torch.Tensor, layer_index: int = None, **kwargs) -> torch.Tensor:
        """
        Transforms the given representations to the embedding space.

        Args:
            representations (torch.Tensor): The intermediate representations to transform.
            layer_index (int): The index of the model layer the representations were extracted from.

        Returns:
            torch.Tensor: The transformed representations in embedding space.
        """
        if self.embedding_maps is None \
                or ((layer_index is None and "all" not in self.embedding_maps)
                    and (layer_index is not None) and str(layer_index) not in self.embedding_maps):
            raise ValueError("The mapping has not been trained yet. Call fit first.")
        if not isinstance(representations, torch.Tensor):
            raise TypeError("Representations must be torch.Tensor.")

        with torch.no_grad():
            if "all" in self.embedding_maps:
                result = self.embedding_maps["all"](representations)
            else:
                result = self.embedding_maps[str(layer_index)](representations)
        return result

    def to_lm_head(self, representations: torch.Tensor, layer_index: int = None, **kwargs) -> torch.Tensor:
        """
        Transforms the given representations to the lm_head space.

        Args:
            representations (torch.Tensor): The intermediate representations to transform.
            layer_index (int): The index of the model layer the representations were extracted from.

        Returns:
            torch.Tensor: The transformed representations in lm_head space.
        """
        if self.lm_head_maps is None \
                or ((layer_index is None and "all" not in self.lm_head_maps)
                    and (layer_index is not None) and str(layer_index) not in self.lm_head_maps):
            raise ValueError("The mapping has not been trained yet. Call fit first.")
        if not isinstance(representations, torch.Tensor):
            raise TypeError("Representations must be torch.Tensor.")

        with torch.no_grad():
            if "all" in self.lm_head_maps:
                result = self.lm_head_maps["all"](representations)
            else:
                result = self.lm_head_maps[str(layer_index)](representations)
        return result


class MLPRepresentationTranslators(LinearRepresentationTranslators):
    """
    Transforms intermediate model representations to the embedding and lm_head spaces
    using MLPs.
    """

    def fit_on_tokens(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            token_ids: Iterable[int] = None,
            prompt: str = "{target}",
            prompt_target: str = "{target}",
            batch_size: int = 128,
            layer_batch_size: int = 8,
            loss_func: str = "mse",
            lr_schedule: str = "linear",
            lr: float = 0.001,
            mlp_batch_size: int = 256,
            weight_decay: float = 0.1,
            num_epochs: int = 10,
            gradient_accumulation_steps: int = 1,
            min_word_len: int = None,
            alpha_only: bool = True,
            space_prefixed_only: bool = False,
    ) -> None:
        """
        Learns transformations that map representations from every layer to the embedding/lm_head spaces,
        by fitting a transformation from intermediate model representations of single tokens from the vocabulary,
        as extracted using the given prompt, to the respective rows in the embedding and lm_head matrices.

        Args:
            model (PreTrainedModel):
                Model to extract embeddings and intermediate representations from.
            tokenizer (PreTrainedTokenizer):
                Tokenizer to use when extracting embeddings.
            token_ids (Iterable[int]):
                The list of token_ids to use as training data.
                If not given, will train over the entire vocabulary.
            prompt (str):
                The prompt to use when extracting token representations, where the parameter prompt_target
                will be replaced with the target token.
            prompt_target (str):
                The placeholder for the target token in the prompt for extracting hidden states.
            batch_size (int):
                Batch size to use when extracting token representations.
            layer_batch_size (int):
                Number of layers to compute translators for in parallel.
        """

        input_embeddings = model.get_input_embeddings().weight.detach().cpu()
        lm_head_weights = model.get_output_embeddings().weight.detach().cpu()
        # some models have embeddings for special tokens that aren't included in the "regular" vocabulary
        input_embeddings = input_embeddings[:tokenizer.vocab_size]
        lm_head_weights = lm_head_weights[:tokenizer.vocab_size]

        if token_ids is not None and len(token_ids) != tokenizer.vocab_size:
            input_embeddings = input_embeddings[token_ids]
            lm_head_weights = lm_head_weights[token_ids]
        else:
            token_ids, tokens_filter = \
                self._filter_tokens(tokenizer, np.arange(tokenizer.vocab_size), alpha_only, min_word_len)
            input_embeddings = input_embeddings[tokens_filter]
            lm_head_weights = lm_head_weights[tokens_filter]

        n_layers = model.config.num_hidden_layers
        layer_batch_size = n_layers if layer_batch_size is None else layer_batch_size
        for start_layer_i in tqdm(range(1, n_layers+1, layer_batch_size), desc="Fitting maps to embedding and lm_head spaces", unit="Layer batch"):
            layers_to_learn = None if layer_batch_size is None else \
                list(range(start_layer_i, min(start_layer_i + layer_batch_size, n_layers+1)))
            all_hidden_states = extract_vocab_hidden_states(model, tokenizer, token_ids, prompt, prompt_target, batch_size, layers_to_learn)
            for layer in tqdm(layers_to_learn, total=len(layers_to_learn), unit="layers", desc="Fitting maps..."):
                hidden_states = all_hidden_states[layer]
                self.embedding_maps[str(layer)] = \
                    learn_ffn(hidden_states, input_embeddings, batch_size=mlp_batch_size, lr=lr, weight_decay=weight_decay, loss_func=loss_func, lr_schedule=lr_schedule, num_epochs=num_epochs, gradient_accumulation_steps=gradient_accumulation_steps)
                self.lm_head_maps[str(layer)] = \
                    learn_ffn(hidden_states, lm_head_weights, batch_size=mlp_batch_size, lr=lr, weight_decay=weight_decay, loss_func=loss_func, lr_schedule=lr_schedule, num_epochs=num_epochs, gradient_accumulation_steps=gradient_accumulation_steps)

                all_hidden_states[layer] = None

    def fit_on_dataset(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            dataset: Union[List[str], Dataset, IterableDataset, "datasets.Dataset"],
    ) -> None:
        """
        Learns transformations that map representations from every layer to the embedding/lm_head spaces,
        by fitting a transformation from intermediate model representations of vocabulary tokens, as computed in
        the dataset's examples, to the corresponding token rows in the embedding and lm_head matrices.

        Args:
            model (PreTrainedModel):
                Model to extract embeddings and intermediate representations from.
            tokenizer (PreTrainedTokenizer):
                Tokenizer to use when extracting embeddings.
            dataset (Union[List[str], Dataset, IterableDataset, "datasets.Dataset"]):
                Data to train transformations on.
        """
        raise NotImplementedError("Fine-tuning MLP translators on a dataset is not implemented.")
