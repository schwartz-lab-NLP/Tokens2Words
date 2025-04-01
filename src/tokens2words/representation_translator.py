import os
from tqdm import tqdm
import math
from abc import ABC, abstractmethod
from typing import Iterable, Union, List, Tuple, Dict
from transformers import PreTrainedModel, PreTrainedTokenizer
import torch
from torch import nn
from torch.utils.data import Dataset, IterableDataset
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Optional

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
        token_ids_to_extract = [token_id for token_id in np.array(token_ids_to_extract)[tokens_filter]]
        return token_ids_to_extract, tokens_filter


class RMSNormalizer(nn.Module):
    """Module that learns to normalize RMS values of mapped representations.
    Can either learn a matrix transformation or a function that predicts scaling factors."""

    def __init__(self, mode: str = 'matrix'):
        """
        Args:
            mode: Either 'matrix' for learning a transformation matrix W,
                 or 'scalar' for learning a function that predicts scaling factors
        """
        super().__init__()
        if mode not in ['matrix', 'vector', 'scalar']:
            raise ValueError("mode must be either 'matrix', 'vector' or 'scalar'")
        self.mode = mode
        self.W: Optional[nn.Parameter] = None

    def fit(self, source_representations: torch.Tensor, target_representations: torch.Tensor):
        """Learn normalization mapping to match RMS values of target representations."""
        with torch.no_grad():
            if self.mode == 'matrix':
                # Learn matrix W that maps source to target representations
                X = source_representations.cpu().to(torch.float32).numpy()
                y = target_representations.cpu().to(torch.float32).numpy()
                reg = LinearRegression(fit_intercept=False)
                reg.fit(X, y)
                self.W = nn.Parameter(torch.tensor(reg.coef_.T, dtype=torch.float32))
            elif self.mode == 'vector':
                # For each coordinate in the representation, learn a function that predicts
                # its scaling factor based on the input vector
                source_dims = source_representations.shape[-1]
                target_dims = target_representations.shape[-1]
                assert source_dims == target_dims, "Source and target representations must have same dimensionality"

                # Calculate per-coordinate scaling factors
                # Add small epsilon to avoid division by zero
                eps = 1e-8
                source_abs = source_representations + eps
                target_abs = target_representations + eps
                scaling_factors = target_abs / source_abs  # Shape: [batch_size, dims]

                # Learn to predict these scaling factors from input vectors
                X = source_representations.cpu().numpy()
                y = scaling_factors.cpu().numpy()
                reg = LinearRegression(fit_intercept=False)
                reg.fit(X, y)
                self.W = nn.Parameter(torch.tensor(reg.coef_.T, dtype=torch.float32))
            else:  # scalar mode
                # Learn weights to predict scaling factors
                source_rms = torch.sqrt(torch.mean(source_representations ** 2, dim=-1, keepdim=True))
                target_rms = torch.sqrt(torch.mean(target_representations ** 2, dim=-1, keepdim=True))
                eps = 1e-8
                scaling_factors = target_rms / (source_rms + eps)

                # Use linear regression to learn how to predict scaling factors from input vectors
                X = source_representations.cpu().numpy()
                y = scaling_factors.cpu().numpy()
                reg = LinearRegression(fit_intercept=False)
                reg.fit(X, y)
                self.W = nn.Parameter(torch.tensor(reg.coef_.T, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.mode == 'matrix' and self.W is not None:
                return torch.matmul(x, self.W)
            elif self.mode == 'vector' and self.W is not None:
                # Predict per-coordinate scaling factors for each input vector
                scaling_factors = torch.matmul(x, self.W)  # Shape: [batch_size, dims]
                return x * scaling_factors  # Element-wise multiplication
            elif self.mode == 'scalar' and self.W is not None:
                # Predict per-coordinate scaling factors for each input vector
                scaling_factors = torch.matmul(x, self.W)  # Shape: [batch_size, dims]
                return x * scaling_factors  # Scalar multiplication
            return x


class ProcrustesLayer(nn.Module):
    def __init__(self, W, b_out=None, b_in=None, alpha_out=None, alpha_in=None, normalize=False):
        super().__init__()
        self.W = nn.Parameter(W.clone().detach().to(torch.float32))

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
            x_rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
            x = x / x_rms

            if self.alpha_in is not None:
                x *= self.alpha_in

        # apply procrustes
        y = torch.matmul(x, self.W.t())

        if self.normalize:  # undo normalization
            if self.alpha_out is not None:
                y *= self.alpha_out

        if self.b_out is not None:
            y = y + self.b_out

        return y


class EnhancedProcrustesLayer(ProcrustesLayer):
    """Enhanced Procrustes layer with optional pre-normalization and post-mapping RMS normalization."""

    def __init__(self, W, rms_normalizer: Optional[RMSNormalizer] = None,
                 normalize: bool = False, **kwargs):
        super().__init__(W, normalize=normalize, **kwargs)
        self.rms_normalizer = rms_normalizer

    def forward(self, h):
        # First apply regular Procrustes with optional pre-normalization
        x = super().forward(h)
        # Then apply RMS normalization if specified
        if self.rms_normalizer is not None:
            x = self.rms_normalizer(x)
        return x


class LinearRegressionLayer(nn.Module):
    """Linear regression layer with optional pre-normalization and post-mapping RMS normalization."""

    def __init__(self, W, rms_normalizer: Optional[RMSNormalizer] = None,
                 normalize: bool = False):
        super().__init__()
        self.W = nn.Parameter(torch.tensor(W, dtype=torch.float32))
        self.rms_normalizer = rms_normalizer
        self.normalize = normalize

    def forward(self, h):
        x = h

        if self.normalize:
            x_rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True))
            x = x / x_rms

        x = torch.matmul(x, self.W.t())

        if self.rms_normalizer is not None:
            x = self.rms_normalizer(x)

        return x


def _fit_with_options(hidden_states: torch.Tensor, target_vectors: torch.Tensor,
                      normalize: bool = False, post_normalize_mode: Optional[str] = None,
                      use_procrustes: bool = True) -> Tuple[torch.Tensor, Dict, Optional[RMSNormalizer]]:
    """Helper function to fit either Procrustes or linear regression with normalization options."""
    additional_params = dict()

    # Center the data
    h_bias = hidden_states.mean(dim=0)
    t_bias = target_vectors.mean(dim=0)

    curr_h = hidden_states - h_bias
    curr_t = target_vectors - t_bias

    # Pre-normalize if requested
    if normalize:
        h_rms = torch.sqrt(torch.mean(curr_h ** 2, dim=-1, keepdim=True))
        curr_h = curr_h / h_rms
        additional_params["alpha_out"] = torch.sqrt(torch.mean(target_vectors ** 2, dim=-1, keepdim=True)).mean()

    # Learn main transformation
    if use_procrustes:
        M = orthogonal_procrustes(curr_h.cpu().to(torch.float32).numpy(),
                                  curr_t.cpu().to(torch.float32).numpy(),
                                  lapack_driver="gesdd", scale=False, translate=False)
        M = torch.tensor(M.t.T)
    else:
        reg = LinearRegression(fit_intercept=False)
        reg.fit(curr_h.cpu().to(torch.float32).numpy(), curr_t.cpu().to(torch.float32).numpy())
        M = torch.tensor(reg.coef_.T, dtype=torch.float32)

    # Learn post-mapping RMS normalization if requested
    rms_normalizer = None
    if post_normalize_mode is not None:
        rms_normalizer = RMSNormalizer(mode=post_normalize_mode)
        mapped_h = torch.matmul(curr_h, M.t())
        rms_normalizer.fit(mapped_h, curr_t)

    return M, additional_params, rms_normalizer


class ProcrustesRepresentationTranslators(RepresentationTranslators):
    """
    Transforms intermediate model representations to the embedding and lm_head spaces
    using procrustes matrices.
    """
    def __init__(self):
        super(ProcrustesRepresentationTranslators, self).__init__()
        self.use_procrustes = True

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
            post_normalize_mode: Optional[str] = None,
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
            layers_to_learn = translation_layers[batch_i * layer_batch_size:(batch_i + 1) * layer_batch_size]
            all_hidden_states = extract_vocab_hidden_states(model, tokenizer, token_ids, prompt, prompt_target,
                                                            batch_size, layers_to_learn)

            for layer in tqdm(layers_to_learn, total=len(layers_to_learn), unit="layers", desc="Fitting maps..."):
                hidden_states = all_hidden_states[layer]

                # Fit transformations for both lm_head and embedding
                M_u, additional_params, rms_norm_u = _fit_with_options(hidden_states, lm_head_weights,
                                                    normalize, post_normalize_mode, use_procrustes=self.use_procrustes)
                # Set learned parameters into torch modules
                self.lm_head_maps[str(layer)] = EnhancedProcrustesLayer(
                    M_u, rms_normalizer=rms_norm_u, normalize=normalize, **additional_params)

                # Repeat for embeddings
                M_e, additional_params, rms_norm_e = _fit_with_options(hidden_states, input_embeddings,
                                                    normalize_embeddings, post_normalize_mode, use_procrustes=self.use_procrustes)
                self.embedding_maps[str(layer)] = EnhancedProcrustesLayer(
                    M_e, rms_normalizer=rms_norm_e, normalize=normalize_embeddings, **additional_params)

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


class LinearRegressionRepresentationTranslators(ProcrustesRepresentationTranslators):
    """Same as ProcrustesRepresentationTranslators but uses linear regression instead of Procrustes."""

    def __init__(self):
        super(ProcrustesRepresentationTranslators, self).__init__()
        self.use_procrustes = False


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


class RMSRepresentationTranslators(RepresentationTranslators):
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
        return representations

    def to_lm_head(self, representations: torch.Tensor, layer_index: int = None, **kwargs) -> torch.Tensor:
        """
        Transforms the given representations to the lm_head space.

        Args:
            representations (torch.Tensor): The intermediate representations to transform.
            layer_index (int): The index of the model layer the representations were extracted from.

        Returns:
            torch.Tensor: The transformed representations in lm_head space.
        """
        x = representations
        with torch.no_grad():
            x_rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True))
            x = x / x_rms
        return x


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
