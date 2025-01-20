"""Provides a class for mapping transformer hidden states to logits (and vice versa).
Example:

from standalone_logit_lens import LogitLens, ReverseLogitLens

model = AutoModelForCausalLM.from_pretrained(model_name).to(device).to(dtype)
lens = LogitLens.from_model(model).to(device).to(dtype)
reverse_lens = ReverseLogitLens.from_model(model).to(device).to(dtype)

hidden_state = ...
result = lens(hidden_state, layer_index)  # layer_index is not really used, you can pass whatever
"""

import abc
import logging

import copy
from typing import Union

import torch
from torch import nn
import torch.nn.functional as F

import transformers
from transformers import models
from transformers import PreTrainedModel


Model = Union[PreTrainedModel]
Norm = Union[
    nn.LayerNorm,
    models.llama.modeling_llama.LlamaRMSNorm,
    models.gemma.modeling_gemma.GemmaRMSNorm,
    models.gemma2.modeling_gemma2.Gemma2RMSNorm,
    nn.Module,
]


def get_unembedding_matrix(model: Model) -> nn.Linear:
    """The final linear tranformation from the model hidden state to the output."""
    if isinstance(model, PreTrainedModel):
        unembed = model.get_output_embeddings()
        if not isinstance(unembed, nn.Linear):
            raise ValueError("We currently only support linear unemebdings")
        return unembed
    else:
        raise ValueError(f"Model class {type(model)} not recognized!")


def get_embedding_matrix(model: nn.Module) -> nn.Embedding:
    """The initial embedding matrix from the input tokens to the model hidden state."""
    if isinstance(model, PreTrainedModel):
        embed = model.get_input_embeddings()
        if not isinstance(embed, nn.Embedding):
            raise ValueError("We currently only support embedding matrices")
        return embed
    else:
        raise ValueError(f"Model class {type(model)} not recognized!")


def get_final_norm(model: Model) -> Norm:
    """Get the final norm from a model.

    This isn't standardized across models, so this will need to be updated as
    we add new models.
    """

    if not hasattr(model, "base_model"):
        raise ValueError("Model does not have a `base_model` attribute.")

    base_model = model.base_model
    if isinstance(base_model, models.opt.modeling_opt.OPTModel):
        final_layer_norm = base_model.decoder.final_layer_norm
    elif isinstance(base_model, models.gpt_neox.modeling_gpt_neox.GPTNeoXModel):
        final_layer_norm = base_model.final_layer_norm
    elif isinstance(
        base_model,
        (
            models.bloom.modeling_bloom.BloomModel,
            models.gpt2.modeling_gpt2.GPT2Model,
            models.gpt_neo.modeling_gpt_neo.GPTNeoModel,
            models.gptj.modeling_gptj.GPTJModel,
        ),
    ):
        final_layer_norm = base_model.ln_f
    elif isinstance(base_model, models.llama.modeling_llama.LlamaModel):
        final_layer_norm = base_model.norm
    elif isinstance(base_model, models.mistral.modeling_mistral.MistralModel):
        final_layer_norm = base_model.norm
    elif isinstance(base_model, models.t5.modeling_t5.T5ForConditionalGeneration):
        # For T5, use the LayerNorm from the last decoder block, before the feed-forward layer.
        final_layer_norm = base_model.decoder.block[-1].layer[1].layer_norm
    else:
        raise NotImplementedError(f"Unknown model type {type(base_model)}")

    if final_layer_norm is None:
        raise ValueError("Model does not have a final layer norm.")

    assert isinstance(final_layer_norm, Norm.__args__)  # type: ignore

    return final_layer_norm


class Unembed(nn.Module):
    """Module that maps transformer hidden states to logits (and vice versa)."""

    final_norm: Norm
    unembedding: nn.Linear

    def __init__(
        self,
        model: Model,
    ):
        """Initialize unmebed.

        Args:
            model: A HuggingFace model from which to extract the unembedding matrix.
        """
        super().__init__()
        final_norm = get_final_norm(model)
        unembedding_matrix = get_unembedding_matrix(model)

        self.final_norm = copy.deepcopy(final_norm)
        self.unembedding = copy.deepcopy(unembedding_matrix)

        # In general we don't want to finetune the unembed operation.
        self.requires_grad_(False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Convert hidden states into logits."""
        return self.unembedding(self.final_norm(h))


class Reembed(nn.Module):
    """Module that maps transformer hidden states to logits (and vice versa)."""
    embedding: torch.Tensor

    def __init__(
        self,
        model: Model,
        distance_metric: str = "logits",
    ):
        """Initialize unmebed.

        Args:
            model: A HuggingFace model from which to extract the unembedding matrix.
        """
        super().__init__()
        embedding_matrix = get_embedding_matrix(model)

        self.embedding = copy.deepcopy(embedding_matrix.weight.data)

        self.distance_metric = distance_metric

        # In general we don't want to finetune the unembed operation.
        self.requires_grad_(False)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Convert hidden states into logits."""
        
        if self.distance_metric == 'logits':
            logits = torch.matmul(hidden_state, self.embedding.T).squeeze(0)

        elif self.distance_metric == 'cosine':
            # Normalize E and h
            E_normalized = F.normalize(self.embedding, p=2, dim=-1)
            h_normalized = F.normalize(hidden_state, p=2, dim=-1)

            # Compute cosine similarity
            logits = torch.matmul(h_normalized, E_normalized.T).squeeze(0)

        elif self.distance_metric == 'euclidean':
            # Compute Euclidean distance
            distances = torch.cdist(hidden_state, self.embedding, p=2).squeeze(0)

            # Convert distances to logits (negative distance for logits-like values)
            logits = -distances

        else:  # Compute regular dot-product as a similarity measure
            logits = torch.matmul(hidden_state, self.embedding.T).squeeze(0)
        return logits


class ReverseLens(abc.ABC, nn.Module):
    """Abstract base class for all Lens."""

    reembed: Reembed

    def __init__(self, reembed: Reembed):
        """Create a Lens.

        Args:
            unembed: The unembed operation to use.
        """
        super().__init__()

        self.reembed = reembed

    @abc.abstractmethod
    def forward(self, h: torch.Tensor, idx: int) -> torch.Tensor:
        """Decode hidden states into logits."""
        ...


class ReverseLogitLens(ReverseLens):
    """Reembeds the residual stream into logits."""

    reembed: Reembed

    def __init__(
        self,
        reembed: Reembed,
    ):
        """Create a Reverse Logit Lens.

        Args:
            reembed: The reembed operation to use.
        """
        super().__init__(reembed)

    @classmethod
    def from_model(
        cls,
        model: PreTrainedModel,
    ) -> "ReverseLogitLens":
        """Create a ReverseLogitLens from a pretrained model.

        Args:
            model: A pretrained model from the transformers library you wish to inspect.
        """
        reembed = Reembed(model)
        return cls(reembed)

    def forward(self, h: torch.Tensor, idx: int) -> torch.Tensor:
        """Decode a hidden state into logits.

        Args:
            h: The hidden state to decode.
            idx: the layer of the transformer these hidden states come from.
        """
        del idx
        return self.reembed.forward(h)


class Lens(abc.ABC, nn.Module):
    """Abstract base class for all Lens."""

    unembed: Unembed

    def __init__(self, unembed: Unembed):
        """Create a Lens.

        Args:
            unembed: The unembed operation to use.
        """
        super().__init__()

        self.unembed = unembed

    @abc.abstractmethod
    def forward(self, h: torch.Tensor, idx: int) -> torch.Tensor:
        """Decode hidden states into logits."""
        ...


class LogitLens(Lens):
    """Unembeds the residual stream into logits."""

    unembed: Unembed

    def __init__(
        self,
        unembed: Unembed,
    ):
        """Create a Logit Lens.

        Args:
            unembed: The unembed operation to use.
        """
        super().__init__(unembed)

    @classmethod
    def from_model(
        cls,
        model: PreTrainedModel,
    ) -> "LogitLens":
        """Create a LogitLens from a pretrained model.

        Args:
            model: A pretrained model from the transformers library you wish to inspect.
        """
        unembed = Unembed(model)
        return cls(unembed)

    def forward(self, h: torch.Tensor, idx: int) -> torch.Tensor:
        """Decode a hidden state into logits.

        Args:
            h: The hidden state to decode.
            idx: the layer of the transformer these hidden states come from.
        """
        del idx
        return self.unembed.forward(h)

