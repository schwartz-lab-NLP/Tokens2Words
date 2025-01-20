import torch
import argparse
from datasets import load_dataset

MODELS_MAP = {
    "Llama-3-8B": "meta-llama/Meta-Llama-3-8B",
    "Mistral-7B": "mistralai/Mistral-7B-v0.1",
    "Yi-6B": "01-ai/Yi-6B",
    "gemma-2-9B": "google/gemma-2-9b",
    'Llama-2-7B': 'meta-llama/Llama-2-7b-hf',
    "pythia-6.9b": "EleutherAI/pythia-6.9B",
}

# Map of string names to torch dtypes for loading models
DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "int8": torch.int8,
}


# Define the custom type for argparse
def dtype_arg(value):
    try:
        return DTYPE_MAP[value]
    except KeyError:
        raise argparse.ArgumentTypeError(f"Invalid dtype: {value}. Choose from {list(DTYPE_MAP.keys())}")


def load_dataset_for_analysis(dataset_name: str):
    if dataset_name.lower() == "wikitext":
        return load_dataset('wikitext', 'wikitext-2-raw-v1', trust_remote_code=True)
    return NotImplementedError(f"Dataset loader for {dataset_name} not implemented for analyses.")