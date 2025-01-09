from tqdm import tqdm
from typing import Iterable, List, Union
from transformers import PreTrainedModel, PreTrainedTokenizer
import torch
from torch import nn
from accelerate.utils import set_seed
from sklearn.linear_model import LinearRegression
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import LambdaLR


def extract_token_i_hidden_states(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        inputs: Union[str, List[str]],
        token_idx_to_extract: int = -1,
        batch_size: int = 1,
        layers_to_extract: List[int] = None,
        return_dict: bool = True,
        verbose: bool = True,
) -> torch.Tensor:
    device = model.device
    model.eval()

    if isinstance(inputs, str):
        inputs = [inputs]

    if layers_to_extract is None:
        layers_to_extract = list(range(1, model.config.num_hidden_layers + 1))  # extract all but initial embeddings
    all_hidden_states = {layer: [] for layer in layers_to_extract}

    with torch.no_grad():
        for i in tqdm(range(0, len(inputs), batch_size), desc="Extracting hidden states", unit="batch", disable=not verbose):
            input_ids = tokenizer(inputs[i:i+batch_size], return_tensors="pt", return_attention_mask=False)['input_ids']
            try:
                outputs = model(input_ids.to(device), output_hidden_states=True)
            except:
                import pdb; pdb.set_trace()
                # from transformers import AutoModelForCausalLM
                # model2 = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", torch_dtype=torch.bfloat16).to(device)
            for input_i in range(len(input_ids)):
                for layer in layers_to_extract:
                    hidden_states = outputs.hidden_states[layer]
                    all_hidden_states[layer].append(hidden_states[:, token_idx_to_extract, :].detach().cpu())
    for layer in all_hidden_states:
        all_hidden_states[layer] = torch.concat(all_hidden_states[layer], dim=0)

    if not return_dict:
        all_hidden_states = torch.concat([all_hidden_states[layer] for layer in layers_to_extract], dim=0)

    return all_hidden_states


def extract_vocab_hidden_states(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        tokens_ids_to_extract: Iterable[int] = None,
        prompt: str = "{target}",
        prompt_target: str = "{target}",
        batch_size: int = 128,
        layers_to_extract: List[int] = None
) -> torch.Tensor:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    if layers_to_extract is None:
        layers_to_extract = list(range(1, model.config.num_hidden_layers + 1))  # extract all but initial embeddings
    all_hidden_states = {layer: [] for layer in layers_to_extract}
    tokens_ids_to_extract = tokens_ids_to_extract if tokens_ids_to_extract is not None else range(tokenizer.vocab_size)
    tokens_to_extract = [tokenizer.decode(tok_id) for tok_id in tokens_ids_to_extract]

    # add pad token if necessary
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with torch.no_grad():
        for i in tqdm(range(0, len(tokens_to_extract), batch_size), desc="Extracting hidden states", unit="batch"):
            prompts = [prompt.replace(prompt_target, target) for target in tokens_to_extract[i:i+batch_size]]
            input_ids = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left")["input_ids"]
            # input_ids = tokenizer(prompts, return_tensors="pt")["input_ids"]
            outputs = model(input_ids.to(device), output_hidden_states=True)
            for layer in layers_to_extract:
                hidden_states = outputs.hidden_states[layer]
                all_hidden_states[layer].append(hidden_states[:, -1, :].detach().cpu())

    for layer in all_hidden_states:
        all_hidden_states[layer] = torch.concat(all_hidden_states[layer], dim=0)

    return all_hidden_states


def get_vocab_tokens(tokenizer: PreTrainedTokenizer, min_word_len: int = None):
    vocab_size = tokenizer.vocab_size
    tokens = list(range(vocab_size))
    if min_word_len:
        tokens_str = [tokenizer.decode(i) for i in tokens]
        tokens_len = [len(x) for x in tokens_str]
        tokens = [tok for tok, tok_len in zip(tokens, tokens_len) if tok_len >= min_word_len]
    return tokens


def learn_linear_map(X: torch.Tensor, Y: torch.Tensor, fit_intercept=False):
    input_dtype = X.dtype
    linear_reg = LinearRegression(fit_intercept=fit_intercept).fit(X.cpu().to(float).numpy(), Y.cpu().to(float).numpy())
    linear_map = nn.Linear(X.size(1), Y.size(1), bias=fit_intercept)
    with torch.no_grad():
        linear_map.weight.data = torch.Tensor(linear_reg.coef_.T)
        if fit_intercept:
            linear_map.bias.data = torch.Tensor(linear_reg.intercept_)
    linear_map = linear_map.to(input_dtype)
    return linear_map


def train_model(
    model,
    dataloader,
    optimizer,
    loss_func="mse",
    scheduler=None,
    num_epochs=5,
    gradient_accumulation_steps=1,
    max_grads_norm=1.0,
):
    """
    Trains a two-layer MLP to map hidden states from X to Y.

    Parameters:
        X (torch.Tensor): Input tensor of shape (N, D).
        Y (torch.Tensor): Target tensor of shape (N, D).
        activation_func (nn.Module): Activation function for the hidden layer. Default is SiLU.
        lr (float): Learning rate. Default is 0.001.
        weight_decay (float): Weight decay for the optimizer. Default is 0.0.
        loss_func (str): Loss function to use ('mse', 'huber', 'cosine'). Default is 'mse'.
        lr_schedule (str): Learning rate schedule. Default is 'linear'.
        num_epochs (int): Number of training epochs. Default is 20.
        batch_size (int): Batch size for DataLoader. Default is 32.
        gradient_accumulation_steps (int): Number of steps to accumulate gradients. Default is 1.

    Returns:
        nn.Module: Trained MLP model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Select loss function
    if loss_func == "mse":
        criterion = nn.MSELoss()
    elif loss_func == "huber":
        criterion = nn.HuberLoss()
    elif loss_func == "cosine":
        criterion = nn.CosineEmbeddingLoss()
    else:
        raise ValueError("Unsupported loss function. Choose from 'mse', 'huber', or 'cosine'.")

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i, (x_batch, y_batch) in enumerate(dataloader):
            outputs = model(x_batch.to(device))
            if loss_func == "cosine":
                # Cosine loss requires an additional target tensor of 1s
                loss = criterion(outputs, y_batch.to(device), torch.ones(x_batch.size(0)))
            else:
                loss = criterion(outputs, y_batch.to(device))

            loss = loss / gradient_accumulation_steps
            loss.backward()

            if max_grads_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), max_grads_norm)

            if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()
                if scheduler:
                    scheduler.step()

            epoch_loss += loss.item() * gradient_accumulation_steps

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.6f}")

    return model.cpu()


def learn_mlp(
    X: torch.Tensor, Y: torch.Tensor,
    activation_func=nn.SiLU,
    batch_size=128,
    lr=0.001,
    weight_decay=0.0,
    loss_func="mse",
    lr_schedule="linear",
    expansion_alpha=1.0,
    num_epochs=5,
    gradient_accumulation_steps=1,
    max_grads_norm=1.0,
):
    """
    Trains a two-layer MLP to map hidden states from X to Y.

    Parameters:
        X (torch.Tensor): Input tensor of shape (N, D).
        Y (torch.Tensor): Target tensor of shape (N, D).
        activation_func (nn.Module): Activation function for the hidden layer. Default is SiLU.
        lr (float): Learning rate. Default is 0.001.
        weight_decay (float): Weight decay for the optimizer. Default is 0.0.
        loss_func (str): Loss function to use ('mse', 'huber', 'cosine'). Default is 'mse'.
        lr_schedule (str): Learning rate schedule. Default is 'linear'.
        num_epochs (int): Number of training epochs. Default is 20.
        batch_size (int): Batch size for DataLoader. Default is 32.
        gradient_accumulation_steps (int): Number of steps to accumulate gradients. Default is 1.

    Returns:
        nn.Module: Trained MLP model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X.shape[1]
    hidden_dim = int(input_dim * expansion_alpha)
    output_dim = Y.shape[1]
    model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                activation_func(),
                nn.Linear(hidden_dim, output_dim)
    ).to(device)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # DataLoader setup
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Learning rate scheduler
    if lr_schedule == "linear":
        total_steps = (len(dataloader) * num_epochs) // gradient_accumulation_steps
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 - step / total_steps)
    else:
        scheduler = None

    return train_model(
        model,
        dataloader,
        optimizer,
        loss_func=loss_func,
        scheduler=scheduler,
        num_epochs=num_epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grads_norm=max_grads_norm,
    )


class FFN(nn.Module):
    def __init__(self, input_dim):
        super(FFN, self).__init__()
        self.gate_proj = nn.Linear(input_dim, input_dim)
        self.activation = nn.SiLU()
        self.map_proj = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        return (self.activation(self.gate_proj(x)) * x) + self.map_proj(x)


def learn_ffn(
    X: torch.Tensor, Y: torch.Tensor,
    activation_func=nn.SiLU,
    batch_size=128,
    lr=0.001,
    weight_decay=0.0,
    loss_func="mse",
    lr_schedule="linear",
    num_epochs=5,
    gradient_accumulation_steps=1,
    max_grads_norm=1.0,
):
    """
    Trains a two-layer MLP to map hidden states from X to Y.

    Parameters:
        X (torch.Tensor): Input tensor of shape (N, D).
        Y (torch.Tensor): Target tensor of shape (N, D).
        activation_func (nn.Module): Activation function for the hidden layer. Default is SiLU.
        lr (float): Learning rate. Default is 0.001.
        weight_decay (float): Weight decay for the optimizer. Default is 0.0.
        loss_func (str): Loss function to use ('mse', 'huber', 'cosine'). Default is 'mse'.
        lr_schedule (str): Learning rate schedule. Default is 'linear'.
        num_epochs (int): Number of training epochs. Default is 20.
        batch_size (int): Batch size for DataLoader. Default is 32.
        gradient_accumulation_steps (int): Number of steps to accumulate gradients. Default is 1.

    Returns:
        nn.Module: Trained MLP model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X.shape[1]
    model = FFN(input_dim).to(device)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # DataLoader setup
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Learning rate scheduler
    if lr_schedule == "linear":
        total_steps = (len(dataloader) * num_epochs) // gradient_accumulation_steps
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 - step / total_steps)
    else:
        scheduler = None

    return train_model(
        model,
        dataloader,
        optimizer,
        loss_func=loss_func,
        scheduler=scheduler,
        num_epochs=num_epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grads_norm=max_grads_norm,
    )
