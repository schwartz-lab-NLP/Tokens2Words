import os
import torch
from torch import nn
from tqdm import tqdm
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from transformers import get_scheduler
from accelerate import Accelerator
from accelerate.utils import set_seed
from collections import defaultdict
from torch.utils.data import DataLoader
import torch.optim as optim

from ..utils.data_utils import load_lm_dataset, extract_new_words_from_dataset, get_group_texts_func, get_tokenize_func


class EmbeddingCalibrator(nn.Module):
    def __init__(self, hidden_size, lora_r=None, lora_alpha=None, dtype=torch.bfloat16):
        super().__init__()
        self.use_lora = lora_r is not None

        if not self.use_lora:
            self.weight = nn.Parameter(torch.zeros(hidden_size, hidden_size, dtype=dtype))
        else:
            self.lora_scaling = lora_alpha / lora_r if lora_alpha is not None else 1.0
            self.lora_A = nn.Parameter(torch.randn(lora_rank, hidden_size, dtype=dtype) * (1/lora_r))
            self.lora_B = nn.Parameter(torch.zeros(hidden_size, lora_rank, dtype=dtype))

    def forward(self, x):
        if not self.use_lora:
            return x + torch.matmul(x, self.weight.t())
        else:
            # Low-rank adaptation
            lora_out = torch.matmul(x, self.lora_A.t())
            lora_out = torch.matmul(lora_out, self.lora_B.t())
            return x + self.lora_scaling * lora_out


class CalibrationModel(nn.Module):
    def __init__(
            self,
            base_model, lm_head, original_vocab_size, num_new_tokens,
            calibrate_embedding=True, calibrate_lm_head=True, empty_init=False,
            lora_alpha=None, lora_r=None,
            target_loss_weight=0.15, subsequent_loss_weight=0.15,
    ):
        super().__init__()
        self.base_model = base_model
        self.lm_head = lm_head
        self.new_tokens_start = original_vocab_size
        self.new_tokens_end = original_vocab_size + num_new_tokens
        
        self.calibrate_lm_head = calibrate_lm_head
        self.calibrate_embedding = calibrate_embedding
        if not empty_init:
            self.lm_head_calibrator = EmbeddingCalibrator(base_model.config.hidden_size, lora_r, lora_alpha)
            self.embedding_calibrator = EmbeddingCalibrator(base_model.config.hidden_size, lora_r, lora_alpha)

        self.loss_fct = nn.CrossEntropyLoss(reduction="none")
        self.subsequent_tokens_loss_alpha = subsequent_loss_weight
        self.new_tokens_loss_alpha = target_loss_weight
        self.original_tokens_loss_alpha = 1 - self.new_tokens_loss_alpha - self.subsequent_tokens_loss_alpha

    def forward(self, input_ids, labels, attention_mask=None):
        # shift labels by 1 for CLM
        labels = labels[:, 1:].contiguous()
        input_ids = input_ids[:, :-1].contiguous()

        if self.calibrate_embedding:
            E_weights = self.base_model.get_input_embeddings().weight.data
            E_weights = torch.cat((E_weights[:self.new_tokens_start], self.embedding_calibrator(E_weights[self.new_tokens_start:])))
            input_embeddings = E_weights[input_ids]
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.long)
            outputs = self.base_model(inputs_embeds=input_embeddings, attention_mask=attention_mask)
        else:
            with torch.no_grad():
                # Forward pass through the base model
                outputs = self.base_model(input_ids, attention_mask=attention_mask)

        if self.calibrate_lm_head:
            with torch.no_grad():
                lm_head_weights = self.lm_head.weight
                normed_weights = lm_head_weights.clone()
            normed_weights[self.new_tokens_start:self.new_tokens_end] = self.lm_head_calibrator(lm_head_weights[self.new_tokens_start:self.new_tokens_end])
            logits = torch.matmul(outputs['last_hidden_state'], normed_weights.T)
        else:
            if self.calibrate_embedding:
                logits = self.lm_head(outputs['last_hidden_state'])
            else:
                with torch.no_grad():
                    logits = self.lm_head(outputs['last_hidden_state'])

        per_example_loss = self.loss_fct(logits.transpose(1,2), labels)
        original_tokens_mask = labels < self.new_tokens_start
        new_tokens_mask = ~original_tokens_mask
        loss = 0.0
        if self.original_tokens_loss_alpha > 0.0:
            loss += self.original_tokens_loss_alpha * per_example_loss[original_tokens_mask].mean()
        if self.new_tokens_loss_alpha > 0.0:
            loss += self.new_tokens_loss_alpha * per_example_loss[new_tokens_mask].mean()
        if self.subsequent_tokens_loss_alpha > 0.0:
            subsequent_tokens_mask = torch.zeros_like(original_tokens_mask, dtype=torch.bool)
            subsequent_tokens_mask[:, 1:][new_tokens_mask[:, :-1]] = True
            loss += self.subsequent_tokens_loss_alpha * per_example_loss[subsequent_tokens_mask].mean()

        return {'loss': loss, 'logits': logits}

    def get_calibrators(self):
        embedding_calibrator = self.embedding_calibrator if self.calibrate_embedding else None
        lm_head_calibrator = self.lm_head_calibrator if self.calibrate_lm_head else None
        return {
            "embedding_calibrator": embedding_calibrator,
            "lm_head_calibrator": lm_head_calibrator,
            "new_tokens_start": self.new_tokens_start,
            "new_tokens_end": self.new_tokens_end,
        }

    def set_calibrators(self, embedding_calibrator=None, lm_head_calibrator=None):
        self.embedding_calibrator = embedding_calibrator
        self.lm_head_calibrator = lm_head_calibrator
        
    def save_calibrators(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        if self.calibrate_embedding:
            torch.save(self.embedding_calibrator, os.path.join(save_dir, "embedding_calibrator.pt"))
        if self.calibrate_lm_head:
            torch.save(self.lm_head_calibrator, os.path.join(save_dir, "lm_head_calibrator.pt"))

    def load_calibrators(self, load_dir, fail_ok=False):
        """Loads the model's state dictionary from a file."""
        try:
            if self.calibrate_embedding:
                self.embedding_calibrator = torch.load(os.path.join(load_dir, "embedding_calibrator.pt"))
            if self.calibrate_lm_head:
                self.lm_head_calibrator = torch.load(os.path.join(load_dir, "lm_head_calibrator.pt"))
            return True
        except:
            if fail_ok:
                return False
            raise FileNotFoundError(f"Loading calibrators from '{load_dir}' failed")


def get_calibration_model(model, original_vocab_size, num_new_tokens, target_loss_weight=0.15, subsequent_loss_weight=0.15):
    calibrated_model = CalibrationModel(model.model, model.lm_head, original_vocab_size, num_new_tokens, target_loss_weight=target_loss_weight, subsequent_loss_weight=subsequent_loss_weight)
    calibrated_model.base_model.eval()
    calibrated_model.lm_head.eval()

    for param in calibrated_model.base_model.parameters():
        param.requires_grad = False
    for param in calibrated_model.lm_head.parameters():
        param.requires_grad = False
    for param in calibrated_model.lm_head_calibrator.parameters():
        param.requires_grad = True
    for param in calibrated_model.embedding_calibrator.parameters():
        param.requires_grad = True

    return calibrated_model


def train_calibration_model(calibrated_model: CalibrationModel, tokenizer, dataset, save_dir=None, max_samples=None, filter_examples_without_new_tokens=True, lr=1e-4, lr_schedule="linear", num_epochs=1, batch_size=8, max_length=256, n_warmup_steps=0, text_col_name="text", clip_grad_norm=1.0, mixed_precision=None):
    accelerator = Accelerator(mixed_precision=mixed_precision)
    # Optimizer
    optimizer = optim.AdamW(calibrated_model.parameters(), lr=lr)

    # Tokenize data
    if tokenizer.bos_token is not None and max_length:
        add_start_token = True
        # leave room for <BOS> token to be added:
        max_tokenized_len = max_length - 1
    else:
        add_start_token = False
        max_tokenized_len = max_length

    def _add_start_token(batch):
        bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * batch["input_ids"].size(dim=0)).to(batch["input_ids"].device)
        batch["input_ids"] = torch.cat([bos_tokens_tensor, batch["input_ids"]], dim=1)
        batch["attention_mask"] = torch.cat(
            [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(batch["attention_mask"].device), batch["attention_mask"]], dim=1)
        return batch

    tokenize_function = get_tokenize_func(tokenizer, text_col_name)

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

    if filter_examples_without_new_tokens:
        examples_w_new_token = np.arange(len(lm_dataset))[np.any(np.array(lm_dataset['input_ids']) >= calibrated_model.new_tokens_start, axis=1)]
        lm_dataset = lm_dataset.select(examples_w_new_token)

    if max_samples is not None:
        lm_dataset = lm_dataset.select(np.arange(max_samples))

    data_collator = default_data_collator

    # Create data loaders
    dataloader = DataLoader(
        lm_dataset, collate_fn=data_collator, batch_size=batch_size, drop_last=True, shuffle=True,
    )

    # Learning rate scheduler
    if isinstance(n_warmup_steps, float):
        n_warmup_steps = n_warmup_steps * len(dataloader)
    scheduler = get_scheduler(lr_schedule, optimizer=optimizer, num_warmup_steps=n_warmup_steps, num_training_steps=len(dataloader) * num_epochs)

    calibrated_model, dataloader = accelerator.prepare(calibrated_model, dataloader)

    # Freeze the original lm_head weights
    for param in calibrated_model.lm_head.parameters():
        param.requires_grad = False

    calibrated_model.train()
    for epoch in tqdm(range(num_epochs), unit="epochs", desc="Fitting calibration"):
        total_loss = 0.0
        for step, batch in tqdm(enumerate(dataloader), total=len(dataloader), miniters=10, unit="batches"):
            if add_start_token:
                batch = _add_start_token(batch)
            batch["labels"] = batch["input_ids"]
            optimizer.zero_grad()
            outputs = calibrated_model(**batch)
            loss = outputs['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(calibrated_model.parameters(), max_norm=clip_grad_norm)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            # # Log loss
            # if step % 10 == 0:
            #     print(f"Epoch {epoch + 1}, Step {step}, Loss: {loss.item()}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss}")

    if save_dir is not None:
        calibrated_model.save_calibrators(save_dir)

    return calibrated_model


def merge_calibrators_to_hf_model(hf_model, new_tokens_start, new_tokens_end=None, embedding_calibrator=None, lm_head_calibrator=None):
    embedding_calibrator.to(hf_model.device)
    lm_head_calibrator.to(hf_model.device)
    if embedding_calibrator is not None:
        embedding_weights = hf_model.get_input_embeddings().weight
        with torch.no_grad():
            calibrated_weights = embedding_calibrator(embedding_weights[new_tokens_start:new_tokens_end])
            hf_model.model.embed_tokens.weight.data[
            new_tokens_start:new_tokens_end] = calibrated_weights

    if lm_head_calibrator is not None:
        lm_head_weights = hf_model.get_output_embeddings().weight
        with torch.no_grad():
            calibrated_weights = lm_head_calibrator(lm_head_weights[new_tokens_start:new_tokens_end])
            hf_model.lm_head.weight.data[new_tokens_start:new_tokens_end] = calibrated_weights

    return hf_model


def merge_calibration_model_to_hf_model(hf_model, calibrated_model):
    calibrated_model.to(hf_model.device)
    if calibrated_model.calibrate_lm_head:
        lm_head_weights = calibrated_model.lm_head.weight
        normed_weights = calibrated_model.lm_head_calibrator(lm_head_weights[calibrated_model.new_tokens_start:calibrated_model.new_tokens_end])
        with torch.no_grad():
            hf_model.lm_head.weight.data[calibrated_model.new_tokens_start:calibrated_model.new_tokens_end] = normed_weights
    if calibrated_model.calibrate_embedding:
        embedding_weights = calibrated_model.base_model.get_input_embeddings().weight
        normed_weights = calibrated_model.embedding_calibrator(embedding_weights[calibrated_model.new_tokens_start:calibrated_model.new_tokens_end])
        with torch.no_grad():
            hf_model.model.embed_tokens.weight.data[calibrated_model.new_tokens_start:calibrated_model.new_tokens_end] = normed_weights
    return hf_model

