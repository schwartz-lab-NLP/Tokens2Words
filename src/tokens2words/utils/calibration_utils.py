import os
import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import unicodedata
from typing import Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from transformers import get_scheduler
from accelerate import Accelerator
from accelerate.utils import set_seed
from collections import defaultdict
from torch.utils.data import DataLoader
import torch.optim as optim

from ..utils.data_utils import load_lm_dataset, extract_new_words_from_dataset, get_group_texts_func, get_tokenize_func

def get_language_items(word_list, language):
    # Language to Unicode script name mapping
    language_scripts = {
        "Hebrew": "HEBREW",
        "Arabic": "ARABIC",
    }
    if language not in language_scripts and language.title() not in language_scripts:
        raise ValueError(f"Language {language} not supported")
    target_scripts = language_scripts.get(language, language_scripts[language.title()])
    if not isinstance(target_scripts, list):
        target_scripts = [target_scripts]
    language_tokens = []
    for token in word_list:
        for char in token:
            script_name = unicodedata.name(char, "").split()
            if len(script_name) > 0:
                script = script_name[0]
                if script in target_scripts:
                    language_tokens.append(token)
                    break
    return language_tokens

def get_language_tokens(tokenizer, language):
    """
    Find all tokens in a given language from a HuggingFace tokenizer.

    Args:
        tokenizer: HuggingFace tokenizer
        language: String name of language (e.g., "Hebrew", "Arabic")

    Returns:
        List of token IDs sorted in ascending order
    """
    # Language to Unicode script name mapping
    language_scripts = {
        "Hebrew": "HEBREW",
        "Arabic": "ARABIC",
    }

    if language not in language_scripts and language.title() not in language_scripts:
        raise ValueError(f"Language {language} not supported")

    target_scripts = language_scripts.get(language, language_scripts[language.title()])
    if not isinstance(target_scripts, list):
        target_scripts = [target_scripts]

    language_tokens = []
    vocab = tokenizer.get_vocab()

    for token, token_id in vocab.items():
        # Skip special tokens
        if token.startswith("<") and token.endswith(">"):
            continue

        # Check if token contains characters from target script
        for char in token:
            script_name = unicodedata.name(char, "").split()
            if len(script_name) > 0:
                script = script_name[0]
                if script in target_scripts:
                    language_tokens.append(token_id)
                    break

    return sorted(language_tokens)


class EmbeddingCalibrator(nn.Module):
    def __init__(self, num_new_tokens, hidden_size, lora_r=None, lora_alpha=None, learn_gate_activation=False, use_bias=False, dtype=torch.bfloat16):
        super().__init__()
        self.use_lora = lora_r is not None

        if not self.use_lora:
            self.weight = nn.Parameter(torch.zeros(hidden_size, hidden_size, dtype=dtype))
        else:
            self.lora_scaling = lora_alpha / lora_r if lora_alpha is not None else 1.0
            self.lora_A = nn.Parameter(torch.randn(lora_rank, hidden_size, dtype=dtype) * (1/lora_r))
            self.lora_B = nn.Parameter(torch.zeros(hidden_size, lora_rank, dtype=dtype))

        self.do_gate_act = learn_gate_activation
        if learn_gate_activation:
            self.gate_weight = nn.Parameter(torch.zeros(hidden_size, hidden_size, dtype=dtype))
            self.gate_act = nn.SiLU()

        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.zeros(num_new_tokens, hidden_size, dtype=dtype))

    def forward(self, x):
        batch_size = x.size(0) if len(x.shape) > 2 else 1
        result = x
        if hasattr(self, "do_gate_act") and self.do_gate_act:
            result = (1 + self.gate_act(torch.matmul(x, self.gate_weight.t()))) * x

        if not self.use_lora:
            result = result + torch.matmul(x, self.weight.t())
        else:
            lora_out = torch.matmul(x, self.lora_A.t())
            lora_out = torch.matmul(lora_out, self.lora_B.t())
            result = result + self.lora_scaling * lora_out

        if hasattr(self, "use_bias") and self.use_bias:
            if batch_size == 1:
                result = result + self.bias
            else:
                result = result + self.bias.unsqueeze(0)

        return result

    def set_use_bias(self, use_bias: bool = True):
        self.use_bias = use_bias
        if use_bias:
            if self.use_lora:
                self.lora_A.requires_grad = False
                self.lora_B.requires_grad = False
            else:
                self.weight.requires_grad = False

    def freeze(self):
        if self.use_lora:
            self.lora_A.requires_grad = False
            self.lora_B.requires_grad = False
        else:
            self.weight.requires_grad = False

        if self.use_bias:
            self.bias.requires_grad = False

    def unfreeze(self):
        if self.use_bias:
            self.freeze()
            self.bias.requires_grad = True
        else:
            if self.use_lora:
                self.lora_A.requires_grad = True
                self.lora_B.requires_grad = True
            else:
                self.weight.requires_grad = True


class CalibrationModel(nn.Module):
    def __init__(
            self,
            base_model, lm_head, original_vocab_size, num_new_tokens,
            calibrate_embedding=True, calibrate_lm_head=True, empty_init=False,
            lora_alpha=None, lora_r=None,
            learn_gate_activation=False, use_bias=False,
            target_loss_weight=0.10, subsequent_loss_weight=0.20,
            post_subsequent_loss_weight=0.0, post_subsequent_window_len=3,
            similarity_loss_weight=0.1,
            existing_tokens_to_calibrate=None,
            new_tokens_to_orig_first_map=None, soft_allow_orig_first_tokens=False,
    ):
        super().__init__()
        self.base_model = base_model
        self.lm_head = lm_head
        self.new_tokens_start = original_vocab_size
        self.new_tokens_end = original_vocab_size + num_new_tokens

        self.calibrate_lm_head = calibrate_lm_head
        self.calibrate_embedding = calibrate_embedding

        # use to map new tokens to the original 1st token in their encoding, for a softer loss
        self.soft_allow_orig_first_tokens = soft_allow_orig_first_tokens
        self.token_id_to_orig_first_id_map = None
        if self.soft_allow_orig_first_tokens:
            self.set_new_to_orig_id_map(new_tokens_to_orig_first_map)

        # Store existing tokens to calibrate
        self.existing_tokens_to_calibrate = existing_tokens_to_calibrate
        if existing_tokens_to_calibrate is not None and not isinstance(existing_tokens_to_calibrate, torch.Tensor):
            self.existing_tokens_to_calibrate = torch.Tensor(existing_tokens_to_calibrate).to(torch.long)

        if not empty_init:
            # Separate calibrators for new tokens
            self.lm_head_calibrator = EmbeddingCalibrator(
                num_new_tokens,
                base_model.config.hidden_size,
                lora_r, lora_alpha,
                learn_gate_activation,
                use_bias
            )
            self.embedding_calibrator = EmbeddingCalibrator(
                num_new_tokens,
                base_model.config.hidden_size,
                lora_r, lora_alpha,
                learn_gate_activation,
                use_bias
            )

            # Separate calibrators for existing tokens if specified
            if existing_tokens_to_calibrate is not None:
                self.existing_tokens_lm_head_calibrator = EmbeddingCalibrator(
                    len(existing_tokens_to_calibrate),
                    base_model.config.hidden_size,
                    lora_r, lora_alpha,
                    learn_gate_activation,
                    use_bias
                )
                self.existing_tokens_embedding_calibrator = EmbeddingCalibrator(
                    len(existing_tokens_to_calibrate),
                    base_model.config.hidden_size,
                    lora_r, lora_alpha,
                    learn_gate_activation,
                    use_bias
                )

        self.loss_fct = nn.CrossEntropyLoss(reduction="none")
        self.subsequent_tokens_loss_alpha = subsequent_loss_weight
        self.post_subsequent_tokens_loss_alpha = post_subsequent_loss_weight
        self.post_subsequent_window_len = post_subsequent_window_len
        self.new_tokens_loss_alpha = target_loss_weight
        self.original_tokens_loss_alpha = 1 - self.new_tokens_loss_alpha - self.subsequent_tokens_loss_alpha \
                                          - self.post_subsequent_tokens_loss_alpha

        self.similarity_loss_weight = similarity_loss_weight
        self.similarity_input_target_embed, self.similarity_output_target_embed = None, None

    def set_new_to_orig_id_map(self, mapping_dict: Dict[int, int]):
        if mapping_dict is not None:
            indices = torch.Tensor(list(mapping_dict.keys())).to(torch.long)
            values = torch.Tensor(list(mapping_dict.values())).to(torch.long)
            self.token_id_to_orig_first_id_map = torch.arange(self.new_tokens_end, dtype=torch.long)
            self.token_id_to_orig_first_id_map[indices] = values

    def _map_labels_to_orig_first(self, tokens: torch.Tensor) -> torch.Tensor:
        """Efficiently map tokens using tensor operations."""
        if self.token_id_to_orig_first_id_map is None:
            return tokens

        # Move mapping tensors to same device as input
        if self.token_id_to_orig_first_id_map.device != tokens.device:
            self.token_id_to_orig_first_id_map = self.token_id_to_orig_first_id_map.to(tokens.device)

        return self.token_id_to_orig_first_id_map[tokens]

    def _get_calibrated_embeddings(self, weights):
        # Handle new tokens
        calibrated_weights = torch.cat((
            weights[:self.new_tokens_start],
            self.embedding_calibrator(weights[self.new_tokens_start:self.new_tokens_end])
        ))

        # Handle existing tokens if specified
        if self.existing_tokens_to_calibrate is not None:
            existing_token_weights = weights[self.existing_tokens_to_calibrate]
            calibrated_existing = self.existing_tokens_embedding_calibrator(existing_token_weights)
            calibrated_weights[self.existing_tokens_to_calibrate] = calibrated_existing

        return calibrated_weights

    def _get_calibrated_lm_head_weights(self, weights):
        normed_weights = weights.clone()

        # Handle new tokens
        normed_weights[self.new_tokens_start:self.new_tokens_end] = self.lm_head_calibrator(
            weights[self.new_tokens_start:self.new_tokens_end]
        )

        # Handle existing tokens if specified
        if self.existing_tokens_to_calibrate is not None:
            existing_token_weights = weights[self.existing_tokens_to_calibrate]
            calibrated_existing = self.existing_tokens_lm_head_calibrator(existing_token_weights)
            normed_weights[self.existing_tokens_to_calibrate] = calibrated_existing

        return normed_weights

    def set_similarity_regularization_embeddings(self, token_ids, strategy="mean"):
        if isinstance(token_ids, list):
            token_ids = torch.Tensor(token_ids).to(torch.long)
        token_ids = token_ids.to(self.base_model.device)
        with torch.no_grad():
            inputs_embeds = self.base_model.get_input_embeddings()(token_ids).detach()
            outputs_embeds = self.lm_head.weight.data[token_ids].detach()
        if strategy == "mean":
            self.similarity_input_target_embed = inputs_embeds.mean(dim=0, keepdim=True)
            self.similarity_output_target_embed = outputs_embeds.mean(dim=0, keepdim=True)

    def forward(self, input_ids, labels, attention_mask=None):
        # shift labels by 1 for CLM
        labels = labels[:, 1:].contiguous()
        input_ids = input_ids[:, :-1].contiguous()

        if self.calibrate_embedding:
            E_weights = self.base_model.get_input_embeddings().weight.data
            E_weights = self._get_calibrated_embeddings(E_weights)
            input_embeddings = E_weights[input_ids]
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.long)
            outputs = self.base_model(inputs_embeds=input_embeddings, attention_mask=attention_mask)
        else:
            with torch.no_grad():
                outputs = self.base_model(input_ids, attention_mask=attention_mask)

        if self.calibrate_lm_head:
            with torch.no_grad():
                lm_head_weights = self.lm_head.weight
            U_weights = self._get_calibrated_lm_head_weights(lm_head_weights)
            logits = torch.matmul(outputs['last_hidden_state'], U_weights.T)
        else:
            if self.calibrate_embedding:
                logits = self.lm_head(outputs['last_hidden_state'])
            else:
                with torch.no_grad():
                    logits = self.lm_head(outputs['last_hidden_state'])

        per_example_loss = self.loss_fct(logits.transpose(1, 2), labels)
        # Create masks for new and existing tokens
        original_tokens_mask = labels < self.new_tokens_start
        new_tokens_mask = (labels >= self.new_tokens_start) & (labels < self.new_tokens_end)

        loss = 0.0
        if self.original_tokens_loss_alpha > 0.0:
            loss += self.original_tokens_loss_alpha * per_example_loss[original_tokens_mask].mean()
        if self.new_tokens_loss_alpha > 0.0:
            curr_loss = per_example_loss[new_tokens_mask]
            if self.soft_allow_orig_first_tokens:
                orig_loss = self.loss_fct(
                    logits[new_tokens_mask],
                    self._map_labels_to_orig_first(labels[new_tokens_mask])
                )
                curr_loss = (curr_loss + orig_loss) / 2
            loss += self.new_tokens_loss_alpha * curr_loss.mean()
        if self.subsequent_tokens_loss_alpha > 0.0:
            subsequent_tokens_mask = torch.zeros_like(original_tokens_mask, dtype=torch.bool)
            subsequent_tokens_mask[:, 1:][new_tokens_mask[:, :-1]] = True
            loss += self.subsequent_tokens_loss_alpha * per_example_loss[subsequent_tokens_mask].mean()
        if self.post_subsequent_tokens_loss_alpha > 0.0:
            post_subsequent_tokens_mask = torch.zeros_like(original_tokens_mask, dtype=torch.bool)
            for i in range(1, self.post_subsequent_window_len + 1):
                post_subsequent_tokens_mask[:, 1 + i:][new_tokens_mask[:, :-1 - i]] = True
            loss += self.post_subsequent_tokens_loss_alpha * per_example_loss[post_subsequent_tokens_mask].mean()

        if self.similarity_loss_weight > 0.0 and self.similarity_input_target_embed is not None:
            similarity_loss = 0
            if self.calibrate_embedding:
                finetuned_new_E_weights = E_weights[self.new_tokens_start:self.new_tokens_end]
                similarity_loss += torch.mean(torch.norm(finetuned_new_E_weights - self.similarity_input_target_embed, p=2, dim=1) ** 2)
                if self.existing_tokens_to_calibrate is not None:
                    finetuned_existing_E_weights = E_weights[self.existing_tokens_to_calibrate]
                    similarity_loss += torch.mean(torch.norm(finetuned_existing_E_weights - self.similarity_input_target_embed, p=2, dim=1) ** 2)
            if self.calibrate_lm_head:
                finetuned_new_U_weights = U_weights[self.new_tokens_start:self.new_tokens_end]
                similarity_loss += torch.mean(torch.norm(finetuned_new_U_weights - self.similarity_output_target_embed, p=2, dim=1) ** 2)
                if self.existing_tokens_to_calibrate is not None:
                    finetuned_existing_U_weights = U_weights[self.existing_tokens_to_calibrate]
                    similarity_loss += torch.mean(torch.norm(finetuned_existing_U_weights - self.similarity_output_target_embed, p=2, dim=1) ** 2)
            loss += self.similarity_loss_weight * similarity_loss

        return {'loss': loss, 'logits': logits}

    def freeze_base_model(self):
        for param in self.base_model.parameters():
            param.requires_grad = False
        for param in self.lm_head.parameters():
            param.requires_grad = False

    def freeze_calibrators(self):
        self.embedding_calibrator.freeze()
        self.lm_head_calibrator.freeze()
        if hasattr(self, 'existing_tokens_embedding_calibrator'):
            self.existing_tokens_embedding_calibrator.freeze()
            self.existing_tokens_lm_head_calibrator.freeze()

    def freeze_existing_tokens_calibrators(self):
        if hasattr(self, 'existing_tokens_embedding_calibrator'):
            self.existing_tokens_embedding_calibrator.freeze()
            self.existing_tokens_lm_head_calibrator.freeze()

    def unfreeze_calibrators(self):
        self.embedding_calibrator.unfreeze()
        self.lm_head_calibrator.unfreeze()
        if hasattr(self, 'existing_tokens_embedding_calibrator'):
            self.existing_tokens_embedding_calibrator.unfreeze()
            self.existing_tokens_lm_head_calibrator.unfreeze()

    def get_calibrators(self):
        embedding_calibrator = self.embedding_calibrator if self.calibrate_embedding else None
        lm_head_calibrator = self.lm_head_calibrator if self.calibrate_lm_head else None

        existing_tokens_embedding_calibrator = None
        existing_tokens_lm_head_calibrator = None
        if hasattr(self, 'existing_tokens_embedding_calibrator'):
            existing_tokens_embedding_calibrator = self.existing_tokens_embedding_calibrator if self.calibrate_embedding else None
            existing_tokens_lm_head_calibrator = self.existing_tokens_lm_head_calibrator if self.calibrate_lm_head else None

        return {
            "embedding_calibrator": embedding_calibrator,
            "lm_head_calibrator": lm_head_calibrator,
            "existing_tokens_embedding_calibrator": existing_tokens_embedding_calibrator,
            "existing_tokens_lm_head_calibrator": existing_tokens_lm_head_calibrator,
            "new_tokens_start": self.new_tokens_start,
            "new_tokens_end": self.new_tokens_end,
            "existing_tokens_to_calibrate": self.existing_tokens_to_calibrate,
        }

    def set_use_bias(self, use_bias: bool = True):
        self.embedding_calibrator.set_use_bias(use_bias)
        self.lm_head_calibrator.set_use_bias(use_bias)
        if hasattr(self, 'existing_tokens_embedding_calibrator'):
            self.existing_tokens_embedding_calibrator.set_use_bias(use_bias)
            self.existing_tokens_lm_head_calibrator.set_use_bias(use_bias)

    def save_calibrators(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        if self.calibrate_embedding:
            torch.save(self.embedding_calibrator,
                       os.path.join(save_dir, "embedding_calibrator.pt"))
            if hasattr(self, 'existing_tokens_embedding_calibrator'):
                torch.save(self.existing_tokens_embedding_calibrator,
                           os.path.join(save_dir, "existing_tokens_embedding_calibrator.pt"))
        if self.calibrate_lm_head:
            torch.save(self.lm_head_calibrator, os.path.join(save_dir, "lm_head_calibrator.pt"))
            if hasattr(self, 'existing_tokens_lm_head_calibrator'):
                torch.save(self.existing_tokens_lm_head_calibrator,
                           os.path.join(save_dir, "existing_tokens_lm_head_calibrator.pt"))

    def load_calibrators(self, load_dir, fail_ok=False):
        try:
            if self.calibrate_embedding:
                self.embedding_calibrator = torch.load(
                    os.path.join(load_dir, "embedding_calibrator.pt"))
                if hasattr(self, 'existing_tokens_embedding_calibrator'):
                    self.existing_tokens_embedding_calibrator = torch.load(
                        os.path.join(load_dir, "existing_tokens_embedding_calibrator.pt"))
            if self.calibrate_lm_head:
                self.lm_head_calibrator = torch.load(
                    os.path.join(load_dir, "lm_head_calibrator.pt"))
                if hasattr(self, 'existing_tokens_lm_head_calibrator'):
                    self.existing_tokens_lm_head_calibrator = torch.load(
                        os.path.join(load_dir, "existing_tokens_lm_head_calibrator.pt"))
            return True
        except:
            if fail_ok:
                return False
            raise FileNotFoundError(f"Loading calibrators from '{load_dir}' failed")


def get_calibration_model(model, original_vocab_size, num_new_tokens,
                          learn_gate_activation=False,
                          existing_tokens_to_calibrate=None,
                          new_tokens_to_orig_first_map=None, soft_allow_orig_first_tokens=False,
                          ):
    calibrated_model = CalibrationModel(
        model.model,
        model.lm_head,
        original_vocab_size,
        num_new_tokens,
        learn_gate_activation=learn_gate_activation,
        existing_tokens_to_calibrate=existing_tokens_to_calibrate,
        new_tokens_to_orig_first_map=new_tokens_to_orig_first_map,
        soft_allow_orig_first_tokens=soft_allow_orig_first_tokens,
    )

    calibrated_model.base_model.eval()
    calibrated_model.lm_head.eval()

    calibrated_model.freeze_base_model()
    calibrated_model.unfreeze_calibrators()

    return calibrated_model


def train_calibration_model(calibrated_model: CalibrationModel, tokenizer, dataset,
                            save_dir=None, max_samples=None,
                            filter_examples_without_calibration_tokens=True,
                            lr=1e-4, lr_schedule="linear", num_epochs=1,
                            batch_size=8, max_length=256, n_warmup_steps=0,
                            text_col_name="text", clip_grad_norm=1.0,
                            mixed_precision=None, freeze_existing_tokens=False,
                            existing_tokens_for_similarity_loss=None,):
    accelerator = Accelerator(mixed_precision=mixed_precision)
    # Optimizer
    optimizer = optim.AdamW(calibrated_model.parameters(), lr=lr)

    # Tokenize data
    if tokenizer.bos_token is not None and max_length:
        add_start_token = True
        max_tokenized_len = max_length - 1
    else:
        add_start_token = False
        max_tokenized_len = max_length

    def _add_start_token(batch):
        bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * batch["input_ids"].size(dim=0)).to(
            batch["input_ids"].device)
        batch["input_ids"] = torch.cat([bos_tokens_tensor, batch["input_ids"]], dim=1)
        batch["attention_mask"] = torch.cat(
            [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(batch["attention_mask"].device),
             batch["attention_mask"]], dim=1)
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

    if filter_examples_without_calibration_tokens:
        # Create mask for both new and existing tokens
        tokens_to_check = list(range(calibrated_model.new_tokens_start, calibrated_model.new_tokens_end))

        def has_tokens_to_calibrate(example):
            return any(token in example['input_ids'] for token in tokens_to_check)

        lm_dataset = lm_dataset.filter(has_tokens_to_calibrate)

    if max_samples is not None and len(lm_dataset) > max_samples:
        lm_dataset = lm_dataset.select(np.arange(max_samples))

    # Rest of the training function remains the same
    data_collator = default_data_collator
    dataloader = DataLoader(
        lm_dataset, collate_fn=data_collator, batch_size=batch_size, drop_last=True, shuffle=True,
    )

    if isinstance(n_warmup_steps, float):
        n_warmup_steps = n_warmup_steps * len(dataloader)
    scheduler = get_scheduler(lr_schedule, optimizer=optimizer, num_warmup_steps=n_warmup_steps,
                              num_training_steps=len(dataloader) * num_epochs)

    calibrated_model, dataloader = accelerator.prepare(calibrated_model, dataloader)

    calibrated_model.train()
    calibrated_model.freeze_base_model()
    calibrated_model.unfreeze_calibrators()
    if freeze_existing_tokens:
        calibrated_model.freeze_existing_tokens_calibrators()

    # prepare targets for embedding similarity regularization
    if existing_tokens_for_similarity_loss is not None:
        calibrated_model.set_similarity_regularization_embeddings(existing_tokens_for_similarity_loss)

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

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss}")

    if save_dir is not None:
        calibrated_model.save_calibrators(save_dir)

    return calibrated_model


def merge_calibrators_to_hf_model(hf_model, new_tokens_start, new_tokens_end=None,
                                  embedding_calibrator=None, lm_head_calibrator=None,
                                  existing_tokens_to_calibrate=None,
                                  existing_tokens_embedding_calibrator=None, existing_tokens_lm_head_calibrator=None):
    if embedding_calibrator is not None:
        embedding_calibrator.to(hf_model.device)
    if lm_head_calibrator is not None:
        lm_head_calibrator.to(hf_model.device)
    if existing_tokens_embedding_calibrator is not None:
        existing_tokens_embedding_calibrator.to(hf_model.device)
    if existing_tokens_lm_head_calibrator is not None:
        existing_tokens_lm_head_calibrator.to(hf_model.device)

    # Handle input embeddings
    if embedding_calibrator is not None or existing_tokens_embedding_calibrator is not None:
        embedding_weights = hf_model.get_input_embeddings().weight
        with torch.no_grad():
            if embedding_calibrator is not None:
                calibrated_weights = embedding_calibrator(embedding_weights[new_tokens_start:new_tokens_end])
                hf_model.model.embed_tokens.weight.data[new_tokens_start:new_tokens_end] = calibrated_weights

            if existing_tokens_embedding_calibrator is not None and existing_tokens_to_calibrate is not None:
                existing_weights = embedding_weights[existing_tokens_to_calibrate]
                calibrated_existing = existing_tokens_embedding_calibrator(existing_weights)
                hf_model.model.embed_tokens.weight.data[existing_tokens_to_calibrate] = calibrated_existing

    # Handle LM head
    if lm_head_calibrator is not None or existing_tokens_lm_head_calibrator is not None:
        lm_head_weights = hf_model.get_output_embeddings().weight
        with torch.no_grad():
            if lm_head_calibrator is not None:
                calibrated_weights = lm_head_calibrator(lm_head_weights[new_tokens_start:new_tokens_end])
                hf_model.lm_head.weight.data[new_tokens_start:new_tokens_end] = calibrated_weights

            if existing_tokens_lm_head_calibrator is not None and existing_tokens_to_calibrate is not None:
                existing_weights = lm_head_weights[existing_tokens_to_calibrate]
                calibrated_existing = existing_tokens_lm_head_calibrator(existing_weights)
                hf_model.lm_head.weight.data[existing_tokens_to_calibrate] = calibrated_existing

    return hf_model


def merge_calibration_model_to_hf_model(hf_model, calibrated_model):
    calibrated_model.to(hf_model.device)

    calibrators = calibrated_model.get_calibrators()

    return merge_calibrators_to_hf_model(
        hf_model=hf_model,
        new_tokens_start=calibrators["new_tokens_start"],
        new_tokens_end=calibrators["new_tokens_end"],
        embedding_calibrator=calibrators["embedding_calibrator"],
        lm_head_calibrator=calibrators["lm_head_calibrator"],
        existing_tokens_to_calibrate=calibrators["existing_tokens_to_calibrate"],
        existing_tokens_embedding_calibrator=calibrators["existing_tokens_embedding_calibrator"],
        existing_tokens_lm_head_calibrator=calibrators["existing_tokens_lm_head_calibrator"]
    )

