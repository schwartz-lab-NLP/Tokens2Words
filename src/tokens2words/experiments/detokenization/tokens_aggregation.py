import os
import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import pandas as pd
import torch

from utils import MODELS_MAP, dtype_arg, load_dataset_for_analysis


# Function to measure attention weights and generate word from hidden states
def measure_attention_weights(model, tokenizer, dataset, device='cuda', max_length=1000, num_tokens_to_generate=3, model_name=None):
    model.to(device)
    model.eval()
    whitespace_token = 'Ġ' if model_name in ['gemma-2-9b', 'pythia-6.9b', 'LLaMA3-8B', 'Yi-6B'] else '▁'

    results = []

    for text in tqdm(dataset['train']['text'][:1000]):
        tokenized_input = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length).to(device)

        tokens = tokenized_input.input_ids[0]
        if len(text) > 0:

            with torch.no_grad():
                outputs = model(**tokenized_input, output_attentions=True, output_hidden_states=True)

            attentions = outputs.attentions

            i = 5
            while i < len(tokens):
                token_str = tokenizer.convert_ids_to_tokens(tokens[i].item())

                if token_str.startswith(whitespace_token):
                    # Collect word tokens until the next word boundary
                    word_tokens = [tokens[i]]
                    j = i + 1
                    while j < len(tokens) and not tokenizer.convert_ids_to_tokens(tokens[j].item()).startswith(whitespace_token):
                        word_tokens.append(tokens[j])
                        j += 1
                    if not tokenizer.decode(word_tokens).isalpha():  # Assuming this checks for English words
                        i = j
                        continue

                    is_1_token = True if len(word_tokens) == 1 else False
                    last_token_index = i + len(word_tokens) - 1
                    prefix_indices = list(range(i, last_token_index))
                    other_indices = list(range(1, i))
                    prev_token_index = i - 1
                    prev_prev_token_index = i - 2
                    prev_prev_prev_token_index = i - 3

                    for layer_idx, attention_matrix in enumerate(attentions):
                        last_token_attentions = attention_matrix[0, :, last_token_index, :].sum(dim=0)
                        sum_attention_weights = last_token_attentions.sum()
                        normalized_attention_weights = (last_token_attentions / sum_attention_weights).tolist()

                        avg_prefix_attention = sum(normalized_attention_weights[idx] for idx in prefix_indices) / len(prefix_indices) if not is_1_token else 0
                        avg_other_attention = sum(normalized_attention_weights[idx] for idx in other_indices) / len(other_indices)
                        prefix_attention = sum(normalized_attention_weights[idx] for idx in prefix_indices)
                        other_attention = sum(normalized_attention_weights[idx] for idx in other_indices)
                        prev_token_attention = normalized_attention_weights[prev_token_index]
                        prev_prev_token_attention = normalized_attention_weights[prev_prev_token_index]
                        prev_prev_prev_token_attention = normalized_attention_weights[prev_prev_prev_token_index]
                        avg_prev_tokens_attention = (prev_prev_prev_token_attention + prev_prev_token_attention + prev_token_attention) / 3
                        len_prefix = len(prefix_indices)
                        len_other = len(other_indices)
                        self_attention = normalized_attention_weights[last_token_index]
                        bos_attention = normalized_attention_weights[0]
                        avg_other_all_attention = (sum(normalized_attention_weights[idx] for idx in other_indices) + self_attention + bos_attention) / (len(other_indices) + 2)

                        # # Call `replace_generate` for the word and layer
                        # new_vector = hidden_states[layer_idx][0, last_token_index, :].unsqueeze(0)
                        # generated_word = replace_generate(model, source_input_ids, new_vector, start_index,
                        #                                   end_index, tokenizer, num_tokens_to_generate)
                        #
                        # # Collect the generated word tokens
                        # generated_word_tokens = [tokenizer.decode(token_id[0]) for token_id in generated_word]
                        # generated_word_str = "".join(generated_word_tokens).replace(" ", "")

                        results.append({
                            'text': text,
                            'word': tokenizer.decode(word_tokens),
                            'word_tokens': tokenizer.convert_ids_to_tokens(word_tokens),
                            'layer': layer_idx,
                            'avg_prefix_attention': avg_prefix_attention,
                            'self_attention': self_attention,
                            'avg_other_attention': avg_other_attention,
                            'bos_attention': bos_attention,
                            'avg_other_all_attention': avg_other_all_attention,
                            'prefix_attention': prefix_attention,
                            'other_attention': other_attention,
                            'len_prefix': len_prefix,
                            'len_other': len_other,
                            'avg_prev_tokens_attention': avg_prev_tokens_attention,
                            'prev_token_attention': prev_token_attention,
                            'prev_prev_token_attention': prev_prev_token_attention,
                            'prev_prev_prev_token_attention': prev_prev_prev_token_attention,
                            'is_1_token': is_1_token,
                            'last_token': tokenizer.convert_ids_to_tokens(tokens[last_token_index].item()),
                            # 'Generated Word': generated_word_tokens,
                            # 'Generated Word Str': generated_word_str,
                        })

                    i = j  # Move index to next word boundary
                else:
                    i += 1

    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", nargs="+", type=str, default="Yi-6B")
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--output_dir", type=str, default="/cs/labs/roys/guy.kaplan3/Tokens2Word/output/tokens_aggregation")
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--dtype", type=dtype_arg, default=torch.bfloat16)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    models_info = {
        model_name: MODELS_MAP[model_name] for model_name in args.model_name
    } if MODELS_MAP != "all" else MODELS_MAP
    dtype = args.dtype
    dataset = load_dataset_for_analysis(args.dataset)

    for model_name, model_path in models_info.items():
        print(f"Running model: {model_name}")

        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=auth_token)
        model = AutoModelForCausalLM.from_pretrained(model_path, use_auth_token=auth_token).to(device)

        # Assuming dataset is already loaded or provided
        # Run the attention weights measurement
        results = measure_attention_weights(model, tokenizer, dataset, device=device, model_name=model_name)
        results_df = pd.DataFrame(results)

        # Save the results DataFrame (optional)
        results_df.to_csv(os.path.join(args.output_dir, f'{model_name}_results.csv'))

