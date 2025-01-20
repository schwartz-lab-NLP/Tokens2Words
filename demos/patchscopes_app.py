import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr
from plotly import graph_objects as go
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B", )
    parser.add_argument("--hf_token", type=str, default=None,)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)
    args = parser.parse_args()
    return args


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device {device} for inference")

args = parse_args()
model_name = args.model_name
load_in_bf16 = args.bf16
load_in_fp16 = args.fp16
hf_token = args.hf_token


def _load_model(model_name):
    dtype = torch.bfloat16 if load_in_bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token).to(dtype)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

    return model, tokenizer


model, tokenizer = _load_model(model_name)



def parse_int_list(s):
    return [int(num) for num in s.replace(",", " ").split()]


def make_plot(model_choice, embedding_text, patchscopes_text, patchscopes_input_ids, word_to_embed, embedding_index, replace_token, num_new_tokens=10):
    global model_name, model, tokenizer
    if model_choice != model_name:
        model, tokenizer = _load_model(model_choice)
        model_name = model_choice

    # prepare input ids
    embedding_input_ids = tokenizer.encode(embedding_text.replace(replace_token, word_to_embed))
    embedding_input_ids = torch.tensor(embedding_input_ids, dtype=torch.int64)

    patchscopes_input_ids = parse_int_list(patchscopes_input_ids)
    if patchscopes_input_ids:
        replace_token_id = tokenizer.encode(replace_token)[-1]
        patchscopes_input_ids = torch.tensor(patchscopes_input_ids, dtype=torch.int64)
    else:
        patchscopes_input_ids = tokenizer.encode(patchscopes_text)
        patchscopes_input_ids = torch.tensor(patchscopes_input_ids, dtype=torch.int64)
        if f'"{replace_token}"' in patchscopes_text or f"'{replace_token}'" in patchscopes_text:
            replace_token_id = tokenizer.encode(f'"{replace_token}')[2]
        else:
            replace_token_id = tokenizer.encode(replace_token)[-1]

    # extract embedding from model for a new vocabulary word, then use it in the prompt
    embedding_index = int(embedding_index)
    with torch.no_grad():
        # first, extract the word's embedding from all layers
        outputs = model(input_ids=embedding_input_ids.unsqueeze(0).to(model.device), output_hidden_states=True)
        word_embedding_candidates = torch.cat([outputs.hidden_states[layer_i][:, embedding_index] for layer_i in range(1, len(outputs.hidden_states))])

        # now, run patchscopes to see what the model can "read" from each embedding
        idx_to_replace_mask = patchscopes_input_ids == replace_token_id
        inputs_embeds = model.get_input_embeddings()(patchscopes_input_ids.to(model.device)).unsqueeze(0)
        batched_patchscope_inputs = inputs_embeds.repeat(len(word_embedding_candidates), 1, 1)
        batched_patchscope_inputs[:, idx_to_replace_mask] = word_embedding_candidates.unsqueeze(1)
        batched_patchscope_inputs = batched_patchscope_inputs.to(model.device)
        attention_mask = (patchscopes_input_ids != tokenizer.eos_token_id).long().unsqueeze(0).repeat(len(word_embedding_candidates), 1).to(model.device)
        patchscope_outputs = model.generate(
            do_sample=False, num_beams=1, top_p=1.0, temperature=None,
            inputs_embeds=batched_patchscope_inputs, attention_mask=attention_mask, max_new_tokens=int(num_new_tokens), pad_token_id=tokenizer.eos_token_id, )
        decoded_patchscope_outputs = tokenizer.batch_decode(patchscope_outputs)

    return go.Figure(data=[go.Table(
        header=dict(values=["Layer Index", "Patchscope Result"],
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[list(range(1, len(decoded_patchscope_outputs)+1)), decoded_patchscope_outputs],
                   fill_color='lavender',
                   align='left'))
    ])


preamble = """
# Patchscopes Interface 🔎
TODO
## Usage
TODO
"""


with gr.Blocks() as demo:
    gr.Markdown(preamble)
    with gr.Column():
        model_choice = gr.Textbox(
            value=model_name,
            label="Model Name",
        )
        num_new_tokens = gr.Textbox(
            value='10',
            label="Number of tokens to generate",
        )

        with gr.Row():
            embedding_text = gr.Textbox(
                value=' X  X  X  X',
                label="Embedding Prompt Text",
            )

        with gr.Row():
            patchscopes_text = gr.Textbox(
                value=' X.  X.  X.  X',
                # value='"X" repeated endlessly, like a mantra: ',
                label="Patchscopes Prompt Text",
            )
            patchscopes_input_ids = gr.Textbox(
                value='',
                label="Prompt Input IDs (Optional, replaces text)",
            )
        with gr.Row():
            word_to_embed = gr.Textbox(
                value='at least',
                label="Word to embed",
            )
            embedding_index = gr.Textbox(
                value='-1',
                label="Take embedding from index:",
            )
            replace_token = gr.Textbox(
                value='X',
                label="Token to replace with target word:",
            )

        examine_btn = gr.Button(value="Submit")
        plot = gr.Plot()
    examine_btn.click(make_plot, [model_choice, embedding_text, patchscopes_text, patchscopes_input_ids, word_to_embed, embedding_index, replace_token, num_new_tokens], plot)
    demo.load(make_plot, [model_choice, embedding_text, patchscopes_text, patchscopes_input_ids, word_to_embed, embedding_index, replace_token, num_new_tokens], plot)

if __name__ == "__main__":
    demo.launch(share=True,)
    # make_plot(
    #     model_name,
    #     embedding_text="X X X X", patchscopes_text='"X" repeated endlessly, like a mantra: ', patchscopes_input_ids="", word_to_embed="at least", embedding_index="-1", replace_token="X",
    #     num_new_tokens=10)
