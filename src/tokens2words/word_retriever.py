import torch
from tqdm import tqdm
from abc import ABC, abstractmethod

from .utils.enums import MultiTokenKind, RetrievalTechniques
from .processor import RetrievalProcessor
from .utils.logit_lens import ReverseLogitLens
from .utils.model_utils import extract_token_i_hidden_states


class WordRetrieverBase(ABC):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @abstractmethod
    def retrieve_word(self, hidden_states, layer_idx=None, num_tokens_to_generate=3):
        pass


class PatchscopesRetriever(WordRetrieverBase):
    def __init__(
            self,
            model,
            tokenizer,
            representation_prompt: str = "{word}",
            patchscopes_prompt: str = "Next is the same word twice: 1) {word} 2)",
            prompt_target_placeholder: str = "{word}",
            representation_token_idx_to_extract: int = -1,
            num_tokens_to_generate: int = 10,
    ):
        super().__init__(model, tokenizer)
        self.prompt_input_ids, self.prompt_target_idx = \
            self._build_prompt_input_ids_template(patchscopes_prompt, prompt_target_placeholder)
        self._prepare_representation_prompt = \
            self._build_representation_prompt_func(representation_prompt, prompt_target_placeholder)
        self.representation_token_idx = representation_token_idx_to_extract
        self.num_tokens_to_generate = num_tokens_to_generate

    def _build_prompt_input_ids_template(self, prompt, target_placeholder):
        prompt_input_ids = [self.tokenizer.bos_token_id] if self.tokenizer.bos_token_id is not None else []
        target_idx = []

        if prompt:
            assert target_placeholder is not None, \
                "Trying to set a prompt for Patchscopes without defining the prompt's target placeholder string, e.g., [MASK]"

            prompt_parts = prompt.split(target_placeholder)
            for part_i, prompt_part in enumerate(prompt_parts):
                prompt_input_ids += self.tokenizer.encode(prompt_part, add_special_tokens=False)
                if part_i < len(prompt_parts)-1:
                    target_idx += [len(prompt_input_ids)]
                    prompt_input_ids += [0]
        else:
            prompt_input_ids += [0]
            target_idx = [len(prompt_input_ids)]

        prompt_input_ids = torch.tensor(prompt_input_ids, dtype=torch.long)
        target_idx = torch.tensor(target_idx, dtype=torch.long)
        return prompt_input_ids, target_idx

    def _build_representation_prompt_func(self, prompt, target_placeholder):
        return lambda word: prompt.replace(target_placeholder, word)

    def generate_states(self, tokenizer, word='Wakanda', with_prompt=True):
        prompt = self.generate_prompt() if with_prompt else word
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        return input_ids

    def retrieve_word(self, hidden_states, layer_idx=None, num_tokens_to_generate=None):
        self.model.eval()

        # insert hidden states into patchscopes prompt
        if hidden_states.dim() == 1:
            hidden_states = hidden_states.unsqueeze(0)

        inputs_embeds = self.model.get_input_embeddings()(self.prompt_input_ids.to(self.model.device)).unsqueeze(0)
        batched_patchscope_inputs = inputs_embeds.repeat(len(hidden_states), 1, 1).to(hidden_states.dtype)
        batched_patchscope_inputs[:, self.prompt_target_idx] = hidden_states.unsqueeze(1).to(self.model.device)

        attention_mask = (self.prompt_input_ids != self.tokenizer.eos_token_id).long().unsqueeze(0).repeat(
            len(hidden_states), 1).to(self.model.device)

        num_tokens_to_generate = num_tokens_to_generate if num_tokens_to_generate else self.num_tokens_to_generate

        with torch.no_grad():
            patchscope_outputs = self.model.generate(
                do_sample=False, num_beams=1, top_p=1.0, temperature=None,
                inputs_embeds=batched_patchscope_inputs, attention_mask=attention_mask,
                max_new_tokens=num_tokens_to_generate, pad_token_id=self.tokenizer.eos_token_id, )

        decoded_patchscope_outputs = self.tokenizer.batch_decode(patchscope_outputs)
        return decoded_patchscope_outputs

    def extract_hidden_states(self, word):
        representation_input = self._prepare_representation_prompt(word)

        last_token_hidden_states = extract_token_i_hidden_states(
            self.model, self.tokenizer, representation_input, token_idx_to_extract=self.representation_token_idx, return_dict=False, verbose=False)

        return last_token_hidden_states

    def get_hidden_states_and_retrieve_word(self, word, num_tokens_to_generate=None):
        last_token_hidden_states = self.extract_hidden_states(word)
        patchscopes_description_by_layers = self.retrieve_word(
            last_token_hidden_states, num_tokens_to_generate=num_tokens_to_generate)

        return patchscopes_description_by_layers, last_token_hidden_states


class ReverseLogitLensRetriever(WordRetrieverBase):
    def __init__(self, model, tokenizer, device='cuda', dtype=torch.float16):
        super().__init__(model, tokenizer)
        self.reverse_logit_lens = ReverseLogitLens.from_model(model).to(device).to(dtype)

    def retrieve_word(self, hidden_states, layer_idx=None, num_tokens_to_generate=3):
        result = self.reverse_logit_lens(hidden_states, layer_idx)
        token = self.tokenizer.decode(torch.argmax(result, dim=-1).item())
        return token


class AnalysisWordRetriever:
    def __init__(self, model, tokenizer, multi_token_kind, num_tokens_to_generate=1, add_context=True,
                 model_name='LLaMa-2B', device='cuda', dataset=None):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.multi_token_kind = multi_token_kind
        self.num_tokens_to_generate = num_tokens_to_generate
        self.add_context = add_context
        self.model_name = model_name
        self.device = device
        self.dataset = dataset
        self.retriever = self._initialize_retriever()
        self.RetrievalTechniques = (RetrievalTechniques.Patchscopes if self.multi_token_kind == MultiTokenKind.Natural
                                    else RetrievalTechniques.ReverseLogitLens)
        self.whitespace_token = 'Ġ' if model_name in ['gemma-2-9b', 'pythia-6.9b', 'LLaMA3-8B', 'Yi-6B'] else '▁'
        self.processor = RetrievalProcessor(self.model, self.tokenizer, self.multi_token_kind,
                                            self.num_tokens_to_generate, self.add_context, self.model_name,
                                            self.whitespace_token)

    def _initialize_retriever(self):
        if self.multi_token_kind == MultiTokenKind.Natural:
            return PatchscopesRetriever(self.model, self.tokenizer)
        else:
            return ReverseLogitLensRetriever(self.model, self.tokenizer)

    def retrieve_words_in_dataset(self, number_of_examples_to_retrieve=2, max_length=1000):
        self.model.eval()
        results = []

        for text in tqdm(self.dataset['train']['text'][:number_of_examples_to_retrieve], self.model_name):
            tokenized_input = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length).to(
                self.device)
            tokens = tokenized_input.input_ids[0]
            print(f'Processing text: {text}')
            i = 5
            while i < len(tokens):
                if self.multi_token_kind == MultiTokenKind.Natural:
                    j, word_tokens, word, context, tokenized_combined_text, combined_text, original_word = self.processor.get_next_word(
                        tokens, i, device=self.device)
                elif self.multi_token_kind == MultiTokenKind.Typo:
                    j, word_tokens, word, context, tokenized_combined_text, combined_text, original_word = self.processor.get_next_full_word_typo(
                        tokens, i, device=self.device)
                else:
                    j, word_tokens, word, context, tokenized_combined_text, combined_text, original_word = self.processor.get_next_full_word_separated(
                        tokens, i, device=self.device)

                if len(word_tokens) > 1:
                    with torch.no_grad():
                        outputs = self.model(**tokenized_combined_text, output_hidden_states=True)

                    hidden_states = outputs.hidden_states
                    for layer_idx, hidden_state in enumerate(hidden_states):
                        postfix_hidden_state = hidden_states[layer_idx][0, -1, :].unsqueeze(0)
                        retrieved_word_str = self.retriever.retrieve_word(postfix_hidden_state, layer_idx=layer_idx,
                                                                          num_tokens_to_generate=len(word_tokens))
                        results.append({
                            'text': combined_text,
                            'original_word': original_word,
                            'word': word,
                            'word_tokens': self.tokenizer.convert_ids_to_tokens(word_tokens),
                            'num_tokens': len(word_tokens),
                            'layer': layer_idx,
                            'retrieved_word_str': retrieved_word_str,
                            'context': "With Context" if self.add_context else "Without Context"
                        })
                else:
                    i = j
        return results
