import random
import torch


class RetrievalProcessor:
    def __init__(self, model, tokenizer, multi_token_kind, num_tokens_to_generate,
                 add_context, model_name, whitespace_token='Ġ'):
        self.model = model
        self.tokenizer = tokenizer
        self.multi_token_kind = multi_token_kind
        self.num_tokens_to_generate = num_tokens_to_generate
        self.add_context = add_context
        self.model_name = model_name
        self.whitespace_token = whitespace_token

    def get_next_word(self, tokens, i, max_length=1000, device='cuda'):
        token_str = self.tokenizer.convert_ids_to_tokens(tokens[i].item())
        j = i + 1
        word_tokens = [tokens[i]]
        if token_str.startswith(self.whitespace_token):
            while j < len(tokens) and (
                    self.is_alpha_not_prefix(tokens[j])):
                word_tokens.append(tokens[j])
                j += 1
        word = self.tokenizer.decode(word_tokens)
        original_word = word
        context = self.tokenizer.decode(tokens[:i]) if self.add_context else ""
        combined_text = context + word

        tokenized_combined_text = self.tokenizer(combined_text, return_tensors='pt', truncation=True,
                                                 max_length=max_length).to(device)
        return j, word_tokens, word, context, tokenized_combined_text, combined_text, original_word

    def get_next_full_word_typo(self, tokens, i, max_length=1000, device='cuda'):
        tokens_str = self.tokenizer.convert_ids_to_tokens(tokens)
        word_tokens = [tokens[i]]
        word = self.tokenizer.decode(word_tokens)
        original_word = word
        if self.is_full_word(tokens_str, i, word, word_tokens):
            word = self.introduce_typo(word)
        word_tokens = self.tokenizer(word, return_tensors='pt', truncation=True, max_length=max_length).input_ids[0][1:]
        context = self.tokenizer.decode(tokens[:i]) if self.add_context else ""
        combined_text = context + word

        tokenized_combined_text = self.tokenizer(combined_text, return_tensors='pt', truncation=True,
                                                 max_length=max_length).to(device)
        j = len(tokenized_combined_text.input_ids[0]) - 1 if self.add_context else len(tokenized_combined_text.input_ids[0]) - 1 + i
        return j, word_tokens, word, context, tokenized_combined_text, combined_text, original_word

    def get_next_full_word_separated(self, tokens, i, max_length=1000, device='cuda'):
        tokens_str = self.tokenizer.convert_ids_to_tokens(tokens)
        word_tokens = [tokens[i]]
        word = self.tokenizer.decode(word_tokens)
        original_word = word
        if self.is_full_word(tokens_str, i, word, word_tokens):
            word = torch.tensor(self.separate_word(word)).unsqueeze(0)
        else:
            word = word_tokens[0].unsqueeze(0).unsqueeze(0)
        context = self.tokenizer.decode(tokens[:i]) if self.add_context else ""
        tokenized_combined_text = self.tokenizer(context, return_tensors='pt', truncation=True,
                                                 max_length=max_length).to(device)
        print(tokenized_combined_text.input_ids)
        print(word)
        tokenized_combined_text.input_ids = torch.cat((tokenized_combined_text.input_ids, word), dim=1)
        word_tokens = word
        j = i+1
        return j, word_tokens, word, context, tokenized_combined_text, self.tokenizer.decode(tokenized_combined_text.input_ids[0]), original_word

    def is_alpha_not_prefix(self, token):
        return (not self.tokenizer.convert_ids_to_tokens(token.item()).startswith(self.whitespace_token)
                and self.tokenizer.convert_ids_to_tokens(token.item()).isalpha())

    def introduce_typo(self, word, typo_type=None):
        letters = 'abcdefghijklmnopqrstuvwxyz'
        if typo_type is None:
            typo_type = random.choice(["substitution", "deletion", "insertion", "transposition"])

        if typo_type == "substitution":
            position = random.randint(1, len(word) - 1)
            original_char = word[position]
            typo_char = random.choice([c for c in letters if c != original_char])
            return word[:position] + typo_char + word[position + 1:]
        elif typo_type == "deletion":
            position = random.randint(1, len(word) - 1)
            return word[:position] + word[position + 1:]
        elif typo_type == "insertion":
            position = random.randint(1, len(word) - 1)
            typo_char = random.choice(letters)
            return word[:position] + typo_char + word[position:]
        elif typo_type == "transposition":
            position = random.randint(1, len(word) - 2)
            return word[:position] + word[position + 1] + word[position] + word[position + 2:]
        else:
            return word

    def separate_word(self, word):
        character_tokens = [self.tokenizer.encode(f'\n{char}')[-1] for char in ''.join(word)]
        character_tokens = character_tokens[3:]
        return character_tokens

    def is_full_word(self, token_str, i, token, word_tokens):
        next_token = self.tokenizer.decode(word_tokens[i + 1]) if i + 1 < len(word_tokens) else ""
        return (token[1:].isalpha() and
                len(token) > 5 and
                token_str[i].startswith(self.whitespace_token) and
                not next_token.isalpha())
