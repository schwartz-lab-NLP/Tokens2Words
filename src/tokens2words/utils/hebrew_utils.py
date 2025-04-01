import os
from collections import Counter
import re
from transformers import AutoTokenizer

from ..utils.file_utils import save_string_list_to_file


# Define the Hebrew alphabet (with final letters)
standard_letters = ['א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'כ', 'ל', 'מ', 'נ', 'ס', 'ע', 'פ', 'צ', 'ק', 'ר',
                    'ש', 'ת']
final_letters = ['ך', 'ם', 'ן', 'ף', 'ץ']
all_letters = standard_letters + final_letters


def filter_strings_with_hebrew_letters(list_of_strings):
    # Compile a regex pattern to match any Hebrew letter (Unicode range for Hebrew letters)
    hebrew_pattern = re.compile(r'[א-ת]')

    # Filter the strings that contain at least one Hebrew letter
    return [s for s in list_of_strings if hebrew_pattern.search(s)]


def get_hebrew_tokenizer_map(model_name):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create dictionary mapping token representations to letters
    tokenizer_map = {}
    for letter in all_letters:
        tokens = tokenizer.tokenize(letter)
        token_str = "".join(tokens)  # Join tokens into a single string
        tokenizer_map[token_str] = letter

    pattern = re.compile('|'.join(re.escape(key) for key in tokenizer_map.keys()))

    return tokenizer_map, pattern


def translate_string(bad_str, pattern, translation_map):
    # Function to replace each match with the corresponding readable character
    replace_match = lambda match: translation_map[match.group(0)]

    # Use re.sub to replace all matches in the string
    return pattern.sub(replace_match, bad_str)


# Generate all two-letter combinations
def generate_two_letter_combinations():
    combinations = []

    for first in all_letters:
        for second in all_letters:
            # Ensure that final forms only appear in the second position
            if first in final_letters:
                continue  # Skip if a final form is at the start

            combinations.append(first + second)

    return combinations


def calculate_letter_ngrams_batch(batch, n, text_col, pattern):
    # Filter text to include only the specified language's letters
    filtered_texts = ["".join(re.findall(pattern, text)) for text in batch[text_col]]

    # Generate n-grams and count frequencies for the batch
    batch_frequencies = Counter()
    for filtered_text in filtered_texts:
        batch_frequencies.update(
            filtered_text[i:i + n] for i in range(len(filtered_text) - n + 1)
        )
    return {"ngrams": batch_frequencies}


def calculate_letter_ngrams_from_hf_dataset(dataset, n, text_col="text", language="Hebrew", batch_size=1000):
    # Define language-specific character sets
    language_characters = {
        "Hebrew": r"א-תךםןףץ",
        "English": r"a-zA-Z"
    }

    if language not in language_characters:
        raise ValueError(f"Unsupported language: {language}")

    # Compile the regex pattern for filtering text
    pattern = f"[{language_characters[language]}]+"

    # Use map to process the dataset in batches
    result = dataset.map(
        lambda batch: calculate_letter_ngrams_batch(batch, n, text_col, pattern),
        batched=True,
        batch_size=batch_size
    )

    # Aggregate results from all batches
    overall_frequencies = Counter()
    for batch_frequencies in result["ngrams"]:
        overall_frequencies.update(batch_frequencies)

    return overall_frequencies


def calculate_letter_ngrams(text, n, language="Hebrew"):
    # Define language-specific character sets
    language_characters = {
        "Hebrew": r"א-ת",
        "English": r"a-zA-Z"
    }

    if language not in language_characters:
        raise ValueError(f"Unsupported language: {language}")

    # Filter text to include only the specified language's letters
    pattern = f"[{language_characters[language]}]+"
    filtered_text = "".join(re.findall(pattern, text))

    # Generate n-grams
    ngrams = [filtered_text[i:i + n] for i in range(len(filtered_text) - n + 1)]

    # Count n-grams frequencies
    frequencies = Counter(ngrams)
    return frequencies


if __name__ == "__main__":
    output_dir = "/cs/labs/roys/yuval.reif/Tokens2Words/word_lists/"
    filename = "hebrew_all_2_letter_pairs.txt"
    combinations = generate_two_letter_combinations()
    save_string_list_to_file(combinations, os.path.join(output_dir, filename))
    for combo in combinations:
        print(combo)
