# Detokenization Analysis and Vocabulary Expansion Using PatchScopes and Logit Lens

This repository contains the implementation of a research project based on the paper: [From Tokens to Words: On the Inner Lexicon of LLMs
](https://arxiv.org/abs/2410.05864). The project focuses on analyzing the detokenization process and leveraging the inner representation of language models to expand a tokenizer's vocabulary effectively.

---

## Features

- **PatchScopes Analysis**: Investigate the inner workings of detokenization by examining representations at various layers of the language model.
- **Logit Lens Insights**: Use logit lens to trace back the input embedding space and understand the vocabulary projection.
- **Vocabulary Expansion**: Evaluate techniques for expanding tokenizer vocabularies using learned embeddings.

---

## File Structure

### Main Scripts

1. **`representation_translator.py`**  
   Implements utilities for translating and analyzing inner model representations during detokenization.

2. **`run_new_vocab_success_estimate.py`**  
   Evaluates the success rate of a new vocabulary against the original tokenizer's outputs.

3. **`run_patchscopes.py`**  
   Runs the PatchScopes mechanism to inspect inner layers of a language model during detokenization.

4. **`run_vocab_expansion_eval.py`**  
   Performs quantitative evaluation of vocabulary expansion strategies.

5. **`vocab_modifier.py`**  
   Includes methods for modifying and expanding tokenizer vocabularies using learned embeddings.

### Supporting Utilities

6. **`word_retriever.py`**  
   Retrieves candidate words for vocabulary expansion based on model embeddings.

7. **`processor.py`**  
   Handles preprocessing and intermediate computations for detokenization and evaluation tasks.

---

## Installation

### Prerequisites

- Python >= 3.8
- PyTorch >= 1.10
- Transformers >= 4.0
- numpy
- tqdm

### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/detokenization-analysis.git
   cd detokenization-analysis
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Analyze Detokenization with PatchScopes

```bash
python run_patchscopes.py --model <model_name> --input <input_text>
```

### 2. Evaluate Vocabulary Expansion

```bash
python run_vocab_expansion_eval.py --model <model_name> --vocab <vocab_file>
```

### 3. Estimate New Vocabulary Success

```bash
python run_new_vocab_success_estimate.py --model <model_name> --new_vocab <vocab_file>
```

---

## Results

This repository enables deep analysis of detokenization and vocabulary projection processes. Refer to the associated paper for detailed methodologies and findings.

---

## Citation

If you use this repository in your work, please cite:

```bibtex
@misc{kaplan2024tokenswordsinnerlexicon,
      title={From Tokens to Words: On the Inner Lexicon of LLMs}, 
      author={Guy Kaplan and Matanel Oren and Yuval Reif and Roy Schwartz},
      year={2024},
      eprint={2410.05864},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.05864}, 
}
```

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---
