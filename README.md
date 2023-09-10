# Natural Portuguese Language Benchmark (Napolab)

<p align="center">
  <img width="300" height="300" src="https://raw.githubusercontent.com/ruanchaves/napolab/main/images/ideogram_ai_logo.png">
</p>

The **Napolab** is your go-to collection of Portuguese datasets with the following characteristics:

* 🌿 **Natural**: As much as possible, datasets consist of natural Portuguese text or professionally translated text.
* ✅ **Reliable**: Metrics correlate reliably with human judgments (accuracy, F1 score, Pearson correlation, etc.).
* 🌐 **Public**: Every dataset is available through a public link.
* 👩‍🔧 **Human**: Expert human annotations only. No automatic or unreliable annotations.
* 🎓 **General**: No domain-specific knowledge or advanced preparation is needed to solve dataset tasks.

Napolab currently includes the following datasets:

| :---: |  :---:  |  :---: |  :---: |
|[assin](https://huggingface.co/datasets/assin) | [assin2](https://huggingface.co/datasets/assin2) | [rerelem](https://huggingface.co/datasets/ruanchaves/rerelem) | [hatebr](https://huggingface.co/datasets/ruanchaves/hatebr)|
|[reli-sa](https://huggingface.co/datasets/ruanchaves/reli-sa) | [faquad-nli](https://huggingface.co/datasets/ruanchaves/faquad-nli) | [porsimplessent](https://huggingface.co/datasets/ruanchaves/porsimplessent) | |

**💡 Contribute**: We're open to expanding Napolab! Suggest additions in the issues. Plus, if you've evaluated models on this benchmark, we'd love to hear about it, especially results from recent LLMs.

🌍 For broader accessibility, all datasets have translations in **Catalan, English, Galician and Spanish** using the `facebook/nllb-200-1.3B model` via [Easy-Translate](https://github.com/ikergarcia1996/Easy-Translate).

# Quick Start 🚀

The simplest way to use the Napolab benchmark is to run the commands:

```bash
pip install napolab
python -m napolab
```

This fetches all datasets from Hugging Face Hub and saves them as CSVs in your current folder.

For the `datasets` library format:

```python
from napolab import load_napolab_benchmark

napolab = load_napolab_benchmark(include_translations=True)

benchmark = napolab["datasets"]
translated_benchmark = napolab["translations"]
```

## 🤖 Models

We've made several models, fine-tuned on this benchmark, available on Hugging Face Hub:

| Datasets                     | mDeBERTa v3                                                                                                    | BERT Large                                                                                                    | BERT Base                                                                                                     |
|:----------------------------:|:--------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------:|
| **ASSIN 2 - STS**            | [Link](https://huggingface.co/ruanchaves/mdeberta-v3-base-assin2-similarity)                                   | [Link](https://huggingface.co/ruanchaves/bert-large-portuguese-cased-assin2-similarity)                       | [Link](https://huggingface.co/ruanchaves/bert-base-portuguese-cased-assin2-similarity)                       |
| **ASSIN 2 - RTE**            | [Link](https://huggingface.co/ruanchaves/mdeberta-v3-base-assin2-entailment)                                  | [Link](https://huggingface.co/ruanchaves/bert-large-portuguese-cased-assin2-entailment)                       | [Link](https://huggingface.co/ruanchaves/bert-base-portuguese-cased-assin2-entailment)                       |
| **ASSIN - STS**              | [Link](https://huggingface.co/ruanchaves/mdeberta-v3-base-assin-similarity)                                   | [Link](https://huggingface.co/ruanchaves/bert-large-portuguese-cased-assin-similarity)                        | [Link](https://huggingface.co/ruanchaves/bert-base-portuguese-cased-assin-similarity)                        |
| **ASSIN - RTE**              | [Link](https://huggingface.co/ruanchaves/mdeberta-v3-base-assin-entailment)                                   | [Link](https://huggingface.co/ruanchaves/bert-large-portuguese-cased-assin-entailment)                        | [Link](https://huggingface.co/ruanchaves/bert-base-portuguese-cased-assin-entailment)                        |
| **HateBR**                   | [Link](https://huggingface.co/ruanchaves/mdeberta-v3-base-hatebr)                                             | [Link](https://huggingface.co/ruanchaves/bert-large-portuguese-cased-hatebr)                                 | [Link](https://huggingface.co/ruanchaves/bert-base-portuguese-cased-hatebr)                                  |
| **FaQUaD-NLI**               | [Link](https://huggingface.co/ruanchaves/mdeberta-v3-base-faquad-nli)                                         | [Link](https://huggingface.co/ruanchaves/bert-large-portuguese-cased-faquad-nli)                             | [Link](https://huggingface.co/ruanchaves/bert-base-portuguese-cased-faquad-nli)                              |
| **PorSimplesSent**           | [Link](https://huggingface.co/ruanchaves/mdeberta-v3-base-porsimplessent)                                     | [Link](https://huggingface.co/ruanchaves/bert-large-portuguese-cased-porsimplessent)                         | [Link](https://huggingface.co/ruanchaves/bert-base-portuguese-cased-porsimplessent)                          |


For model fine-tuning details and benchmark results, visit [EVALUATION.md](EVALUATION.md). 

## 🎮 Demos

Experience our fine-tuned models on [Hugging Face Spaces](https://huggingface.co/ruanchaves). Check out:

* [Portuguese Offensive Language Detection](https://ruanchaves-portuguese-offensive-language-de-d4d0507.hf.space)
* [Portuguese Question Answering](https://ruanchaves-portuguese-question-answering.hf.space)
* [Portuguese Semantic Similarity](https://ruanchaves-portuguese-semantic-similarity.hf.space)
* [Portuguese Textual Entailment](https://ruanchaves-portuguese-textual-entailment.hf.space)
* [Portuguese Text Simplification](https://ruanchaves-portuguese-text-simplification.hf.space)

## Citation

Our research is ongoing, and we are currently working on describing our experiments in a paper, which will be published soon. In the meanwhile, if you would like to cite our work or models before the publication of the paper, please use the following BibTeX citation for this repository: 

```
@software{Chaves_Rodrigues_napolab_2023,
author = {Chaves Rodrigues, Ruan and Tanti, Marc and Agerri, Rodrigo},
doi = {10.5281/zenodo.7781848},
month = {3},
title = {{Natural Portuguese Language Benchmark (Napolab)}},
url = {https://github.com/ruanchaves/napolab},
version = {1.0.0},
year = {2023}
}
```

## Disclaimer

The HateBR dataset, including all its components, is provided strictly for academic and research purposes. The use of the HateBR dataset for any commercial or non-academic purpose is expressly prohibited without the prior written consent of [SINCH](https://www.sinch.com/).