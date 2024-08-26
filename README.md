# 🌎 Natural Portuguese Language Benchmark (Napolab)

The [**Napolab**](https://huggingface.co/datasets/ruanchaves/napolab) is your go-to collection of Portuguese datasets for the evaluation of Large Language Models.

## 📊 Napolab for Large Language Models (LLMs)

A format of Napolab specifically designed for researchers experimenting with Large Language Models (LLMs) is now available. This format includes two main fields:

* **Prompt**: The input prompt to be fed into the LLM.
* **Answer**: The expected classification output label from the LLM, which is always a number between 0 and 5.

The dataset in this format can be accessed at [https://huggingface.co/datasets/ruanchaves/napolab](https://huggingface.co/datasets/ruanchaves/napolab). If you’ve used Napolab for LLM evaluations, please share your findings with us!

## Guidelines

Napolab adopts the following guidelines for the inclusion of datasets:

* 🌿 **Natural**: As much as possible, datasets consist of natural Portuguese text or professionally translated text.
* ✅ **Reliable**: Metrics correlate reliably with human judgments (accuracy, F1 score, Pearson correlation, etc.).
* 🌐 **Public**: Every dataset is available through a public link.
* 👩‍🔧 **Human**: Expert human annotations only. No automatic or unreliable annotations.
* 🎓 **General**: No domain-specific knowledge or advanced preparation is needed to solve dataset tasks.

[Napolab](https://huggingface.co/datasets/ruanchaves/napolab) currently includes the following datasets:

| | | |
| :---: |  :---:  |  :---: |
|[assin](https://huggingface.co/datasets/assin) | [assin2](https://huggingface.co/datasets/assin2) | [rerelem](https://huggingface.co/datasets/ruanchaves/rerelem)|
|[hatebr](https://huggingface.co/datasets/ruanchaves/hatebr)| [reli-sa](https://huggingface.co/datasets/ruanchaves/reli-sa) | [faquad-nli](https://huggingface.co/datasets/ruanchaves/faquad-nli) |
|[porsimplessent](https://huggingface.co/datasets/ruanchaves/porsimplessent) | | |

**💡 Contribute**: We're open to expanding Napolab! Suggest additions in the issues. For more information, read our [CONTRIBUTING.md](CONTRIBUTING.md).

🌍 For broader accessibility, all datasets have translations in **Catalan, English, Galician and Spanish** using the `facebook/nllb-200-1.3B model` via [Easy-Translate](https://github.com/ikergarcia1996/Easy-Translate).

## Leaderboards 

The [Open PT LLM Leaderboard](https://huggingface.co/spaces/eduagarcia/open_pt_llm_leaderboard) incorporates datasets from Napolab. 

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
