# Natural Portuguese Language Benchmark (Napolab)

<p align="center">
  <img width="300" height="300" src="https://raw.githubusercontent.com/ruanchaves/napolab/main/images/ideogram_ai_logo.png">
</p>

The Natural Portuguese Language Benchmark (Napolab) is designed to be a collection of Portuguese datasets that are:

* **Natural**: Composed, as far as possible, of natural Portuguese text or text that has undergone professional translation.
* **Reliable**: The standard metrics used to evaluate the tasks on the dataset should reliably correlate with human judgments (such as accuracy, F1 score, Pearson correlation coefficient, etc.).
* **Public**: All datasets must be available and downloadable via a publicly accessible link.
* **Human**: All datasets must be annotated by human experts without any automation.
* **General**: The datasets should not necessitate domain-specific knowledge or any preparation for an educated Portuguese speaker to solve the suggested tasks.

This repository contains links to demos, models fine-tuned on the benchmark, and instructions for using the datasets in the most convenient manner. The benchmark currently includes the following datasets:

* [assin](https://huggingface.co/datasets/assin)
* [assin2](https://huggingface.co/datasets/assin2)
* [rerelem](https://huggingface.co/datasets/ruanchaves/rerelem)
* [hatebr](https://huggingface.co/datasets/ruanchaves/hatebr)
* [reli-sa](https://huggingface.co/datasets/ruanchaves/reli-sa)
* [faquad-nli](https://huggingface.co/datasets/ruanchaves/faquad-nli)
* [porsimplessent](https://huggingface.co/datasets/ruanchaves/porsimplessent)

We are open to expanding the benchmark, and suggestions for future additions are welcome in the issues. We also welcome evaluation results from any models on this benchmark, and we are particularly curious about the outcomes of evaluating recent LLMs on these datasets.

In addition to making these datasets available in Portuguese, all datasets have also been automatically translated into **English, Spanish, Galician, and Catalan** using the `facebook/nllb-200-1.3B` model through the [Easy-Translate](https://github.com/ikergarcia1996/Easy-Translate) library.

The simplest way to use the Napolab benchmark is to run the commands:

```bash
pip install napolab
python -m napolab
```

This will download all the datasets from the Hugging Face Hub and save them in CSV format under the current folder.

If you would rather deal with the datasets in the format of the `datasets` library, you can also load the benchmark in this way:

```python
from napolab import load_napolab_benchmark

napolab = load_napolab_benchmark(include_translations=True)

benchmark = napolab["datasets"]
translated_benchmark = napolab["translations"]
```

## Demos

All of our fine-tuned models have been integrated into an appropriate [Hugging Face Space](https://huggingface.co/ruanchaves).

Interact with our models in your browser by exploring our demos:

* [Portuguese Offensive Language Detection](https://ruanchaves-portuguese-offensive-language-de-d4d0507.hf.space)
* [Portuguese Question Answering](https://ruanchaves-portuguese-question-answering.hf.space)
* [Portuguese Semantic Similarity](https://ruanchaves-portuguese-semantic-similarity.hf.space)
* [Portuguese Textual Entailment](https://ruanchaves-portuguese-textual-entailment.hf.space)
* [Portuguese Text Simplification](https://ruanchaves-portuguese-text-simplification.hf.space)

## Models

We are making a few models fine-tuned on the datasets in this benchmark available on the Hugging Face Hub. 

| Datasets                     | mDeBERTa v3                                                                                                    | BERT Large                                                                                                    | BERT Base                                                                                                     |
|:----------------------------:|:--------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------:|
| **ASSIN 2 - STS**            | [Link](https://huggingface.co/ruanchaves/mdeberta-v3-base-assin2-similarity)                                   | [Link](https://huggingface.co/ruanchaves/bert-large-portuguese-cased-assin2-similarity)                       | [Link](https://huggingface.co/ruanchaves/bert-base-portuguese-cased-assin2-similarity)                       |
| **ASSIN 2 - RTE**            | [Link](https://huggingface.co/ruanchaves/mdeberta-v3-base-assin2-entailment)                                  | [Link](https://huggingface.co/ruanchaves/bert-large-portuguese-cased-assin2-entailment)                       | [Link](https://huggingface.co/ruanchaves/bert-base-portuguese-cased-assin2-entailment)                       |
| **ASSIN - STS**              | [Link](https://huggingface.co/ruanchaves/mdeberta-v3-base-assin-similarity)                                   | [Link](https://huggingface.co/ruanchaves/bert-large-portuguese-cased-assin-similarity)                        | [Link](https://huggingface.co/ruanchaves/bert-base-portuguese-cased-assin-similarity)                        |
| **ASSIN - RTE**              | [Link](https://huggingface.co/ruanchaves/mdeberta-v3-base-assin-entailment)                                   | [Link](https://huggingface.co/ruanchaves/bert-large-portuguese-cased-assin-entailment)                        | [Link](https://huggingface.co/ruanchaves/bert-base-portuguese-cased-assin-entailment)                        |
| **HateBR**                   | [Link](https://huggingface.co/ruanchaves/mdeberta-v3-base-hatebr)                                             | [Link](https://huggingface.co/ruanchaves/bert-large-portuguese-cased-hatebr)                                 | [Link](https://huggingface.co/ruanchaves/bert-base-portuguese-cased-hatebr)                                  |
| **FaQUaD-NLI**               | [Link](https://huggingface.co/ruanchaves/mdeberta-v3-base-faquad-nli)                                         | [Link](https://huggingface.co/ruanchaves/bert-large-portuguese-cased-faquad-nli)                             | [Link](https://huggingface.co/ruanchaves/bert-base-portuguese-cased-faquad-nli)                              |
| **PorSimplesSent**           | [Link](https://huggingface.co/ruanchaves/mdeberta-v3-base-porsimplessent)                                     | [Link](https://huggingface.co/ruanchaves/bert-large-portuguese-cased-porsimplessent)                         | [Link](https://huggingface.co/ruanchaves/bert-base-portuguese-cased-porsimplessent)                          |


More details about how the models have been fine-tuned and their results on the benchmark can be found under [EVALUATION.md](EVALUATION.md). 

## Citation

Our research is ongoing, and we are currently working on describing our experiments in a paper, which will be published soon. In the meanwhile, if you would like to cite our work or models before the publication of the paper, please use the following BibTeX citation for this repository: 

```
@software{Chaves_Rodrigues_eplm_2023,
author = {Chaves Rodrigues, Ruan and Tanti, Marc and Agerri, Rodrigo},
doi = {10.5281/zenodo.7781848},
month = {3},
title = {{Evaluation of Portuguese Language Models}},
url = {https://github.com/ruanchaves/napolab},
version = {1.0.0},
year = {2023}
}
```