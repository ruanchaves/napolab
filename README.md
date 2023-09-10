# Natural Portuguese Language Benchmark (Napolab)

![](ideogram_ai_logo.png)

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

In addition to making these datasets available in Portuguese, all datasets have also been automatically translated into English, Spanish, Galician, and Catalan using the `facebook/nllb-200-1.3B` model through the [Easy-Translate](https://github.com/ikergarcia1996/Easy-Translate) library.

The small `napolab.py` script in this repository includes a convenient `DatasetLoader`, a thin wrapper around the `datasets` library, to access the datasets available on the Hugging Face Hub:

```python
from napolab import DatasetLoader

loader = DatasetLoader()

>>> loader.DATASET_NAMES
['assin', 'assin2', 'rerelem', 'hatebr', 'reli-sa', 'faquad-nli', 'porsimplessent']

>>> loader.SUPPORTED_LANGUAGES
{'portuguese': 'por', 'english': 'eng', 'spanish': 'spa', 'catalan': 'cat', 'galician': 'glg'}

>>> loader.SUPPORTED_VARIANTS # This field is applicable only to the ASSIN dataset.
['full', 'br', 'pt']
```

The benchmark is open to expansion and suggestions of future additions to this list are also welcome in the issues.

Besides making these datasets available in Portuguese, all datasets have also been automatically translated into English, Spanish, Galician and Catalan using the `facebook/nllb-200-1.3B` model through the [Easy-Translate](https://github.com/ikergarcia1996/Easy-Translate) library.

The small `napolab.py` script in this repository has a convenient `DatasetLoader`, a thin wrapper around the `datasets` library, to access the datasets made available on the Hugging Face Hub:

```python
from napolab import DatasetLoader

loader = DatasetLoader()

>>> loader.DATASET_NAMES
['assin', 'assin2', 'rerelem', 'hatebr', 'reli-sa', 'faquad-nli', 'porsimplessent']

>>> loader.SUPPORTED_LANGUAGES
{'portuguese': 'por', 'english': 'eng', 'spanish': 'spa', 'catalan': 'cat', 'galician': 'glg'}

>>> loader.SUPPORTED_VARIANTS # This field is applicable only to the ASSIN dataset.
['full', 'br', 'pt']
```

A few examples of its usage:

```python
from napolab import DatasetLoader

loader = DatasetLoader()

datasets = {}
# This will load all datasets that make up the Napolab benchmark in the Portuguese language.
for dataset_name in loader.DATASET_NAMES:
    if dataset_name in ["assin", "assin2"]:
        datasets[f"{dataset_name}-rte"] = loader.load(dataset_name, task="rte")
        datasets[f"{dataset_name}-sts"] = loader.load(dataset_name, task="sts")
    else:
        datasets[dataset_name] = loader.load(dataset_name)

# It is also possible to load only the Brazilian Portuguese or European Portuguese portion of ASSIN instead of loading both portions as a single dataset:

datasets["assin-rte-ptbr"] = loader.load("assin", task="rte", hf_args=["ptbr"])
datasets["assin-rte-ptpt"] = loader.load("assin", task="rte", hf_args=["ptpt"])
datasets["assin-sts-ptbr"] = loader.load("assin", task="sts", hf_args=["ptbr"])
datasets["assin-sts-ptpt"] = loader.load("assin", task="sts", hf_args=["ptpt"])

# Let's also load all translated datasets:

translated_datasets = {}
for language in ["english", "spanish", "galician", "catalan"]:
    if language not in translated_datasets:
        translated_datasets[language] = {}
    for dataset_name in loader.DATASET_NAMES:
        if dataset_name in ["assin", "assin2"]:
            # Load the full splits
            translated_datasets[language][f"{dataset_name}-rte"] = loader.load(dataset_name, task="rte", language=language)
            translated_datasets[language][f"{dataset_name}-sts"] = loader.load(dataset_name, task="sts", language=language)
            if dataset_name == "assin":
                # Alternatively, for the ASSIN dataset, load just one variant
                translated_datasets[language]["assin-rte-ptbr"] = loader.load("assin", task="rte", variant="br")
                translated_datasets[language]["assin-rte-ptpt"] = loader.load("assin", task="rte", variant="pt")
                translated_datasets["assin-sts-ptbr"] = loader.load("assin", task="sts", variant="br")
                translated_datasets["assin-sts-ptpt"] = loader.load("assin", task="sts", variant="pt")            
        else:
            translated_datasets[language][dataset_name] = loader.load(dataset_name, language=language)
```

The `load` method of the `DataLoader` object supports the following arguments:

* `dataset_name`: The name of the dataset to be loaded (from `loader.DATASET_NAMES`).
* `language`: The language of the dataset to be loaded, either the name of the language or the language code. Defaults to `"por"`.
* `variant`: Applicable only to the ASSIN dataset. The variant split to be loaded (either "br" for Brazilian Portuguese or "pt" for European Portuguese).
* `clean`: Whether to return the dataset in a cleaned format (i.e., removing irrelevant fields and renaming the fields before returning). If set to `True`, only the fields **sentence1**, **sentence2**, and **label** will be returned for datasets with sentence pairs, or **sentence1** and **label** otherwise. Defaults to `True`.
* `task`: Applicable only to the ASSIN and ASSIN 2 datasets. The task to return. Accepted values are "entailment" or "rte" for the RTE task, and "similarity" or "sts" for the STS task.
* `hf_args`: Extra arguments that will be passed to the `datasets.load_dataset` call.
* `hf_kwargs`: Keyword arguments that will be passed to the `datasets.load_dataset` call.

## Demos

All of our fine-tuned models have been integrated into an appropriate [Hugging Face Space](https://huggingface.co/ruanchaves).

Interact with our models in your browser by exploring our demos:

* [Portuguese Offensive Language Detection](https://ruanchaves-portuguese-offensive-language-de-d4d0507.hf.space)
* [Portuguese Question Answering](https://ruanchaves-portuguese-question-answering.hf.space)
* [Portuguese Semantic Similarity](https://ruanchaves-portuguese-semantic-similarity.hf.space)
* [Portuguese Textual Entailment](https://ruanchaves-portuguese-textual-entailment.hf.space)
* [Portuguese Text Simplification](https://ruanchaves-portuguese-text-simplification.hf.space)

## Models

### Summary of the Fine-Tuning procedure

* **Step 1**. Hyperparameter optimization is performed using quasi-random search based on Google's [Deep Learning Playbook](https://github.com/google-research/tuning_playbook) instructions. The best learning rate, weight decay, and adam beta1 parameters for each Transformer model on each task are identified.

* **Step 2**. The best hyperparameters from step 1 are used to fine-tune each model 40 times with different random seeds for up to one epoch ( [Dodge et al. (2020)](https://arxiv.org/abs/2002.06305) ). The 10 best models after the first epoch are selected for the next step.

* **Step 3**. The top 10 models from step 2 are fine-tuned for 20 epochs, generating predictions for the test set ( [Mosbach et al. (2021)](https://arxiv.org/abs/2006.04884) ). We select the model that is closest to the average of predictions (for regression tasks) or the mode of predictions (for classification tasks). This final model is then uploaded to the Hugging Face Hub and displayed in the tables below.

### Links and Results  

Our fine-tuning procedure has achieved results that are either slightly superior or at the same level as the previous state-of-the-art (if any).
Below is a summary of the results achieved on each dataset.

#### [ASSIN 2](huggingface.co/datasets/assin2) - STS ( Semantic Textual Similarity )

| Model                                                    | Pearson | MSE  |
|----------------------------------------------------------|---------|------|
| **ruanchaves/bert-large-portuguese-cased-assin2-similarity** | **0.86**    | **0.48** |
| Previous SOTA ( for Pearson ) - [Souza et al. (2020)](https://link.springer.com/chapter/10.1007/978-3-030-61377-8_28/tables/2)                           | 0.852   | 0.50 |
| SOTA ( for MSE ) - [Stilingue](https://link.springer.com/chapter/10.1007/978-3-030-41505-1_39/tables/2)                               | 0.817   | **0.47** |
| [ruanchaves/mdeberta-v3-base-assin2-similarity](https://huggingface.co/ruanchaves/mdeberta-v3-base-assin2-similarity)            | 0.847   | 0.62 |
| [ruanchaves/bert-base-portuguese-cased-assin2-similarity](https://huggingface.co/ruanchaves/bert-base-portuguese-cased-assin2-similarity)    | 0.843   | 0.54 |

#### [ASSIN 2](huggingface.co/datasets/assin2) - RTE ( Recognizing Textual Entailment )

| Model                                                    | Accuracy | F1  |
|----------------------------------------------------------|---------|------|
| **[ruanchaves/bert-large-portuguese-cased-assin2-entailment](https://huggingface.co/ruanchaves/bert-large-portuguese-cased-assin2-entailment)** | **0.90**    | **0.90** |
| [ruanchaves/mdeberta-v3-base-assin2-entailment](https://huggingface.co/ruanchaves/mdeberta-v3-base-assin2-entailment)           | **0.90**   | **0.90** |
| Previous SOTA                           | **0.90**  | **0.90** |
| [ruanchaves/bert-base-portuguese-cased-assin2-entailment](https://huggingface.co/ruanchaves/bert-base-portuguese-cased-assin2-entailment) | 0.88   | 0.88 |

#### [ASSIN](https://huggingface.co/datasets/assin) - STS ( Semantic Textual Similarity )

| Model                                                    | Pearson | MSE  |
|----------------------------------------------------------|---------|------|
| [ruanchaves/bert-large-portuguese-cased-assin-similarity](https://huggingface.co/ruanchaves/bert-large-portuguese-cased-assin-similarity) | 0.859    | 0.3 |
| [ruanchaves/mdeberta-v3-base-assin-similarity](https://huggingface.co/ruanchaves/mdeberta-v3-base-assin-similarity)            | 0.855  | 0.39 |
| [ruanchaves/bert-base-portuguese-cased-assin-similarity](https://huggingface.co/ruanchaves/bert-base-portuguese-cased-assin-similarity)  | 0.847   | 0.33 |

#### [ASSIN](https://huggingface.co/datasets/assin) - RTE ( Recognizing Textual Entailment )

| Model                                                    | Accuracy | F1  |
|----------------------------------------------------------|---------|------|
| [ruanchaves/mdeberta-v3-base-assin-entailment](https://huggingface.co/ruanchaves/mdeberta-v3-base-assin-entailment)           | 0.927   | 0.862 |
| [ruanchaves/bert-large-portuguese-cased-assin-entailment](https://huggingface.co/ruanchaves/bert-large-portuguese-cased-assin-entailment) | 0.92    | 0.828 |
| [ruanchaves/bert-base-portuguese-cased-assin-entailment](https://huggingface.co/ruanchaves/bert-base-portuguese-cased-assin-entailment) | 0.92   | 0.827 |

#### [HateBR](https://huggingface.co/datasets/ruanchaves/hatebr) ( Offensive Language Detection )


| Model                                                    | Accuracy | F1  |
|----------------------------------------------------------|---------|------|
| [ruanchaves/bert-large-portuguese-cased-hatebr](https://huggingface.co/ruanchaves/bert-large-portuguese-cased-hatebr) | 0.928    | 0.928 |
| [ruanchaves/mdeberta-v3-base-hatebr](https://huggingface.co/ruanchaves/mdeberta-v3-base-hatebr)          | 0.916 | 0.916 |
| [ruanchaves/bert-base-portuguese-cased-hatebr](https://huggingface.co/ruanchaves/bert-base-portuguese-cased-hatebr) | 0.914 | 0.914 |

#### [FaQUaD-NLI](https://huggingface.co/datasets/ruanchaves/faquad-nli) ( Question Answering )

| Model                                                    | Accuracy | F1  |
|----------------------------------------------------------|---------|------|
| [ruanchaves/bert-large-portuguese-cased-faquad-nli](https://huggingface.co/ruanchaves/bert-large-portuguese-cased-faquad-nli) | 0.929  | 0.93 |
| [ruanchaves/mdeberta-v3-base-faquad-nli](https://huggingface.co/ruanchaves/mdeberta-v3-base-faquad-nli)          | 0.926 | 0.926 |
| [ruanchaves/bert-base-portuguese-cased-faquad-nli](https://huggingface.co/ruanchaves/bert-base-portuguese-cased-faquad-nli) | 0.92 | 0.883 |

#### [PorSimplesSent](https://huggingface.co/datasets/ruanchaves/porsimplessent) ( Text Simplification )

| Model                                                    | Accuracy | F1  |
|----------------------------------------------------------|---------|------|
| [ruanchaves/mdeberta-v3-base-porsimplessent](https://huggingface.co/ruanchaves/mdeberta-v3-base-porsimplessent)          | 0.96 | 0.956 |
| [ruanchaves/bert-base-portuguese-cased-porsimplessent](https://huggingface.co/ruanchaves/bert-base-portuguese-cased-porsimplessent) | 0.942 | 0.937 |
| [ruanchaves/bert-large-portuguese-cased-porsimplessent](https://huggingface.co/ruanchaves/bert-large-portuguese-cased-porsimplessent)  | 0.921  | 0.913 |

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
