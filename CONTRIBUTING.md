# Welcome, Contributors! 🌟

Thank you for considering contributing to the **Natural Portuguese Language Benchmark (Napolab)**. By participating in this project, you agree to abide by our [code of conduct](CODE_OF_CONDUCT.md).

## Table of Contents
- [Adding Datasets 📊](#adding-datasets-📊)
  - [Guidelines](#guidelines)
  - [Step-by-Step Guide](#step-by-step-guide)
- [Sharing Model Evaluations](#sharing-model-evaluations)

---

## Adding Datasets 📊

### Guidelines 

Suggest additions through [issues](https://github.com/ruanchaves/napolab/issues).

Check if your dataset adheres to our main characteristics required for admission to the benchmark, as described below. 

If you are unsure whether your dataset meets all the criteria, open an issue anyway, and we will review it to determine if it can be included in the benchmark. 

#### **Natural**

Whenever possible, datasets should consist of natural Portuguese text or professionally translated text. Datasets consisting mostly or entirely of automatically translated text should not be included in this benchmark.

#### **Reliable**

Metrics should correlate reliably with human judgments (e.g., accuracy, F1 score, Pearson correlation). For instance, tasks based on machine translation metrics (e.g., BLEU, METEOR) should not be included.

#### **Public**

Your dataset must be **publicly available for download via a public URL**, GitHub repository, or any other method that allows easy access. 

Datasets requiring data distribution through proprietary APIs (e.g., the Twitter API), or only available via a registration form, are not considered public for the purposes of this benchmark.

#### **Human**

Every record in your dataset must have been annotated by a human expert. Datasets that have been automatically labeled (e.g., labeling based on keyword matching) should not be admitted to this benchmark.

#### **General**

No domain-specific knowledge or advanced preparation should be necessary to solve dataset tasks. The tasks should be accessible to any educated speaker of the Portuguese language. This excludes, for instance, legal datasets, medical datasets, and college entrance exams.

### Step-by-Step Guide

If you wish to add a dataset to Napolab, simply suggest its addition through the [issues](https://github.com/ruanchaves/napolab/issues). Include relevant information, such as a link for download and an article describing its construction. 
After you open an issue, the project maintainers will handle its integration into the benchmark.

If you prefer to submit a pull request for direct integration into the benchmark, follow the guide below:

1. Open an [issue](https://github.com/ruanchaves/napolab/issues) suggesting the inclusion of your dataset at Napolab.
2. Write a dataset reader and make it available on the [Hugging Face Hub](https://huggingface.co/datasets). Your reader should, if necessary:
   - Define balanced train, validation, and test splits if the dataset authors haven't provided standard splits, ensuring data leakage is minimized.
   - Ensure the dataset is structured as a single-label text classification task. Note that many tasks can be adapted to this format; for instance, question-answering tasks can be converted into textual entailment tasks.
3. After creating a dataset reader on the Hugging Face Hub, submit a pull request to the Napolab repository to modify the library code at [napolab.py](https://github.com/ruanchaves/napolab/blob/main/src/napolab/napolab.py) to include your new dataset.

I must emphasize that there's no need for you to perform steps two and three yourself! You can merely open an issue and let the project maintainers handle steps two and three. Our goal is to make the dataset addition process as straightforward as possible.

## Sharing Model Evaluations

If you've evaluated models on the Natural Portuguese Language Benchmark (Napolab), especially recent Large Language Models (LLMs), please open an issue linking to your work. Noteworthy evaluations will be mentioned in the repository's README. Your detailed insights will greatly benefit the community's understanding of model performance on Portuguese tasks.

---

🎉 **Thank you for dedicating your time and expertise to improve Napolab. Happy contributing!** 🎉

---
