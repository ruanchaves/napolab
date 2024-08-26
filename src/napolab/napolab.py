import datasets
from datasets import DatasetDict, Dataset
from typing import List, Dict, Union, Optional
import os
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s")

class DatasetLoader:
    """
    A class responsible for loading the datasets of the Napolab benchmark and performing various preprocessing operations.

    Attributes:
        DATASET_NAMES (list): List of supported dataset names.
        SELECTED_COLUMNS (dict): Columns to select from datasets.
        RENAME_COLUMNS_KEY (dict): Columns to rename for each dataset.
        SUPPORTED_LANGUAGES (dict): Supported languages with their respective codes.
        SUPPORTED_VARIANTS (list): List of supported dataset variants.
        ASSIN_SPLIT_RANGES (dict): Number of items for each split in ASSIN datasets.
    """

    DATASET_NAMES = [
        "assin",
        "assin2",
        "rerelem",
        "hatebr",
        "reli-sa",
        "faquad-nli",
        "porsimplessent"
    ]

    SELECTED_COLUMNS = {
        "assin": ["premise", "hypothesis", "relatedness_score", "entailment_judgment"],
        "assin2": ["premise", "hypothesis", "relatedness_score", "entailment_judgment"],
        "rerelem": ["sentence1", "sentence2", "label"],
        "hatebr": ["instagram_comments", "offensive_language"],
        "reli-sa": ["sentence", "label"],
        "faquad-nli": ["question", "answer", "label"],
        "porsimplessent": ["sentence1", "sentence2", "label"]
    }

    RENAME_COLUMNS_KEY = {
        "assin": {
            "premise": "sentence1",
            "hypothesis": "sentence2",
        },
        "assin2": {
            "premise": "sentence1",
            "hypothesis": "sentence2",
        },
        "hatebr": {
            "instagram_comments": "sentence1",
            "offensive_language": "label",
        },
        "reli-sa": {
            "sentence": "sentence1"
        },
        "faquad-nli": {
            "question": "sentence1",
            "answer": "sentence2"
        }
    }

    SUPPORTED_LANGUAGES = {
        "portuguese": "por",
        "english": "eng",
        "spanish": "spa",
        "catalan": "cat",
        "galician": "glg"
    }

    SUPPORTED_VARIANTS = ["full", "br", "pt"]
    
    ASSIN_SPLIT_RANGES = {
        "train": 2500,
        "validation": 500,
        "test": 2000
    }

    def validate_parameters(self, dataset_name: str, language: str, variant: str) -> None:
        """
        Validate the provided parameters for loading datasets.

        Args:
            dataset_name (str): Name of the dataset.
            language (str): Language of the dataset.
            variant (str): Variant of the dataset.

        Raises:
            ValueError: If the dataset name, language, or variant is not supported.
        """
        if dataset_name not in self.DATASET_NAMES:
            raise ValueError(f"Dataset name must be one of {self.DATASET_NAMES}")
        if variant not in self.SUPPORTED_VARIANTS:
            raise ValueError(f"Variant must be one of {self.SUPPORTED_VARIANTS}")
        if not (language in self.SUPPORTED_LANGUAGES or language in self.SUPPORTED_LANGUAGES.values()):
            raise ValueError(f"Language must be one of {self.SUPPORTED_LANGUAGES.keys()} or {self.SUPPORTED_LANGUAGES.values()}")

    def get_dataset_name(self, dataset_name: str, language: str) -> str:
        """
        Construct the dataset name based on dataset_name and language.

        Args:
            dataset_name (str): Name of the dataset.
            language (str): Language of the dataset.

        Returns:
            str: Complete dataset name.
        """
        if not (dataset_name.startswith("assin") and language in ["por", "portuguese"]):
            name = f"ruanchaves/{dataset_name}"
        else:
            name = dataset_name

        if language not in ["por", "portuguese"]:
            language_code = self.SUPPORTED_LANGUAGES.get(language, language)
            name = name + "_por_Latn_to_" + language_code + "_Latn"
        return name

    def apply_variant_filter(self, dataset: DatasetDict, name: str, variant: str) -> DatasetDict:
        """
        Apply a variant filter to the dataset, especially for ASSIN datasets.

        Args:
            dataset (DatasetDict): The dataset to filter.
            name (str): Name of the dataset.
            variant (str): Desired dataset variant.

        Returns:
            DatasetDict: Filtered dataset.
        """
        if variant != "full" and name.startswith("assin_"):
            for split in ["train", "validation", "test"]:
                split_range = self.ASSIN_SPLIT_RANGES[split]
                if variant == "br":
                    dataset[split] = dataset[split].select([i for i in range(0,split_range)])
                elif variant == "pt":
                    dataset[split] = dataset[split].select([i for i in range(split_range, split_range * 2)])
        return dataset

    def process_assin(self, dataset: Dataset, task: str) -> Dataset:
        """
        Process ASSIN datasets based on the given task.

        Args:
            dataset (Dataset): Dataset to process.
            task (str): Task type - one of ['entailment', 'rte', 'similarity', 'sts'].

        Returns:
            Dataset: Processed dataset.

        Raises:
            ValueError: If an unsupported task is provided for ASSIN datasets.
        """
        if task in ["entailment", "rte"]:
            dataset = dataset.rename_column("entailment_judgment", "label")
            dataset = dataset.remove_columns("relatedness_score")
        elif task in ["similarity", "sts"]:
            dataset = dataset.rename_column("relatedness_score", "label")
            dataset = dataset.remove_columns("entailment_judgment")
        else:
            raise ValueError("'task' argument value must be one of ['entailment', 'rte', 'similarity', 'sts'] for dataset of type ['assin','assin2']")
        return dataset

    def rename_dataset_columns(self, dataset: Dataset, dataset_name: str) -> Dataset:
        """
        Rename columns of the dataset based on the dataset name.

        Args:
            dataset (Dataset): Dataset with columns to rename.
            dataset_name (str): Name of the dataset.

        Returns:
            Dataset: Dataset with renamed columns.
        """
        if dataset_name in self.RENAME_COLUMNS_KEY:
            for column in self.RENAME_COLUMNS_KEY[dataset_name]:
                dataset = dataset.rename_column(column, self.RENAME_COLUMNS_KEY[dataset_name][column])
        return dataset

    def clean_dataset(self, dataset: DatasetDict, dataset_name: str, task: str) -> DatasetDict:
        """
        Clean the dataset by selecting, renaming, and processing columns.

        Args:
            dataset (DatasetDict): Dataset to clean.
            dataset_name (str): Name of the dataset.
            task (str): Task type.

        Returns:
            DatasetDict: Cleaned dataset.
        """
        for split in ["train", "validation", "test"]:
            print(type(dataset[split]))
            drop_list = [column for column in list(dataset[split].features.keys())
                          if column not in self.SELECTED_COLUMNS[dataset_name]]
            dataset[split] = dataset[split].remove_columns(drop_list)
            
            if dataset_name.startswith("assin"):
                dataset[split] = self.process_assin(dataset[split], task)

            dataset[split] = self.rename_dataset_columns(dataset[split], dataset_name)
            
        return dataset

    def load(self, 
             dataset_name: str, 
             language: str = "por", 
             variant: str = "full", 
             clean: bool = True, 
             task: Optional[str] = None, 
             hf_args: List = [], 
             hf_kwargs: Dict = {}) -> DatasetDict:
        """
        Load a dataset, optionally clean it, and return the processed dataset.

        Args:
            dataset_name (str): Name of the dataset to load.
            language (str, optional): Language of the dataset. Defaults to "por".
            variant (str, optional): Variant of the dataset. Defaults to "full". Relevant only if retrieving cleaned and translated ASSIN datasets.
            clean (bool, optional): Whether to clean the dataset, i.e. drop columns not relevant to the benchmark. Defaults to True.
            task (str, optional): Task type. Relevant only if retrieving cleaned ASSIN datasets.
            hf_args (list, optional): Positional arguments to pass to `datasets.load_dataset`. Defaults to an empty list.
            hf_kwargs (dict, optional): Keyword arguments to pass to `datasets.load_dataset`. Defaults to an empty dict.

        Returns:
            DatasetDict: Loaded (and optionally cleaned) dataset.
        """
        self.validate_parameters(dataset_name, language, variant)

        name = self.get_dataset_name(dataset_name, language)
        dataset = datasets.load_dataset(name, *hf_args, **hf_kwargs)
        dataset = self.apply_variant_filter(dataset, name, variant)

        if clean:
            dataset = self.clean_dataset(dataset, dataset_name, task)

        return dataset

def load_napolab_benchmark(include_translations=True):
    """ 
    Load the Napolab benchmark datasets, and optionally their translations.

    Args:
        include_translations (bool): Determines if translated versions of the datasets should be 
        loaded. Defaults to True.

    Returns:
        dict: A dictionary with two main keys:
            'datasets': A dictionary with dataset names as keys and loaded datasets as values.
            'translations': A dictionary with languages (e.g., 'english', 'spanish') as keys, 
            and a nested dictionary with dataset names as keys and loaded datasets as values.
    """
    loader = DatasetLoader()

    datasets = {}
    # This will load all datasets that make up the Napolab benchmark in the Portuguese language.
    for dataset_name in loader.DATASET_NAMES:
        if dataset_name in ["assin", "assin2"]:
            datasets[f"{dataset_name}-rte"] = loader.load(dataset_name, task="rte",
            hf_kwargs={"trust_remote_code": True})
            datasets[f"{dataset_name}-sts"] = loader.load(dataset_name, task="sts",
            hf_kwargs={"trust_remote_code": True})
        else:
            datasets[dataset_name] = loader.load(dataset_name, 
                hf_kwargs={"trust_remote_code": True})

    # It is also possible to load only the Brazilian Portuguese or European Portuguese portion of ASSIN instead of loading both portions as a single dataset:

    datasets["assin-rte-ptbr"] = loader.load("assin", task="rte", hf_args=["ptbr"])
    datasets["assin-rte-ptpt"] = loader.load("assin", task="rte", hf_args=["ptpt"])
    datasets["assin-sts-ptbr"] = loader.load("assin", task="sts", hf_args=["ptbr"])
    datasets["assin-sts-ptpt"] = loader.load("assin", task="sts", hf_args=["ptpt"])

    # Let's also load all translated datasets:

    translated_datasets = {}
    if include_translations:
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
                        translated_datasets[language]["assin-sts-ptbr"] = loader.load("assin", task="sts", variant="br")
                        translated_datasets[language]["assin-sts-ptpt"] = loader.load("assin", task="sts", variant="pt")            
                else:
                    translated_datasets[language][dataset_name] = loader.load(dataset_name, language=language)
    
    output = {
        "datasets": datasets,
        "translations": translated_datasets
    }

    return output

def convert_to_completion_format(df):
    """
    Converts a dataframe generated by export_napolab_benchmark to a format suitable for LLM Prompt Completion APIs.
    """

    completions_dataset = []

    porsimplessent_prompt = """You will be given two sentences, sentence1 and sentence2. Your task is to determine the complexity relationship between them. There are three possible labels:

    0: sentence1 is more simple than sentence2.
    1: sentence1 is identical to sentence2.
    2: sentence2 is more simple than sentence1.
    Provide only the label (either 0, 1, or 2) as the answer. No other text.

    """

    hatebr_prompt = """You will be given an Instagram comment (sentence1). Your task is to classify the comment as either offensive or non-offensive. The possible labels are:

    0: The comment is offensive.
    1: The comment is non-offensive.
    Provide only the label (either 0 or 1) as the answer. No other text.

    """

    relisa_prompt = """You will be given a sentence (sentence1) taken from a book review. Your task is to classify the sentiment expressed in the sentence. The possible labels are:

    0: The sentence expresses a positive sentiment.
    1: The sentence expresses a neutral sentiment.
    2: The sentence expresses a negative sentiment.
    3: The sentence expresses both positive and negative sentiments.
    
    Provide only the label (either 0, 1, 2 or 3) as the answer. No other text.

    """

    assin_sts_prompt = """You will be given two sentences, sentence1 and sentence2. Your task is to determine the semantic relatedness between these two sentences on a scale from 1 to 5. The guidelines for each label are:

    1: The sentences are completely different, on different subjects.
    2: The sentences are not related, but are roughly on the same subject.
    3: The sentences are somewhat related; they may describe different facts but share some details.
    4: The sentences are strongly related, but some details differ.
    5: The sentences mean essentially the same thing.
    Provide only the label (an integer between 1 and 5) as the answer. No other text.

    """

    assin_rte_prompt = """You will be given two sentences, sentence1 and sentence2. Your task is to classify the relationship between the two sentences. The possible labels are:

    0: No entailment. The sentences are not related in a way where one implies the other.
    1: Entailment. sentence1 (the text) entails sentence2 (the hypothesis). If sentence1 is true, then sentence2 must also be true.
    2: Paraphrase. The sentences are paraphrases of each other, meaning they both entail each other (bidirectional entailment).
    Provide only the label (0, 1, or 2) as the answer. No other text.

    """

    assin2_rte_prompt = """You will be given two sentences, sentence1 and sentence2. Your task is to classify the relationship between the two sentences. The possible labels are:

    0: No entailment. The sentences are not related in a way where one implies the other.
    1: Entailment. sentence1 (the text) entails sentence2 (the hypothesis). If sentence1 is true, then sentence2 must also be true.
    Provide only the label (0 or 1) as the answer. No other text.

    """

    rerelem_prompt = """
    
    """

    faquad_nli_prompt = """You will be given two sentences, sentence1 and sentence2. Your task is to determine if sentence2 is a suitable answer to sentence1. The possible labels are:

    0: Unsuitable. Sentence2 does not appropriately answer or address sentence1. 
    
    1: Suitable. Sentence2 appropriately answers or addresses sentence1.

    Provide only the label (0 or 1) as the answer. No other text.

    """

    prompt_templates = {'porsimplessent' : porsimplessent_prompt, 'hatebr': hatebr_prompt, 'reli-sa' : relisa_prompt, 'assin-sts': assin_sts_prompt, 'assin-rte-ptbr': assin_rte_prompt, 'assin-rte-ptpt': assin_rte_prompt, 'assin-sts-ptpt': assin_sts_prompt, 'rerelem': rerelem_prompt, 'assin-sts-ptbr' : assin_sts_prompt, 'assin2-rte': assin2_rte_prompt, 'faquad-nli' : faquad_nli_prompt, 'assin2-sts': assin_sts_prompt, 'assin-rte': assin_rte_prompt}

    records = df.to_dict("records")
    for idx, record in enumerate(records):
        
        record_language = "portuguese" if record["language"] not in ['english', 'galician', 'spanish', 'catalan'] else record["language"]
        
        record_label = record["label"]
        dataset_name = record["dataset_name"]

        if "assin" in dataset_name and "sts" in dataset_name:
            record_label = round(float(record_label))
            if record_label == 0:
                record_label = 1
            if record_label == 6:
                record_label = 5
        if "hatebr" in dataset_name:
            record_label = 0 if record_label == 'False' else 1

        if record_label == '0':
            record_label = 0
        elif record_label == '1':
            record_label = 1
        elif record_label == '2':
            record_label = 2
        
        relisa_labels = {"positive": 0, "neutral": 1, "negative": 2, "mixed": 3}

        if record_label in relisa_labels.keys():
            record_label = relisa_labels[record_label]

        base_prompt = prompt_templates[dataset_name]
        if dataset_name == "rerelem":
            continue
        if dataset_name in ['hatebr']:
            sentence_prompt = f"""
            Here is the comment:

            sentence1: {record["sentence1"]}

            The language of sentence1 is {record_language.capitalize()}.

            label:
            """            
        elif dataset_name in ['reli-sa', 'faquad-nli']:
            sentence_prompt = f"""
            Here is the sentence:

            sentence1: {record["sentence1"]}

            The language of sentence1 is {record_language.capitalize()}.

            label:
            """
        else:
            sentence_prompt = f"""
            Here are the sentences:

            sentence1: {record["sentence1"]}

            sentence2: {record["sentence2"]}

            The language of sentence1 and sentence2 is {record_language.capitalize()}.

            label:
            """
        final_prompt = base_prompt + sentence_prompt

        new_record = {
            "prompt": final_prompt,
            "system_prompt": base_prompt,
            "user_prompt": sentence_prompt,
            "answer": record_label,
            "dataset_name": dataset_name,
            "language": record_language,
            "sentence1": record["sentence1"],
            "sentence2": record["sentence2"],
            "original_split": record["split"],
        }

        completions_dataset.append(new_record)
    
    return pd.DataFrame(completions_dataset)


def export_napolab_benchmark(output_path=None, include_translations=True, single_file=True, include_train=False, save_single_file=False):
    """
    Load the Napolab benchmark datasets using load_napolab_benchmark and save each split of 
    each dataset as CSV in a structured hierarchy of folders and subfolders.
    
    Args:
        output_path (str): The path where datasets will be saved.
        include_translations (bool): Determines if translated versions of the datasets should be 
        saved. Defaults to True.
    """
    single_file_buffer = []    
    # Load the datasets
    data = load_napolab_benchmark(include_translations=include_translations)

    # Ensure top-level "datasets" and "translations" folders exist
    if not single_file:
        os.makedirs(os.path.join(output_path, "datasets"), exist_ok=True)
        if include_translations:
            os.makedirs(os.path.join(output_path, "translations"), exist_ok=True)

    # Save Portuguese datasets
    for dataset_name, dataset_obj in data["datasets"].items():
        for split, split_data in dataset_obj.items():
            if not include_train and split == "train":
                continue
            if not single_file:
                # Define the path for this dataset split
                split_path = os.path.join(output_path, "datasets", dataset_name, f"{split}.csv")
                os.makedirs(os.path.dirname(split_path), exist_ok=True)
            
            # Convert to pandas and save as CSV
            tmp_pandas = split_data.to_pandas()
            
            if not single_file:
                tmp_pandas.to_csv(split_path, index=False)
                logging.info(f"Saved {split_path}")
            else:
                tmp_pandas["dataset_name"] = dataset_name
                tmp_pandas["split"] = split
                tmp_pandas["language"] = ""
                tmp_pandas["translation"] = False
                single_file_buffer.append(tmp_pandas)

    # Save Translations
    if include_translations:
        for language, datasets in data["translations"].items():
            for dataset_name, dataset_obj in datasets.items():
                for split, split_data in dataset_obj.items():
                    if not include_train and split == "train":
                        continue
                    if not single_file:
                        # Define the path for this dataset split
                        split_path = os.path.join(output_path, "translations", language, dataset_name, f"{split}.csv")
                        os.makedirs(os.path.dirname(split_path), exist_ok=True)
                    
                    # Convert to pandas and save as CSV
                    tmp_pandas = split_data.to_pandas()
                    if not single_file:
                        tmp_pandas.to_csv(split_path, index=False)
                        logging.info(f"Saved {split_path}")
                    else:
                        tmp_pandas["dataset_name"] = dataset_name
                        tmp_pandas["split"] = split
                        tmp_pandas["language"] = language
                        tmp_pandas["translation"] = True
                        single_file_buffer.append(tmp_pandas)
    
    if single_file:
        df = pd.concat(single_file_buffer)
        if save_single_file:
            df.to_csv(output_path)
        return df