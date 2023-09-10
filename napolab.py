import datasets
from datasets import DatasetDict, Dataset
from typing import List, Dict, Union, Optional

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