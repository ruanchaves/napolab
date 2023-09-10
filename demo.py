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
                translated_datasets[language]["assin-sts-ptbr"] = loader.load("assin", task="sts", variant="br")
                translated_datasets[language]["assin-sts-ptpt"] = loader.load("assin", task="sts", variant="pt")            
        else:
            translated_datasets[language][dataset_name] = loader.load(dataset_name, language=language)

print(f"Loaded {len(datasets)} datasets in Portuguese and {len(translated_datasets)} translated datasets.")
print(f"Datasets in Portuguese: {datasets.keys()}")
for language in translated_datasets:
    print(f"Translated datasets in {language}: {translated_datasets[language].keys()}")