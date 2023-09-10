import pytest
from napolab import DatasetLoader

loader = DatasetLoader()

def test_validate_parameters_invalid_dataset_name():
    with pytest.raises(ValueError):
        loader.validate_parameters("invalid_dataset_name", "portuguese", "full")

def test_validate_parameters_invalid_language():
    with pytest.raises(ValueError):
        loader.validate_parameters("assin", "invalid_language", "full")

def test_validate_parameters_invalid_variant():
    with pytest.raises(ValueError):
        loader.validate_parameters("assin", "portuguese", "invalid_variant")

def test_get_dataset_name_non_assin():
    assert loader.get_dataset_name("rerelem", "english") == "ruanchaves/rerelem_por_Latn_to_eng_Latn"

def test_get_dataset_name_assin():
    assert loader.get_dataset_name("assin", "portuguese") == "assin"

def test_get_dataset_name_assin_other_language():
    assert loader.get_dataset_name("assin", "english") == "ruanchaves/assin_por_Latn_to_eng_Latn"
