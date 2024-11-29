r"""Load the dataset according to the configuration."""

from typing import Tuple
from omegaconf import DictConfig

from .multi_pde.dataloader import static_multi_pde_dataset, dynamic_multi_pde_dataset
from .load_single_pde import single_pde_dataset


def load_dataset(config: DictConfig) -> Tuple:
    r"""
    Load the dataset according to the configuration.

    Args:
        config (DictConfig): Training configurations.

    Returns:
        dataloader_train (BatchDataset): Data loader for the training dataset.
        train_loader_dict (Dict[str, Dict[str, Tuple]]): A nested
            dictionary containing the data iterator instances of the training
            dataset for the evaluation, in which the random operations (data
            augmentation, spatio-temporal subsampling, etc) are disabled. Here,
            to be more specific, 'Tuple' actually refers to
            'Concatenate[TupleIterator, Dataset]'.
        test_loader_dict (Dict[str, Dict[str, Tuple]]): Similar to
            `train_loader_dict`, but for the testing dataset.
        data_info (Dict[str, Any]): A dictionary containing basic information
            about the current dataset.
    """
    if config.data.type == "single_pde":
        return single_pde_dataset(config)
    if config.model_type != "pdeformer":
        raise ValueError("multi_pde dataset only supports model_type==pdeformer")
    if config.data.type == "multi_pde":
        if "dynamic" in config.data and config.data.dynamic.enabled:
            return dynamic_multi_pde_dataset(config)
        return static_multi_pde_dataset(config)
    if config.data.type == "dynamic_multi_pde":
        return dynamic_multi_pde_dataset(config)
    raise NotImplementedError
