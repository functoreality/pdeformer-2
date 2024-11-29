r"""Optimizers."""
from typing import List
from omegaconf import DictConfig
from mindspore import nn


def get_optimizer(lr_var: List[float],
                  model: nn.Cell,
                  config_train: DictConfig) -> nn.Cell:
    r"""
    Get the optimizer according to the config.

    Args:
        lr_var (List[float]): A list of learning rate variables.
        model (nn.Cell): The model to be trained.
        config_train (DictConfig): The configuration of the training process.

    Returns:
        nn.Cell: The optimizer.
    """
    if config_train.get("inr2_only", False):
        params = [{'params': model.inr2.trainable_params(),
                   'lr': lr_var,
                   'weight_decay': config_train.weight_decay}]
    else:
        params = [{'params': model.trainable_params(),
                   'lr': lr_var,
                   'weight_decay': config_train.weight_decay}]
    if config_train.optimizer == 'Adam':
        optimizer = nn.Adam(params)
    elif config_train.optimizer == 'AdamW':
        optimizer = nn.AdamWeightDecay(params)
    else:
        raise NotImplementedError(
            "'optimizer' should be one of ['Adam', 'AdamW'], "
            f"but got {config_train.optimizer}")

    return optimizer
