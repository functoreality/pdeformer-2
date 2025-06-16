r"""Optimizers."""
from typing import List
from omegaconf import DictConfig
from mindspore import nn
from ..cell.lora import lora_param_filter


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
    params = []
    module_list = config_train.get("module_list", ["all"])
    for module in module_list:
        module = module.lower()
        if module in ["all", "model"]:
            if len(module_list) > 1:
                raise ValueError(
                    "Length of 'module_list' should be exactly one when all "
                    "model parameters are updated in training.")
            params = model.trainable_params()
        elif module == "lora":
            if len(module_list) > 1:
                raise ValueError(
                    "Length of 'module_list' should be exactly one for LoRA.")
            params = list(filter(lora_param_filter, model.trainable_params()))
        elif module == "inr":
            params.extend(model.inr.trainable_params())
        elif module == "inr2":
            params.extend(model.inr2.trainable_params())
        elif module == "graphormer":
            params.extend(model.pde_encoder.graphormer.trainable_params())
        elif module == "function_encoder":
            params.extend(model.pde_encoder.function_encoder.trainable_params())
        elif module.startswith(("prefix=", "startswith=", "begin=", "start=")):
            _, prefix = module.split("=", 1)
            params.extend([param for param in model.trainable_params()
                           if param.name.startswith(prefix)])
        elif module.startswith(("contains=", "suffix=", "regex=")):
            raise NotImplementedError
        else:
            raise ValueError(
                f"'module_list' contains unexpected value {module}.")

    params = [{'params': params,
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
