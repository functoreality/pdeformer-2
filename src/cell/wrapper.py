r"""This module provides a wrapper for different models."""
from typing import Optional

from omegaconf import DictConfig
import mindspore as ms
from mindspore import nn, context
from mindspore import dtype as mstype

from .pdeformer import PDEformer
from .baseline import DeepONet, FNO
from ..utils.tools import calculate_num_params
from ..utils.record import Record


def get_model(config: DictConfig,
              record: Optional[Record] = None,
              compute_type=None) -> nn.Cell:
    r"""Get the model according to the config."""
    if compute_type is None:  # set automatically
        if context.get_context(attr_key='device_target') == "Ascend":
            compute_type = mstype.float16
        else:
            compute_type = mstype.float32

    if config.model_type == "pdeformer":
        model = PDEformer(config.model, compute_dtype=compute_type)
    elif config.model_type == "deeponet":
        model = DeepONet(config.deeponet.trunk_dim_in,
                         config.deeponet.trunk_dim_hidden,
                         config.deeponet.trunk_num_layers,
                         config.deeponet.branch_dim_in,
                         config.deeponet.branch_dim_hidden,
                         config.deeponet.branch_num_layers,
                         dim_out=config.deeponet.dim_out,
                         num_pos_enc=config.deeponet.num_pos_enc,
                         compute_dtype=compute_type)
    elif config.model_type == "fno3d":
        model = FNO(config.fno3d.in_channels, config.fno3d.out_channels,
                    [config.fno3d.modes for _ in range(3)],
                    config.fno3d.resolution,
                    dft_compute_dtype=compute_type,
                    fno_compute_dtype=compute_type)
    else:
        raise ValueError(f"The model_type {config.model_type} is not supported!")

    # load pre-trained model weights
    load_ckpt = config.model.get("load_ckpt", "none")
    if load_ckpt.lower() != "none":
        param_dict = ms.load_checkpoint(load_ckpt)
        param_not_load, checkpoint_not_load = ms.load_param_into_net(model, param_dict)
        if param_not_load or checkpoint_not_load:  # either list is non-empty
            warning_str = ("WARNING: These model parameters are not loaded: "
                           + str(param_not_load)
                           + '\nWARNING: These checkpoint parameters are not loaded: '
                           + str(checkpoint_not_load))
            if record is not None:
                record.print(warning_str)
            else:
                print(warning_str)

    if record is not None:
        record.print(f"model_type: {config.model_type}, num_parameters: "
                     + calculate_num_params(model))
        # record.print("model architecture:\n" + str(model))

    return model
