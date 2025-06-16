r"""This module provides a wrapper for different models."""
from typing import Optional

from omegaconf import DictConfig
import mindspore as ms
from mindspore import nn, context
from mindspore import dtype as mstype

from .pdeformer import PDEformer
from .baseline import DeepONet, FNO, UNet2D, CNNDeepONet
from .lora import add_lora_into_net_
from ..utils.tools import calculate_num_params
from ..utils.record import Record


def get_model(config: DictConfig,
              record: Optional[Record] = None,
              compute_dtype=None) -> nn.Cell:
    r"""Get the model according to the config."""
    if compute_dtype is None:  # set automatically
        if context.get_context(attr_key='device_target') == "Ascend":
            compute_dtype = mstype.float16
        else:
            compute_dtype = mstype.float32

    # define network model
    if config.model_type == "pdeformer":
        model = PDEformer(config.model, compute_dtype=compute_dtype)
    elif config.model_type == "deeponet":
        model = DeepONet(config.deeponet.trunk_dim_in,
                         config.deeponet.trunk_dim_hidden,
                         config.deeponet.trunk_num_layers,
                         config.deeponet.branch_dim_in,
                         config.deeponet.branch_dim_hidden,
                         config.deeponet.branch_num_layers,
                         dim_out=config.deeponet.dim_out,
                         num_pos_enc=config.deeponet.num_pos_enc,
                         n_vars=config.deeponet.n_vars,
                         compute_dtype=compute_dtype)
    elif config.model_type == "cnn_deeponet":
        model = CNNDeepONet(config.deeponet.trunk_dim_in,
                            config.deeponet.trunk_dim_hidden,
                            config.deeponet.trunk_num_layers,
                            config.deeponet.branch_dim_in,
                            config.deeponet.branch_dim_hidden,
                            dim_out=config.deeponet.dim_out,
                            num_pos_enc=config.deeponet.num_pos_enc,
                            n_vars=config.deeponet.n_vars,
                            compute_dtype=compute_dtype)
    elif config.model_type == "fno3d":
        modes = config.fno3d.modes
        if isinstance(modes, int):  # int -> List[int]
            modes = [modes] * len(config.fno3d.resolution)
        else:  # ListConfig -> List[int]
            modes = list(modes)
        model = FNO(config.fno3d.in_channels,
                    config.fno3d.out_channels,
                    modes,
                    config.fno3d.resolution,
                    hidden_channels=config.fno3d.channels,
                    n_layers=config.fno3d.depths,
                    dft_compute_dtype=compute_dtype,
                    fno_compute_dtype=compute_dtype)
    elif config.model_type == "fno2d":
        modes = config.fno3d.modes
        if isinstance(modes, int):  # int -> List[int]
            modes = [modes] * 2
        else:  # ListConfig -> List[int]
            modes = list(modes)[1:]
        out_channels = config.fno3d.resolution[0] * config.fno3d.out_channels
        model = FNO(config.fno3d.in_channels,
                    out_channels,
                    modes,
                    config.fno3d.resolution[1:],
                    hidden_channels=config.fno3d.channels,
                    n_layers=config.fno3d.depths,
                    dft_compute_dtype=compute_dtype,
                    fno_compute_dtype=compute_dtype)
    elif config.model_type == "unet2d":
        out_channels = config.fno3d.resolution[0] * config.fno3d.out_channels
        model = UNet2D(config.fno3d.in_channels, out_channels, data_nhwc=True,
                       compute_dtype=compute_dtype)
    else:
        raise ValueError(f"The model_type {config.model_type} is not supported!")

    # LoRA before load_ckpt
    if "lora" in config.model:
        lora_mode = config.model.lora.mode.lower()
    else:
        lora_mode = "disabled"
    if lora_mode not in ["disabled", "create", "load", "merge"]:
        raise ValueError(f"Unexpected lora mode '{config.model.lora.mode}'.")
    if lora_mode in ["load", "merge"]:
        add_lora_into_net_(model, config.model.lora, compute_dtype)

    # load pre-trained model weights
    load_ckpt = config.model.get("load_ckpt", "none")
    if load_ckpt.lower() != "none":
        param_dict = ms.load_checkpoint(load_ckpt)
        param_not_load, checkpoint_not_load = ms.load_param_into_net(model, param_dict)
        if param_not_load or checkpoint_not_load:  # either list is non-empty
            warning_str = ("WARNING: These model parameters are not loaded: "
                           + str(param_not_load)
                           + "\nWARNING: These checkpoint parameters are not loaded: "
                           + str(checkpoint_not_load))
            if record is not None:
                record.print(warning_str)
            else:
                print(warning_str)

    # LoRA after load_ckpt
    if lora_mode == "create":
        add_lora_into_net_(model, config.model.lora, compute_dtype)
    elif lora_mode == "merge":
        raise NotImplementedError

    if record is not None:
        record.print(f"model_type: {config.model_type}, num_parameters: "
                     + calculate_num_params(model))
        # record.print("model architecture:\n" + str(model))

    return model
