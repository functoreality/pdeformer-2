#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Low-Rank Adapter algrithm for pretrained model.
Reference: https://arxiv.org/abs/2106.09685
"""
import re
import math
from omegaconf import DictConfig
from mindspore import nn, ops, Tensor, Parameter
from mindspore import dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common.initializer import initializer, HeUniform


class LoRADense(nn.Dense):
    r"""
    Define a dense layer with LoRA structure.
    Adapted from `mindpet.delta.lora.LoRADense`.

    Attributes:
        lora_in_channels (int): The number of channels in the input space.
        lora_out_channels (int): The number of channels in the output space.
        lora_rank(int): The number of rows(columns) in LoRA matrices.
        lora_alpha(float): A constant in lora_rank.
        param_init_type(:class:`mindspore.dtype`): The type of data in initialized tensor.
        compute_dtype(:class:`mindspore.dtype`): The compute type of data.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 lora_rank: int = 8,
                 lora_alpha: float = 16.0,
                 lora_a_init=HeUniform(negative_slope=math.sqrt(5)),
                 lora_b_init='zeros',
                 param_init_type=mstype.float32,
                 compute_dtype=mstype.float16,
                 **kwargs) -> None:
        super().__init__(in_channels, out_channels, **kwargs)

        # Define and initialize params
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.mindpet_delta_lora_a = Parameter(
            initializer(lora_a_init, [lora_rank, in_channels], param_init_type),
            name="mindpet_delta_lora_a")
        self.mindpet_delta_lora_b = Parameter(
            initializer(lora_b_init, [out_channels, lora_rank], param_init_type),
            name="mindpet_delta_lora_b")
        self.scaling = self.lora_alpha / self.lora_rank

        # Calculation utils
        self.mul = ops.Mul()
        self.add = ops.Add()
        self.cast = ops.Cast()
        self.dtype = compute_dtype
        self.lora_a_matmul = P.MatMul(transpose_b=True)
        self.lora_b_matmul = P.MatMul(transpose_b=True)

    def construct(self, input_tensor: Tensor) -> Tensor:
        r"""forward"""
        # Data type operation
        ori_dtype = F.dtype(input_tensor)
        input_tensor = self.cast(input_tensor, self.dtype)
        weight = self.cast(self.weight, self.dtype)
        lora_a = self.cast(self.mindpet_delta_lora_a, self.dtype)
        lora_b = self.cast(self.mindpet_delta_lora_b, self.dtype)
        scaling = self.cast(self.scaling, self.dtype)

        # Shape operations
        x_shape = self.shape_op(input_tensor)
        input_tensor = self.reshape(input_tensor, (-1, x_shape[-1]))

        # Dense result
        dense_result = self.matmul(input_tensor, weight)
        if self.has_bias:
            bias = self.cast(self.bias, self.dtype)
            dense_result = self.bias_add(dense_result, bias)

        # LoRA result
        input_tensor = self.lora_a_matmul(input_tensor, lora_a)
        input_tensor = self.lora_b_matmul(input_tensor, lora_b)
        input_tensor = self.mul(input_tensor, scaling)

        # Result addition and activation
        dense_result = self.add(dense_result, input_tensor)
        if self.activation_flag:
            dense_result = self.activation(dense_result)

        # Shape restore
        if len(x_shape) != 2:
            out_shape = x_shape[:-1] + (-1,)
            dense_result = self.reshape(dense_result, out_shape)
        dense_result = self.cast(dense_result, ori_dtype)

        return dense_result


def add_lora_into_net_(net: nn.Cell,
                       config_lora: DictConfig,
                       compute_dtype=mstype.float16) -> None:
    r"""
    Add LoRA into pre-trained network module.
    Adapted from `mindformers.pet.utils.recursive_replace_dense_cell`.
    """
    for name, cell in net._cells.items():  # pylint: disable=W0212
        if (not cell) or re.match(config_lora.exclude_layers, name):
            continue
        if not (isinstance(cell, nn.Dense)
                and re.match(config_lora.target_modules, name)):
            add_lora_into_net_(cell, config_lora, compute_dtype=compute_dtype)
            continue
        dest_cell = LoRADense(in_channels=cell.in_channels,
                              out_channels=cell.out_channels,
                              lora_rank=config_lora.lora_rank,
                              lora_alpha=config_lora.lora_alpha,
                              param_init_type=cell.weight.dtype,
                              compute_dtype=compute_dtype,
                              has_bias=cell.has_bias,
                              activation=cell.activation)

        # load weight of original layers.
        dest_cell.matmul = cell.matmul
        dest_cell.weight = cell.weight
        if cell.has_bias:
            dest_cell.bias = cell.bias
            dest_cell.bias_add = cell.bias_add

        net._cells[name] = dest_cell  # pylint: disable=W0212
    net.update_parameters_name()  # 'name' -> 'prefix.name' for LoRA params


def lora_param_filter(param: Parameter) -> bool:
    r"""Filter LoRA parameters."""
    return param.name[:-1].endswith(".mindpet_delta_lora_")
