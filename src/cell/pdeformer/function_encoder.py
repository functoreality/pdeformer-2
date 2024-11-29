r"""Function Encoder."""
from typing import Union, List
import math
from omegaconf import DictConfig
from mindspore import nn, Tensor, ops
from mindspore import dtype as mstype

from ..basic_block import MLP
from .inr_with_hypernet import Siren, MFNNet, PolyINR

class DeepSetFuncEncoder(nn.Cell):
    r"""
    Encoder for functions defined on one-dimensional domain.

    Args:
        dim_in (int): Dimension of the input features.
        dim_out (int): Dimension of the output features.
        dim_hidden (int): Dimension of the hidden features.
        num_layers (int): Number of layers.
        point_fn (str): Point function type. Options are "mlp" and "poly_inr". Default: "poly_inr".

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(batch\_size, num\_points, dim\_in)`.

    Outputs:
        Tensor of shape :math:`(batch\_size, num\_points, dim\_out)`.

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, nn
        >>> from src.cell.pdeformer.function_encoder import DeepSetFuncEncoder
        >>> x = Tensor(np.random.randn(2, 10, 3), mstype.float32)
        >>> encoder = DeepSetFuncEncoder(3, 64, 128, 2, point_fn="poly_inr")
        >>> out = encoder(x)
        >>> print(out.shape)
        (2, 10, 64)
    """

    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 dim_hidden: int,
                 num_layers: int = 3,
                 point_fn: str = "poly_inr",
                 compute_dtype=mstype.float16) -> None:
        super().__init__()
        if point_fn == "mlp":
            self.point_fn = MLP(dim_in, dim_hidden, dim_hidden,
                                num_layers, compute_dtype=compute_dtype)
        elif point_fn == "poly_inr":
            self.point_fn = PolyINR(dim_in, dim_hidden, dim_hidden,
                                    num_layers, compute_dtype=compute_dtype)
        elif point_fn == "mfn":
            self.point_fn = MFNNet(dim_in, dim_hidden, dim_hidden,
                                   num_layers, compute_dtype=compute_dtype)
        elif point_fn == "siren":
            self.point_fn = Siren(dim_in, dim_hidden, dim_hidden,
                                  num_layers, compute_dtype=compute_dtype)
        else:
            raise NotImplementedError(
                f"Point function '{point_fn}' not implemented!")
        self.post_fn = MLP(dim_hidden, dim_out, dim_hidden, num_layers=num_layers,
                           compute_dtype=compute_dtype)

    def construct(self, x: Tensor) -> Tensor:
        r"""construct"""
        out = self.point_fn(x)  # [..., num_points, dim_hidden]
        out = ops.mean(out, axis=-2)  # [..., dim_hidden]
        out = self.post_fn(out)  # [..., dim_out]
        return out


class WeightedDeepSetFuncEncoder(nn.Cell):
    r"""Encoder for functions defined on one-dimensional domain.

    Args:
        dim_in (int): Dimension of the input features.
        dim_out (int): Dimension of the output features.
        dim_hidden (int): Dimension of the hidden features.
        num_layers (int): Number of layers. Default: 6.
        point_fn (str): Point function type. Options are "mlp", "poly_inr",
            "poly_inr_shared", and "siren". Default: "poly_inr".
        compute_dtype (mstype.Float): The computation type of the layer.
            Default: ``mstype.float16``.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(..., num\_points, dim\_in)`.

    Outputs:
        Tensor of shape :math:`(..., dim\_out)`.

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, nn
        >>> from src.cell.pdeformer.function_encoder import WeightedDeepSetFuncEncoder
        >>> x = Tensor(np.random.randn(2, 10, 3), mstype.float32)
        >>> encoder = WeightedDeepSetFuncEncoder(3, 64, 128, 5, point_fn="poly_inr")
        >>> out = encoder(x)
        >>> print(out.shape)
        (2, 64)
    """

    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 dim_hidden: int,
                 num_layers: int = 3,
                 point_fn: str = "poly_inr",
                 compute_dtype=mstype.float16) -> None:
        super().__init__()
        if point_fn == "mlp":
            self.point_fn = MLP(dim_in, dim_hidden, dim_hidden, num_layers,
                                compute_dtype=compute_dtype)
            self.weight_fn = MLP(dim_in, dim_hidden, dim_hidden, num_layers,
                                 compute_dtype=compute_dtype)
        elif point_fn == "poly_inr":
            self.point_fn = PolyINR(dim_in, dim_hidden, dim_hidden,
                                    num_layers, compute_dtype=compute_dtype)
            self.weight_fn = PolyINR(dim_in, dim_hidden, dim_hidden,
                                     num_layers, compute_dtype=compute_dtype)
        elif point_fn == "mfn":
            self.point_fn = MFNNet(dim_in, dim_hidden, dim_hidden,
                                   num_layers, compute_dtype=compute_dtype)
            self.weight_fn = MFNNet(dim_in, dim_hidden, dim_hidden,
                                    num_layers, compute_dtype=compute_dtype)
        elif point_fn == "siren":
            self.point_fn = Siren(dim_in, dim_hidden, dim_hidden,
                                  num_layers, compute_dtype=compute_dtype)
            self.weight_fn = Siren(dim_in, dim_hidden, dim_hidden,
                                   num_layers, compute_dtype=compute_dtype)
        else:
            raise NotImplementedError(
                f"Point function '{point_fn}' not implemented!")
        self.post_fn = MLP(dim_hidden, dim_out, dim_hidden,
                           num_layers, compute_dtype=compute_dtype)
        self.cast = ops.Cast()
        self.compute_dtype = compute_dtype

    def construct(self, x: Tensor) -> Tensor:
        r"""construct"""
        feature = self.point_fn(x)  # [..., num_points, dim_hidden]
        weight = self.weight_fn(x)  # [..., num_points, dim_hidden]
        weight = self.cast(weight, mstype.float32)
        probs = ops.softmax(weight, axis=-2)
        probs = self.cast(probs, self.compute_dtype)
        feature_probs = feature * probs
        # [..., num_points, dim_hidden] -> [..., dim_hidden]
        out = ops.sum(feature_probs, dim=-2)
        out = self.post_fn(out)  # [..., dim_out]
        return out


class PerBranchFuncEncoder(nn.Cell):
    r"""Encoder for functions defined on one-dimensional domain."""

    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 dim_hidden: int,
                 num_layers: int = 3,
                 point_fn: Union[str, List[str]] = "poly_inr",
                 num_branches: int = 4,
                 func_enc_type: str = "weighted_deepset",
                 compute_dtype=mstype.float16) -> None:
        super().__init__()
        if func_enc_type == "deepset":
            func_enc_cls = DeepSetFuncEncoder
        elif func_enc_type == "weighted_deepset":
            func_enc_cls = WeightedDeepSetFuncEncoder
        else:
            raise ValueError(f"Unknown func_enc_type {func_enc_type}!")

        if isinstance(point_fn, str):
            point_fn = [point_fn.lower()] * num_branches
        elif len(point_fn) != num_branches:
            raise ValueError("When 'deepset_point_fn' is a list, the length "
                             f"({len(point_fn)}) should be equal to "
                             f"'num_branches' ({num_branches}).")
        self.function_encoder_list = nn.CellList([func_enc_cls(
            dim_in, dim_out, dim_hidden // num_branches, num_layers,
            point_fn[i_branch], compute_dtype=compute_dtype,
            ) for i_branch in range(num_branches)])

    def construct(self, x: Tensor) -> Tensor:
        '''
        Args:
            x (Tensor): shape is [..., num_points, dim_in]
        '''
        out = [func_enc(x) for func_enc in self.function_encoder_list]
        out = ops.stack(out, axis=-2)  # [..., num_branches, dim_out]
        return out


class Patched1DFuncEncoder(nn.Cell):
    r"""Encoder for functions defined on one-dimensional domain.

    Args:
        dim_in (int): Dimension of the input features.
        dim_out (int): Dimension of the output features.
        dim_hidden (int): Dimension of the hidden features.
        num_layers (int): Number of layers. Default: 3.
        patch_len (int): Length of the patch. Default: 16.
        compute_dtype (mstype.Float): The computation type of the layer.
            Default: ``mstype.float16``.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(batch\_size, num\_points, dim\_in)`.

    Outputs:
        Tensor of shape :math:`(batch\_size * num\_points / patch\_len, dim\_out)`.

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, nn
        >>> from src.cell.pdeformer.function_encoder import PatchedFuncEncoder
        >>> dim_in, dim_out, dim_hidden, num_layers, patch_len = 3, 256, 256, 5, 4
        >>> num_points = 128
        >>> x = Tensor(np.random.randn(2, num_points, dim_in), mstype.float32)
        >>> encoder = PatchedFuncEncoder(dim_in, dim_out, dim_hidden, num_layers,
        >>>                              patch_len, compute_dtype=mstype.float32)
        >>> out = encoder(x)
        >>> print(out.shape)
        (64, 256)
    """

    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 dim_hidden: int,
                 num_layers: int = 3,
                 patch_len: int = 16,
                 compute_dtype=mstype.float16) -> None:
        super().__init__()
        self.input_dim = patch_len * dim_in
        self.mlp = MLP(self.input_dim, dim_out, dim_hidden,
                       num_layers, compute_dtype=compute_dtype)

    def construct(self, x: Tensor) -> Tensor:
        '''
        Args:
            x (Tensor): shape is [bsz, num_points, dim_in]
        '''
        out = x.reshape(-1, self.input_dim)
        out = self.mlp(out)  # [bsz*num_patch, dim_out]
        return out


class Patched2DFuncEncoder(nn.Cell):
    r"""Encoder for functions defined on two-dimensional uniform grids."""

    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 dim_hidden: int,
                 num_layers: int = 3,
                 resolution: int = 128,
                 num_patches: int = 16,
                 compute_dtype=mstype.float16) -> None:
        super().__init__()
        self.n_patch_axis = int(math.sqrt(num_patches))
        if self.n_patch_axis**2 != num_patches:
            raise ValueError("For 'patched2d', 'num_branches' should be a "
                             f"perfect square number, but got {num_patches}.")
        self.patch_len, residual = divmod(resolution, self.n_patch_axis)
        if residual > 0:
            raise ValueError(f"num_branches ({num_patches}) should be a factor "
                             f"of the square of 'resolution' ({resolution}).")
        mlp_dim_in = self.patch_len**2 * dim_in
        self.mlp = MLP(mlp_dim_in, dim_out, dim_hidden,
                       num_layers, compute_dtype=compute_dtype)

    def construct(self, x: Tensor) -> Tensor:
        r"""construct"""
        batch_size, _, dim_in = x.shape
        n_patch_axis = self.n_patch_axis
        patch_len = self.patch_len
        # [bsz, npa_x*plen_x*npa_y*plen_y, dim_in] -> [bsz*npa_x, plen_x, npa_y, plen_y*dim_in]
        x = x.reshape(batch_size * n_patch_axis, patch_len,
                      n_patch_axis, patch_len * dim_in)
        # [bsz*npa_x, npa_y, plen_x, plen_y*dim_in]
        x = x.transpose((0, 2, 1, 3))
        # Shape is [bsz*npa_x*npa_y, plen_x*plen_y*dim_in].
        x = x.reshape(batch_size * n_patch_axis**2, patch_len**2 * dim_in)
        x = self.mlp(x)  # [bsz*npa_x*npa_y, dim_out]
        return x


class PatchSet2DFuncEncoder(nn.Cell):
    r"""Encoder for functions defined on two-dimensional uniform grids."""

    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 dim_hidden: int,
                 num_layers: int = 3,
                 resolution: int = 128,
                 num_patches: int = 16,
                 compute_dtype=mstype.float16) -> None:
        super().__init__()
        self.n_patch_axis = int(math.sqrt(num_patches))
        if self.n_patch_axis**2 != num_patches:
            raise ValueError("For 'patched2d', 'num_branches' should be a "
                             f"perfect square number, but got {num_patches}.")
        self.patch_len, residual = divmod(resolution, self.n_patch_axis)
        if residual > 0:
            raise ValueError(f"num_branches ({num_patches}) should be a factor "
                             f"of the square of 'resolution' ({resolution}).")
        mlp_dim_in = self.patch_len**2 * dim_in
        self.mlp = MLP(mlp_dim_in, dim_out, dim_hidden,
                       num_layers, compute_dtype=compute_dtype)

    def construct(self, x: Tensor) -> Tensor:
        r"""construct"""
        batch_size, _, dim_in = x.shape
        n_patch_axis = self.n_patch_axis
        patch_len = self.patch_len
        # [bsz, npa_x*plen_x*npa_y*plen_y, dim_in] -> [bsz*npa_x, plen_x, npa_y, plen_y*dim_in]
        x = x.reshape(batch_size * n_patch_axis, patch_len,
                      n_patch_axis, patch_len * dim_in)
        # [bsz*npa_x, npa_y, plen_x, plen_y*dim_in]
        x = x.transpose((0, 2, 1, 3))
        # Shape is [bsz, npa_x*npa_y, plen_x*plen_y*dim_in].
        x = x.reshape(batch_size, n_patch_axis**2, patch_len**2 * dim_in)
        x = self.mlp(x)  # [bsz, npa_x*npa_y, dim_out]
        x = ops.mean(x, axis=-2)  # [bsz, dim_out]
        return x


class Patched2DConvFuncEncoder(nn.Cell):
    r"""
    Encoder for functions defined on two-dimensional uniform grids,
    implemented by convolution 2D + MLP.
    Args:
        dim_in (int): Dimension of the input features.
        dim_out (int): Dimension of the output features.
        dim_hidden (int): Dimension of the hidden features.
        num_layers (int): Number of layers of the MLP. Default: 3.
        resolution (int): Resolution of the input tensor. Default 128.
        num_patches (int): Number of patches. Default: 16.
        compute_dtype (mstype.Float): The computation type of the layer.
        Default: ``mstype.float16``.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(batch\_size,
        resolution, resolution, dim\_in)`.

    Outputs:
        Tensor of shape :math:`(batch\_size * num\_patches, dim\_out)`.

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, nn
        >>> from src.cell.pdeformer.function_encoder import Patched2D_ConvFuncEncoder
        >>> from mindspore import dtype as mstype
        >>> dim_in, dim_out, dim_hidden, num_layers = 3, 256, 256, 5
        >>> resolution, num_patches = 128, 4
        >>> x = Tensor(np.random.randn(2, resolution, resolution, dim_in), mstype.float32)
        >>> encoder = Patched2D_ConvFuncEncoder(dim_in, dim_out, dim_hidden,
        >>> num_layers, resolution, num_patches)
        >>> out = encoder(x)
        >>> print(out.shape)
        (64, 256)
    """

    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 dim_hidden: int,
                 num_layers: int = 3,
                 resolution: int = 128,
                 num_patches: int = 16,
                 compute_dtype=mstype.float16) -> None:
        super().__init__()
        self.n_patch_axis = int(math.sqrt(num_patches))
        if self.n_patch_axis**2 != num_patches:
            raise ValueError("For 'patched2d', 'num_branches' should be a "
                             f"perfect square number, but got {num_patches}.")
        self.patch_len, residual = divmod(resolution, self.n_patch_axis)
        if residual > 0:
            raise ValueError(f"num_branches ({num_patches}) should be a factor "
                             f"of the square of 'resolution' ({resolution}).")
        self.num_patches = num_patches
        self.dim_hidden = dim_hidden
        self.resolution = resolution
        self.patch_layer = nn.Conv2d(dim_in, dim_hidden,
                                     self.patch_len, self.patch_len).to_float(compute_dtype)
        self.mlp = MLP(dim_hidden, dim_out, dim_hidden, num_layers,
                       compute_dtype=compute_dtype)

    def construct(self, x: Tensor) -> Tensor:
        r"""construct"""
        batch_size, _, dim_in = x.shape
        # [bsz, num_points, dim_in] ->
        # [bsz, resolution, resolution, dim_in]
        x = x.reshape(batch_size, self.resolution, self.resolution, dim_in)
        # [bsz, resolution, resolution, dim_in] ->
        # [bsz, dim_in, resolution, resolution]
        x = x.permute(0, 3, 1, 2)
        # [bsz, dim_in, resolution, resolution] ->
        # [bsz, dim_hidden, n_patch_x, n_patch_y]
        x = self.patch_layer(x)
        # [bsz, dim_hidden, n_patch_x, n_patch_y] ->
        # [bsz,n_patch_x, n_patch_y, dim_hidden]
        x = x.permute(0, 3, 1, 2)
        # [bsz,n_patch_x, n_patch_y, dim_hidden] ->
        # [bsz * num_patches, dim_hidden]
        x = x.reshape(batch_size * self.num_patches, self.dim_hidden)
        x = self.mlp(x)
        # [bsz * num_patches, dim_hidden]
        return x


class Conv2dFuncEncoder(nn.Cell):
    r"""CNN Encoder for functions defined on two-dimensional uniform grids."""

    def __init__(self,
                 in_channels: int = 1,
                 out_dim: int = 256,
                 resolution: int = 128,
                 compute_dtype=mstype.float16):
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution

        layers = []
        layers.append(nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=2, has_bias=True, weight_init='HeUniform',
            bias_init="zeros").to_float(compute_dtype))  # [bsz, 64, H/2, W/2]
        layers.append(nn.GELU())
        layers.append(nn.Conv2d(
            64, 128, kernel_size=3, stride=2, has_bias=True, weight_init='HeUniform',
            bias_init="zeros").to_float(compute_dtype))  # [bsz, 128, H/4, W/4]
        layers.append(nn.GELU())
        layers.append(nn.Conv2d(
            128, 256, kernel_size=3, stride=2, has_bias=True, weight_init='HeUniform',
            bias_init="zeros").to_float(compute_dtype))  # [bsz, 256, H/8, W/8]
        layers.append(nn.GELU())
        layers.append(nn.Conv2d(
            256, 512, kernel_size=3, stride=2, has_bias=True, weight_init='HeUniform',
            bias_init="zeros").to_float(compute_dtype))  # [bsz, 512, H/16, W/16]
        layers.append(nn.GELU())
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))  # [bsz, 512, 1, 1]
        layers.append(nn.Flatten())  # [bsz, 512]
        layers.append(nn.Dense(
            512, 512, has_bias=True, weight_init='HeUniform',
            bias_init="zeros").to_float(compute_dtype))  # [bsz, 512]
        layers.append(nn.GELU())
        layers.append(nn.Dense(
            512, out_dim, has_bias=True, weight_init='HeUniform',
            bias_init="zeros").to_float(compute_dtype))  # [bsz, out_dim]
        self.net = nn.SequentialCell(layers)

    def construct(self, x: Tensor) -> Tensor:
        '''
        Args:
            x (Tensor): shape is [bsz, num_points_ic, dim_in], dim_in=5 or 6.
        '''
        bsz, _, dim_in = x.shape
        x = x.reshape(bsz, self.resolution, self.resolution, dim_in)
        x = x[:, :, :, -self.in_channels:]
        x = x.transpose((0, 3, 1, 2))  # NHWC -> NCHW
        return self.net(x)


class Conv2dFuncEncoderV2(nn.Cell):
    r"""Smaller CNN Encoder for functions defined on two-dimensional uniform grids."""

    def __init__(self,
                 in_channels: int = 1,
                 out_dim: int = 256,
                 resolution: int = 128,
                 cnn_keep_nchw: bool = True,
                 compute_dtype=mstype.float16):
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.cnn_keep_nchw = cnn_keep_nchw

        get_activation_fn = nn.GELU  # nn.ReLU
        conv_kwargs = dict(kernel_size=3, stride=2, has_bias=True,
                           weight_init='HeUniform', bias_init="zeros")
        self.net = nn.SequentialCell([
            nn.Conv2d(in_channels, 16, **conv_kwargs).to_float(compute_dtype),
            get_activation_fn(),  # [bsz, 16, H/2, W/2]
            nn.Conv2d(16, 32, **conv_kwargs).to_float(compute_dtype),
            get_activation_fn(),  # [bsz, 32, H/4, W/4]
            nn.Conv2d(32, 64, **conv_kwargs).to_float(compute_dtype),
            get_activation_fn(),  # [bsz, 64, H/8, W/8]
            nn.Conv2d(64, 128, **conv_kwargs).to_float(compute_dtype),
            get_activation_fn(),  # [bsz, 128, H/16, W/16]
            nn.Conv2d(128, 256, **conv_kwargs).to_float(compute_dtype),
            get_activation_fn(),  # [bsz, 256, H/32, W/32]
            nn.Conv2d(256, out_dim, **conv_kwargs).to_float(compute_dtype),
        ])  # Output shape: [bsz, out_dim, H/64, W/64].

    def construct(self, x: Tensor) -> Tensor:
        '''
        Args:
            x (Tensor): shape is [bsz, num_points_ic, dim_in], dim_in=5 or 6.
        '''
        bsz, _, _ = x.shape
        x = x[:, :, -self.in_channels:]  # required when input_txyz == False
        x = x.reshape(bsz, self.resolution, self.resolution, self.in_channels)
        x = x.transpose((0, 3, 1, 2))  # NHWC -> NCHW
        x = self.net(x)  # [bsz, out_dim, 2, 2].
        if not self.cnn_keep_nchw:  # for backward compatibility of ckpt
            x = x.transpose((0, 2, 3, 1))  # NCHW -> NHWC
        return x  # [bsz, 2, 2, out_dim].


class Conv2dFuncEncoderV3(nn.Cell):
    r"""Smaller CNN Encoder for functions defined on two-dimensional uniform grids."""

    def __init__(self,
                 in_channels: int = 1,
                 out_dim: int = 256,
                 resolution: int = 128,
                 cnn_keep_nchw: bool = True,
                 compute_dtype=mstype.float16):
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.cnn_keep_nchw = cnn_keep_nchw

        get_activation_fn = nn.ReLU
        conv_kwargs = dict(kernel_size=4, stride=4, has_bias=True,
                           weight_init='HeUniform', bias_init="zeros")
        self.net = nn.SequentialCell([
            nn.Conv2d(in_channels, 32, **conv_kwargs).to_float(compute_dtype),
            get_activation_fn(),  # [bsz, 32, H/4, W/4]
            nn.Conv2d(32, 128, **conv_kwargs).to_float(compute_dtype),
            get_activation_fn(),  # [bsz, 128, H/16, W/16]
            nn.Conv2d(128, out_dim, **conv_kwargs).to_float(compute_dtype),
        ])  # Output shape: [bsz, out_dim, H/64, W/64].

    construct = Conv2dFuncEncoderV2.construct


def get_function_encoder(config_fenc: DictConfig,
                         dim_in: int,
                         dim_out: int,
                         compute_dtype=mstype.float16) -> nn.Cell:
    r"""Get the function encoder network."""
    function_encoder_type = config_fenc.type.lower()
    if function_encoder_type == "deepset":
        function_encoder = DeepSetFuncEncoder(
            dim_in,
            dim_out * config_fenc.num_branches,
            config_fenc.dim_hidden,
            config_fenc.num_layers,
            config_fenc.deepset_point_fn.lower(),
            compute_dtype=compute_dtype)
    elif function_encoder_type == "weighted_deepset":
        function_encoder = WeightedDeepSetFuncEncoder(
            dim_in,
            dim_out * config_fenc.num_branches,
            config_fenc.dim_hidden,
            config_fenc.num_layers,
            config_fenc.deepset_point_fn.lower(),
            compute_dtype=compute_dtype)
    elif function_encoder_type == "pb_wdpset":
        function_encoder = PerBranchFuncEncoder(
            dim_in,
            dim_out,
            config_fenc.dim_hidden,
            config_fenc.num_layers,
            config_fenc.deepset_point_fn,
            config_fenc.num_branches,
            compute_dtype=compute_dtype)
    elif function_encoder_type == "patched1d":
        resolution = config_fenc.resolution
        patch_len, residual = divmod(resolution**2, config_fenc.num_branches)
        if residual > 0:
            raise ValueError(
                f"num_branches ({config_fenc.num_branches}) "
                f"should be a factor of the square of 'resolution' ({resolution})!")
        function_encoder = Patched1DFuncEncoder(
            dim_in,
            dim_out,
            config_fenc.dim_hidden,
            config_fenc.num_layers,
            patch_len,
            compute_dtype=compute_dtype)
    elif function_encoder_type == "patched2d":
        function_encoder = Patched2DFuncEncoder(
            dim_in,
            dim_out,
            config_fenc.dim_hidden,
            config_fenc.num_layers,
            config_fenc.resolution,
            config_fenc.num_branches,
            compute_dtype=compute_dtype)
    elif function_encoder_type == "patchset2d":
        function_encoder = PatchSet2DFuncEncoder(
            dim_in,
            dim_out * config_fenc.num_branches,
            config_fenc.dim_hidden,
            config_fenc.num_layers,
            config_fenc.resolution,
            compute_dtype=compute_dtype)
    elif function_encoder_type == 'patched2dconv':
        function_encoder = Patched2DConvFuncEncoder(
            dim_in,
            dim_out,
            config_fenc.dim_hidden,
            config_fenc.num_layers,
            config_fenc.resolution,
            config_fenc.num_branches,
            compute_dtype=compute_dtype
        )
    elif function_encoder_type == "conv2d":
        resolution = config_fenc.get("resolution", 128)
        if config_fenc.get("conv2d_input_txyz", False):
            in_channels = dim_in
        else:
            in_channels = 1
        function_encoder = Conv2dFuncEncoder(
            in_channels,
            dim_out * config_fenc.num_branches,
            resolution,
            compute_dtype=compute_dtype)
    elif function_encoder_type in "cnn2dv2 cnn2dv3".split():
        resolution = config_fenc.get("resolution", 128)
        cnn_keep_nchw = config_fenc.get("cnn_keep_nchw", True)
        if config_fenc.get("conv2d_input_txyz", False):
            in_channels = dim_in
        else:
            in_channels = 1
        if config_fenc.num_branches != 4:
            raise NotImplementedError
        if function_encoder_type == "cnn2dv2":
            function_encoder = Conv2dFuncEncoderV2(
                in_channels, dim_out, resolution, cnn_keep_nchw,
                compute_dtype=compute_dtype)
        else:
            function_encoder = Conv2dFuncEncoderV3(
                in_channels, dim_out, resolution, cnn_keep_nchw,
                compute_dtype=compute_dtype)
    else:
        raise NotImplementedError(
            "'function_encoder_type' should be in ['deepset', 'weighted_deepset', 'patched'], "
            f"but got '{config_fenc.type}'.")

    return function_encoder
