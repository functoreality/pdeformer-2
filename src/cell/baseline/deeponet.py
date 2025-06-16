r"""DeepONet model."""
from mindspore import nn, Tensor, Parameter, ops
import mindspore.common.dtype as mstype

from ..basic_block import MLP, CoordPositionalEncoding


class DeepONet(nn.Cell):
    r"""
    The DeepONet model.
    DeepONet model is composed of branch net and trunk net, both of which are MLPs.
    The number of output neurons in branch net and trunk Net is the same, and
    the inner product of their outputs is the output of DeepONet.
    The details can be found in `Lu L, Jin P, Pang G, et al. Learning nonlinear
    operators via DeepONet based on the universal approximation theorem of operators[J].
    Nature machine intelligence, 2021, 3(3): 218-229.
    <https://www.nature.com/articles/s42256-021-00302-5>`.

    For the multiple-output case with `n_vars` outputs, we split the outputs of
    the trunk net output into `n_vars` groups, each with `dim_out` neurons. The
    k-th output value is given by taking the inner product of the k-th group
    trunk net output and the entire branch net output. Please refer to approach
    4 of section 3.1.6 of the following paper:
    Lu L, Xuhui Meng, et al. A comprehensive and fair comparison of two neural
    operators (with practical extensions) based on FAIR data[J].
    Computer methods in applied mechanics and engineering, 2022-03, Vol.393 (C).

    Args:
        trunk_dim_in (int): number of input neurons of trunk net.
        trunk_dim_hidden (int): number of neurons in hidden layers of trunk net.
        trunk_num_layers (int): number of layers of trunk net.
        branch_dim_in (int): number of input neurons of branch net.
        branch_dim_hidden (int): number of neurons in hidden layers of branch net.
        branch_num_layers (int): number of layers of branch net.
        dim_out (int): the number of output neurons of trunk net and branch net.
        n_vars (int, optional): the number of the final network output, default 1.

    Inputs:
        - **trunk_in** (Tensor) - input tensor of trunk net,
          shape is :math:`(num\_pdes, num\_points, trunk\_dim\_in)`.
        - **branch_in** (Tensor) - input tensor of branch net,
          shape is :math:`(num\_points, trunk\_dim\_in)`.

    Outputs:
        Output tensor, shape is :math:`(num\_pdes, num\_points, 1)`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from model.baseline.deeponet import DeepONet
        >>> trunk_dim_in = 2
        >>> trunk_dim_hidden = 128
        >>> trunk_num_layers = 5
        >>> branch_dim_in = 129
        >>> branch_dim_hidden = 128
        >>> branch_num_layers = 5
        >>> dim_out = 128
        >>> deeponet = DeepONet(trunk_dim_in, trunk_dim_hidden, trunk_num_layers, \
                     branch_dim_in, branch_dim_hidden, branch_num_layers, \
                     dim_out=dim_out)
        >>> num_pdes = 10
        >>> num_points = 8192
        >>> trunk_in = Tensor(np.random.rand(num_pdes, num_points, trunk_dim_in), dtype=mstype.float32)
        >>> branch_in = Tensor(np.random.rand(num_pdes, branch_dim_in), dtype=mstype.float32)
        >>> out = deeponet(trunk_in, branch_in)  # [num_pdes, num_points, 1]
        >>> print(out.shape)
        (10, 8192, 1)

    """

    def __init__(self,
                 trunk_dim_in: int,
                 trunk_dim_hidden: int,
                 trunk_num_layers: int,
                 branch_dim_in: int,
                 branch_dim_hidden: int,
                 branch_num_layers: int,
                 dim_out: int = 256,
                 num_pos_enc: int = 5,
                 n_vars: int = 1,
                 compute_dtype=mstype.float32) -> None:
        super().__init__()
        self.n_vars = n_vars
        self.pos_enc = CoordPositionalEncoding(num_pos_enc)
        ext_dim_in = trunk_dim_in * (2 * num_pos_enc + 1)

        self.trunk_net = MLP(ext_dim_in, n_vars * dim_out,
                             dim_hidden=trunk_dim_hidden,
                             num_layers=trunk_num_layers,
                             compute_dtype=compute_dtype)

        self.trunk_act = nn.ReLU()

        self.branch_net = MLP(branch_dim_in, dim_out,
                              dim_hidden=branch_dim_hidden,
                              num_layers=branch_num_layers,
                              compute_dtype=compute_dtype)

        self.b0 = Parameter(Tensor([0.0], dtype=compute_dtype))

        self.reduce_sum = ops.ReduceSum(keep_dims=True)

    def construct(self,
                  branch_in: Tensor,
                  trunk_in: Tensor) -> Tensor:
        r"""construct"""
        trunk_in = self.pos_enc(trunk_in)

        # Shape is [num_pdes, num_points, n_vars * dim_out].
        out_trunk = self.trunk_act(self.trunk_net(trunk_in))
        num_pdes, num_points, _ = out_trunk.shape
        # [num_pdes, num_points, n_vars * dim_out] -> [num_pdes, num_points * n_vars, dim_out]
        out_trunk = out_trunk.reshape([num_pdes, num_points * self.n_vars, -1])

        out_branch = self.branch_net(branch_in)  # [num_pdes, dim_out]
        # [num_pdes, 1, dim_out] -> [num_pdes, num_points * n_vars, dim_out]
        out_branch = out_branch.unsqueeze(1).repeat(num_points * self.n_vars, axis=1)

        out = out_trunk * out_branch  # [num_pdes, num_points * n_vars, dim_out]
        out = self.reduce_sum(out, -1)  # [num_pdes, num_points * n_vars, 1]
        out = out + self.b0  # [num_pdes, num_points * n_vars, 1]
        out = out.reshape([num_pdes, num_points, self.n_vars])

        return out  # [num_pdes, num_points, n_vars]


class CNNDeepONet(DeepONet):
    r"""
    The CNN variant of DeepONet model.
    The branch net is replaced with a CNN encoder while keeping the trunk net
    as MLP. CNN adapted from `..pdeformer.function_encoder.Conv2dFuncEncoder`.

    Args:
        trunk_dim_in (int): number of input neurons of trunk net.
        trunk_dim_hidden (int): number of neurons in hidden layers of trunk net.
        trunk_num_layers (int): number of layers of trunk net.
        branch_dim_in (int): number of input channels for branch net.
        dim_out (int): the number of output neurons of trunk net and branch net.
        num_pos_enc (int): number of positional encoding dimensions. Default: 5.
        n_vars (int): number of variables. Default: 1.
        compute_dtype: computation type. Default: mstype.float32.

    Inputs:
        - **branch_in** (Tensor) - input tensor of branch net,
          shape is :math:`(num\_pdes, branch\_dim\_in, 128, 128)`.
        - **trunk_in** (Tensor) - input tensor of trunk net,
          shape is :math:`(num\_pdes, num\_points, trunk\_dim\_in)`.

    Outputs:
        Output tensor, shape is :math:`(num\_pdes, num\_points, 1)`.
    """

    def __init__(self,
                 trunk_dim_in: int,
                 trunk_dim_hidden: int,
                 trunk_num_layers: int,
                 branch_dim_in: int,
                 branch_dim_hidden: int,
                 dim_out: int = 256,
                 num_pos_enc: int = 5,
                 n_vars: int = 1,
                 compute_dtype=mstype.float32) -> None:
        super().__init__(trunk_dim_in=trunk_dim_in,
                         trunk_dim_hidden=trunk_dim_hidden,
                         trunk_num_layers=trunk_num_layers,
                         branch_dim_in=2,  # Effect overriden
                         branch_dim_hidden=2,  # Effect overriden
                         branch_num_layers=2,  # Effect overriden
                         dim_out=dim_out,
                         num_pos_enc=num_pos_enc,
                         n_vars=n_vars,
                         compute_dtype=compute_dtype)

        # Replace branch_net with CNN encoder
        get_activation_fn = nn.ReLU
        conv_kwargs = dict(kernel_size=3, stride=2, has_bias=True,
                           weight_init="HeUniform", bias_init="zeros")
        self.branch_net = nn.SequentialCell([
            nn.Conv2d(branch_dim_in, 16, **conv_kwargs).to_float(compute_dtype),
            get_activation_fn(),  # [bsz, 16, H/2, W/2]
            nn.Conv2d(16, 32, **conv_kwargs).to_float(compute_dtype),
            get_activation_fn(),  # [bsz, 32, H/4, W/4]
            nn.Conv2d(32, 64, **conv_kwargs).to_float(compute_dtype),
            get_activation_fn(),  # [bsz, 64, H/8, W/8]
            nn.Conv2d(64, 128, **conv_kwargs).to_float(compute_dtype),
            get_activation_fn(),  # [bsz, 128, H/16, W/16]
            nn.Conv2d(128, 256, **conv_kwargs).to_float(compute_dtype),
            get_activation_fn(),  # [bsz, 256, H/32, W/32]
            nn.Conv2d(256, 256, **conv_kwargs).to_float(compute_dtype),
            get_activation_fn(),  # [bsz, 256, H/64, W/64]
            nn.Flatten(),  # [bsz, 256*2*2]
            nn.Dense(1024, branch_dim_hidden, has_bias=True, weight_init="HeUniform",
                     bias_init="zeros").to_float(compute_dtype),
            get_activation_fn(),  # [bsz, dim_hidden]
            nn.Dense(branch_dim_hidden, dim_out, has_bias=True, weight_init="HeUniform",
                     bias_init="zeros").to_float(compute_dtype),
        ])  # Output shape: [bsz, dim_out].
