r"""PDEformer model."""
from omegaconf import DictConfig
from mindspore import dtype as mstype
from mindspore import nn, Tensor, ops

from .inr_with_hypernet import get_inr_with_hypernet
from .graphormer.graphormer_encoder import GraphormerEncoder
from ..basic_block import MLP
from .function_encoder import get_function_encoder
from ..env import SPACE_DIM


class PDEEncoder(nn.Cell):
    r"""
    PDEEncoder is used for encoding the input graph and function into a fixed-size representation.
    It consists of a GraphormerEncoder, a scalar encoder and a function encoder.
    The GraphormerEncoder encodes the PDE formulation into a fixed-size representation,
    and the scalar encoder and function encoder encode the scalar and function information
    into a fixed-size representation.

    Args:
        config_model (Dict): Configurations.
        compute_dtype (mstype.Float): The computation type of the layer.
            Default: ``mstype.float16``.

    Inputs:
        - **node_type** (Tensor) - The type of each node, shape :math:`(n\_graph, n\_node, 1)`.
        - **node_scalar** (Tensor) - The scalar value of each node, shape
          :math:`(n\_graph, num\_scalar, 1)`.
        - **node_function** (Tensor) - The function value of each node,
          shape :math:`(n\_graph, num\_function, num\_points\_function, 5)`.
        - **in_degree** (Tensor) - The in-degree of each node, shape :math:`(n\_graph, n\_node)`.
        - **out_degree** (Tensor) - The out-degree of each node, shape :math:`(n\_graph, n\_node)`.
        - **attn_bias** (Tensor) - The attention bias of the graph, shape
          :math:`(n\_graph, n\_node, n\_node)`.
        - **spatial_pos** (Tensor) - The spatial position from each node to each other node,
          shape :math:`(n\_graph, n\_node, n\_node)`.

    Outputs:
        The output representation of teh PDE, shape :math:`(n\_node, n\_graph, embed\_dim)`.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    def __init__(self, config_model: DictConfig, compute_dtype=mstype.float16) -> None:
        super().__init__()

        graphormer_config = config_model.graphormer
        self.graphormer = GraphormerEncoder(
            **config_model.graphormer,
            compute_dtype=compute_dtype
        )

        self.scalar_encoder = MLP(
            1, graphormer_config.embed_dim,
            dim_hidden=config_model.scalar_encoder.dim_hidden,
            num_layers=config_model.scalar_encoder.num_layers,
            compute_dtype=compute_dtype)

        self.function_encoder = get_function_encoder(
            config_model.function_encoder,
            1 + SPACE_DIM + 1,  # [t, x, y, z, f]
            graphormer_config.embed_dim,
            compute_dtype)

    def construct(self,
                  node_type: Tensor,
                  node_scalar: Tensor,
                  node_function: Tensor,
                  in_degree: Tensor,
                  out_degree: Tensor,
                  attn_bias: Tensor,
                  spatial_pos: Tensor) -> Tensor:
        r"""construct"""
        node_scalar_feature = self.scalar_encoder(node_scalar)  # [n_graph, num_scalar, embed_dim]

        (n_graph, num_function, num_points_function, _) = node_function.shape
        node_function = node_function.reshape(
            n_graph * num_function, num_points_function, 1 + SPACE_DIM + 1)
        # deepset, conv2d: [n_graph*num_function, num_branches*embed_dim]
        # patched: [n_graph*num_function*num_branches, embed_dim]
        node_function_feature = self.function_encoder(node_function)
        # Shape is [n_graph, num_function*num_branches, embed_dim].
        node_function_feature = node_function_feature.reshape(
            n_graph, -1, node_scalar_feature.shape[-1])

        # Shape is [n_graph, num_scalar+num_function*num_branches, embed_dim].
        node_input_feature = ops.cat((node_scalar_feature, node_function_feature), axis=1)

        out = self.graphormer(node_type, node_input_feature, in_degree,
                              out_degree, attn_bias, spatial_pos)  # [n_node, n_graph, embed_dim]

        return out  # [n_node, n_graph, embed_dim]


class PDEformer(nn.Cell):
    r"""
    PDEformer consists of a PDEEncoder and an INR (with hypernet). The
    PDEEncoder encodes the PDE into a fixed-size representation, and the INR
    (with hypernet) represents the solution of the PDE equation at each point.
    The PDEformer takes the PDE formulation, the scalar and function
    information as input, and outputs the fixed-size representation of the PDE.
    In addition, the PDE formulation is represented by the graph structure. The
    INR (with hypernet) takes the fixed-size representation of the PDE and the
    coordinate of each point as input, and outputs the solution of the PDE
    equation at each point.

    Args:
        config_model (Dict): Configurations.
        compute_dtype (mstype.Float): The computation type of the layer.
            Default: ``mstype.float16``.

    Inputs:
        - **node_type** (Tensor) - The type of each node, shape :math:`(n\_graph, n\_node, 1)`.
        - **node_scalar** (Tensor) - The scalar value of each node,
          shape :math:`(n\_graph, num\_scalar, 1)`.
        - **node_function** (Tensor) - The function value of each node,
          shape :math:`(n\_graph, num\_function, num\_points\_function, 5)`.
        - **in_degree** (Tensor) - The in-degree of each node, shape :math:`(n\_graph, n\_node)`.
        - **out_degree** (Tensor) - The out-degree of each node, shape :math:`(n\_graph, n\_node)`.
        - **attn_bias** (Tensor) - The attention bias of the graph,
          shape :math:`(n\_graph, n\_node, n\_node)`.
        - **spatial_pos** (Tensor) - The spatial position from each node to each other node,
          shape :math:`(n\_graph, n\_node, n\_node)`.
        - **coordinate** (Tensor) - The coordinate of each point,
          shape :math:`(n\_graph, num\_points, 4)`.

    Outputs:
        The solution of the PDE equation at each point, shape
            :math:`(n\_graph, num\_points, dim\_out)`.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    def __init__(self, config_model: DictConfig, compute_dtype=mstype.float16) -> None:
        super().__init__()
        self.n_inr_nodes = config_model.inr.num_layers - 1

        self.pde_encoder = PDEEncoder(config_model, compute_dtype=compute_dtype)
        self.inr = get_inr_with_hypernet(
            config_model,
            dim_in=1 + SPACE_DIM,  # [t, x, y, z]
            dim_out=1,
            # dim_in=SPACE_DIM,  # [x, y, z]; <spatial-INR>
            # dim_out=101,  # n_t_grid
            compute_dtype=compute_dtype)
        self.multi_inr = config_model.multi_inr.enable
        self.separate_latent = config_model.multi_inr.get("separate_latent", False)
        if self.multi_inr:
            self.inr2 = get_inr_with_hypernet(
                config_model,
                dim_in=1 + SPACE_DIM,  # [t, x, y, z]
                dim_out=1,
                inr_base=False,
                compute_dtype=compute_dtype)

    def construct(self,
                  node_type: Tensor,
                  node_scalar: Tensor,
                  node_function: Tensor,
                  in_degree: Tensor,
                  out_degree: Tensor,
                  attn_bias: Tensor,
                  spatial_pos: Tensor,
                  coordinate: Tensor) -> Tensor:
        r"""construct"""
        pde_feature = self.pde_encoder(node_type, node_scalar, node_function, in_degree,
                                       out_degree, attn_bias, spatial_pos)

        # Shape is [n_graph, num_points, dim_out].
        out = self.inr(coordinate, pde_feature[:self.n_inr_nodes])
        if self.multi_inr:
            if self.separate_latent:
                out2 = self.inr2(coordinate, pde_feature[self.n_inr_nodes:])
            else:
                out2 = self.inr2(coordinate, pde_feature[:self.n_inr_nodes])
            return out + out2
        return out
