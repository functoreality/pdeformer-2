r"""Wrapper of INRs with hypernet."""
from omegaconf import DictConfig
from mindspore import dtype as mstype
from mindspore import nn

from .siren import SirenWithHypernet
from .mfn import MFNNetWithHypernet
from .poly_inr import PolyINRWithHypernet


def get_inr_with_hypernet(config_model: DictConfig,
                          dim_in: int = 1,
                          dim_out: int = 1,
                          inr_base: bool = True,
                          compute_dtype=mstype.float16) -> nn.Cell:
    r"""
    INR with hypernet is used for representing the solution of the PDE at each
    point. It consists of an INR and a Hypernet. The Hypernet takes the PDE
    feature as input, and outputs the modulations to each INR  hidden layer.
    The INR takes the modulated PDE feature and the coordinate of each point as
    input, and outputs the solution of the PDE equation at each point.

    Args:
        config_model (Dict): Configurations.
        dim_in (int): Dimension of INR input coordinates. Default: 1.
        dim_out (int): Dimension of INR outputs. Default: 1.
        compute_dtype (mstype.Float): The computation type of the layer.
            Default: ``mstype.float16``.
        inr_base (bool): Whether it is the base inr of the model, as pdeformer
            may include multiple inrs for decoding.

    Inputs:
         - **coordination** (Tensor) - The coordinate of each point, shape
               :math:`(n\_graph, num\_points, dim\_in)`.
         - **hyper_input** (Tensor) - The PDE feature, shape
               :math:`(n\_inr_node, n\_graph, embed\_dim)`.

    Outputs:
        Predicted field values at the specific coordinate points, with shape
            :math:`(n\_graph, num\_points, dim\_out)`.

    Supported Platforms:
        ``Ascend`` ``GPU``  ``CPU``
    """
    if inr_base:
        config_inr = config_model.inr
    else:
        config_inr = config_model.inr2
    base_kwargs = {
        "inr_dim_in": dim_in,
        "inr_dim_out": dim_out,
        "inr_dim_hidden": config_inr.dim_hidden,
        "inr_num_layers": config_inr.num_layers,
        "hyper_dim_in": config_model.graphormer.embed_dim,
        "hyper_dim_hidden": config_model.hypernet.dim_hidden,
        "hyper_num_layers": config_model.hypernet.num_layers,
        "share_hypernet": config_model.hypernet.shared,
        "compute_dtype": compute_dtype}
    inr_type = config_inr.type.lower()

    if inr_type == "siren":
        return SirenWithHypernet(**base_kwargs, **config_inr.siren)
    if inr_type == "mfn":
        return MFNNetWithHypernet(**base_kwargs, **config_inr.mfn)
    if inr_type == "poly_inr":
        return PolyINRWithHypernet(**base_kwargs, **config_inr.poly_inr)
    raise ValueError("'inr_type' should be in ['siren', 'mfn', 'poly_inr'], "
                     f"but got {config_inr.type}")
