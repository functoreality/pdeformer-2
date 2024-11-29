r"""Load boundary condition v2 from python dictionary instead of HDF5 file."""
from typing import Union, List, Dict

from numpy.typing import NDArray

from src.data.multi_pde import boundary_v2
from src.data.multi_pde.terms_from_dict import PDETermBase, ConstOrField


class EdgeBoundaryConditionV2(PDETermBase):  # pylint: disable=missing-docstring
    __doc__ = boundary_v2.EdgeBoundaryConditionV2.__doc__
    ROBIN_D = 0
    ROBIN_N = 1
    key_name: str
    bc_type: int
    alpha: ConstOrField
    beta2: ConstOrField
    sdf: NDArray[float]

    def __init__(self,
                 term: Dict[str, Union[NDArray[float], float, str]],
                 keep_all_coef: bool) -> None:
        super().__init__(term, keep_all_coef)
        if not isinstance(term, dict):
            raise TypeError("Input 'term' of EdgeBoundaryConditionV2 should be a dictionary.")
        self.key_name = term["key_name"]
        self.bc_type = term["bc_type"]
        self.alpha = ConstOrField(term["alpha"], keep_all_coef)
        beta = ConstOrField(term["beta"], keep_all_coef)
        self.beta2 = -beta  # negation via custom __neg__ functions

    full_type = boundary_v2.EdgeBoundaryConditionV2.full_type
    edge_location = boundary_v2.EdgeBoundaryConditionV2.edge_location
    nodes = boundary_v2.EdgeBoundaryConditionV2.nodes
    add_latex = boundary_v2.EdgeBoundaryConditionV2.add_latex
    record_field_vals_ = boundary_v2.EdgeBoundaryConditionV2.record_field_vals_
    assign_sdf = boundary_v2.EdgeBoundaryConditionV2.assign_sdf


class FullBoundaryConditions(PDETermBase):  # pylint: disable=missing-docstring
    __doc__ = boundary_v2.FullBoundaryConditions.__doc__
    edge_bc_cls: type = EdgeBoundaryConditionV2
    edges_by_type_dict: Dict[str, List[edge_bc_cls]]

    def __init__(self,
                 term: Dict[str, Dict[str, Union[NDArray[float], float, str]]],
                 keep_all_coef: bool) -> None:
        super().__init__(term, keep_all_coef)
        if not isinstance(term, dict):
            raise TypeError("Input 'term' of FullBoundaryConditions "
                            "should be a nested dictionary.")
        edges_by_type_dict = {}
        for key_name in term:
            if not isinstance(term[key_name], dict):
                raise TypeError("Input 'term' of FullBoundaryConditions "
                                "should be a nested dictionary.")
            sub_term = term[key_name].copy()
            sub_term["key_name"] = key_name
            edge_bc = self.edge_bc_cls(sub_term, keep_all_coef)
            edge_type = edge_bc.full_type
            if "c" in edge_type:
                edge_type = key_name  # do not merge, treat separately
            if edge_type not in edges_by_type_dict:
                edges_by_type_dict[edge_type] = []
            edges_by_type_dict[edge_type].append(edge_bc)

        self.edges_by_type_dict = edges_by_type_dict

    nodes = boundary_v2.FullBoundaryConditions.nodes
    add_latex = boundary_v2.FullBoundaryConditions.add_latex
    assign_sdf = boundary_v2.FullBoundaryConditions.assign_sdf
    _get_field_coef_dict = boundary_v2.FullBoundaryConditions._get_field_coef_dict  # pylint: disable=protected-access


class EdgeBCWithMurV2(EdgeBoundaryConditionV2):  # pylint: disable=missing-docstring
    __doc__ = boundary_v2.EdgeBCWithMurV2.__doc__
    MUR_R = 2
    gamma: ConstOrField

    def __init__(self,
                 term: Dict[str, Union[NDArray[float], float, str]],
                 keep_all_coef: bool) -> None:
        super().__init__(term, keep_all_coef)
        if not isinstance(term, dict):
            raise TypeError("Input 'term' of EdgeBCWithMurV2 should be a dictionary.")
        self.gamma = ConstOrField(term["gamma"], keep_all_coef)

    full_type = boundary_v2.EdgeBCWithMurV2.full_type
    nodes = boundary_v2.EdgeBCWithMurV2.nodes
    add_latex = boundary_v2.EdgeBCWithMurV2.add_latex
    record_field_vals_ = boundary_v2.EdgeBCWithMurV2.record_field_vals_
    record_field_locs_ = boundary_v2.EdgeBCWithMurV2.record_field_locs_


class FullBCsWithMur(FullBoundaryConditions):  # pylint: disable=missing-docstring
    __doc__ = boundary_v2.FullBCsWithMur.__doc__
    edge_bc_cls: type = EdgeBCWithMurV2
