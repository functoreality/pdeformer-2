r"""
Generate DAG nodes as well as LaTeX representations for the boundary conditions
(BCs) involved in the custom multi_pde dataset. Employing the interface
`PDENodesCollector.bc_sum_eq0`, the BCs are represented in the version 2 form
$Bu = 0$, in which $B$ may contain inhomogeneous terms.
"""
from typing import Union, Tuple, List, Dict, Optional

import numpy as np
from numpy.typing import NDArray
import h5py

from ..pde_dag import PDENodesCollector, PDENode, ExtendedFunc, merge_extended_func
from .terms import PDETermBase, LaTeXSum, ConstOrField
from .boundary import DirichletOrNeumannBoxDomainBC


class EdgeBoundaryConditionV2(PDETermBase):
    r"""
    Non-periodic boundary condition (BC) of a PDE, applied to one single edge
    (surface) of a square (cubic) domain (if the domain is non-periodic along
    this axis), or the outer edge of a disk domain. The BC takes the general
    form $Bu(r)=0$ for $r\in\Gamma$.

    The boundary operator $Bu$ is taken to be Robin with D-type
    $Bu = u + b(r)\partial u/\partial n + c(r)$, or Robin with N-type
    $Bu = a(r)u + \partial u/\partial n + c(r)$, with equal
    probability.

    Each of the coefficient field $a(r),b(r),c(r)$ is taken to be zero, one, a
    random scalar, or a random field with certain probability. Note that when
    $a(r)$ or $b(r)$ equals zero, the boundary condition would degenerate to
    the Dirichlet type or the Neumann type. We may also set $c(r)$ to meet the
    initial condition.
    """
    ROBIN_D = 0
    ROBIN_N = 1
    key_name: str
    bc_type: int
    alpha: ConstOrField
    beta2: ConstOrField
    sdf: NDArray[float]

    def __init__(self,
                 key_name: str,
                 hdf5_group: h5py.Group,
                 idx_pde: Union[int, Tuple[int]],
                 keep_all_coef: bool) -> None:
        super().__init__(hdf5_group, idx_pde, keep_all_coef)
        self.key_name = key_name
        self.bc_type = hdf5_group["bc_type"][idx_pde]
        self.alpha = ConstOrField(hdf5_group["alpha"], idx_pde, keep_all_coef)
        beta = ConstOrField(hdf5_group["beta"], idx_pde, keep_all_coef)
        self.beta2 = -beta  # negation via custom __neg__ functions

    @property
    def full_type(self) -> str:
        r"""
        Return the string representation of the BC type, containing 4 chars
        representing the types of coefficient of $u_t$, $u$, $u_n$, $1$,
        respectively.
        Eg. "011c$ for inhomogeneous Robin BC $u+u_n+c=0$,
            "10f0$ for Mur BC $u_t+b(r)u_n=0$.
        """
        alpha_char = self.alpha.type_char
        beta_char = self.beta2.type_char
        if self.bc_type == self.ROBIN_D:
            type_str = "01" + alpha_char + beta_char
        elif self.bc_type == self.ROBIN_N:
            type_str = "0" + alpha_char + "1" + beta_char
        else:
            raise RuntimeError(f"Unexpected 'bc_type' {self.bc_type}.")
        return type_str

    @property
    def edge_location(self) -> str:
        r"""
        Location of the current edge. Values include:
            L (left, x_low), R (right, x_high),
            D (down, y_low), U (up, y_high),
            B (back, z_low), F (front, z_high),
            I (inner, r_low), O (outer, r_high).
        """
        if self.key_name == "inner":
            return "I"
        if self.key_name == "outer":
            return "O"

        i_ax = "xyzw".find(self.key_name[0])
        if self.key_name.endswith("high"):
            return "RUF"[i_ax]
        if self.key_name.endswith("low"):
            return "LDB"[i_ax]
        return self.key_name

    def nodes(self,  # pylint: disable=arguments-differ
              pde: PDENodesCollector,
              u_node: PDENode = None,
              domain: PDENode = None,
              boundary_sdf: PDENode = None,
              field_coef_dict: Dict[str, PDENode] = None) -> None:
        if None in [u_node, domain, boundary_sdf, field_coef_dict]:
            raise TypeError("Arguments 'u_node', 'domain', 'boundary_sdf' and "
                            "'field_coef_dict' must be specified.")

        def coef_node(role, coef_obj):
            if coef_obj.type_char == "f":
                return field_coef_dict[role]
            return coef_obj.nodes(pde)

        beta2 = coef_node("1", self.beta2)

        # sum_list
        if self.bc_type == self.ROBIN_D:
            alpha = coef_node("un", self.alpha)
            sum_list = [u_node, beta2] + pde.dn_sum_list(
                u_node, domain, coef=alpha)
        elif self.bc_type == self.ROBIN_N:
            alpha = coef_node("u", self.alpha)
            sum_list = [alpha * u_node, beta2] + pde.dn_sum_list(
                u_node, domain)

        pde.bc_sum_eq0(boundary_sdf, sum_list)

    def add_latex(self,  # pylint: disable=arguments-differ
                  latex_sum: LaTeXSum,
                  boundary_loc: str = None,
                  field_loc_dict: Dict[str, str] = None,
                  symbol: str = "u") -> str:
        if boundary_loc is None or field_loc_dict is None:
            raise TypeError("Arguments 'boundary_loc' and 'field_loc_dict' "
                            "must be specified.")
        if latex_sum.term_list:  # list non-empty
            raise RuntimeError("Expected empty 'latex_sum', but it has terms "
                               f"{latex_sum.term_list}.")

        def add_coef(coef_obj, role, coef_symbol, term_latex=""):
            if coef_obj.type_char == "f":
                subscript = field_loc_dict[role]
            else:
                subscript = boundary_loc
            if len(subscript) > 1:
                subscript = "{" + subscript + "}"
            subscript = "_" + subscript
            coef_obj.add_latex(latex_sum, coef_symbol + subscript, term_latex)

        # homogeneous part
        if self.bc_type == self.ROBIN_D:
            latex_sum.add_term(symbol)
            add_coef(self.alpha, "un", "b", r"\partial_n" + symbol)
        elif self.bc_type == self.ROBIN_N:
            latex_sum.add_term(r"\partial_n" + symbol)
            add_coef(self.alpha, "u", "a", symbol)
        else:
            raise RuntimeError(f"Unexpected 'bc_type' {self.bc_type}.")

        # generate bc_latex
        add_coef(self.beta2, "1", "c")
        bc_latex = latex_sum.strip_sum("(", ")|_{" + boundary_loc + "}=0")
        return bc_latex

    def record_field_vals_(
            self, field_by_role_dict: Dict[str, List[ExtendedFunc]]) -> None:
        r"""Record field-valued coefficients to a dictionary."""
        def append2key(role, coef_obj):
            if coef_obj.type_char != "f":
                return
            ext_values = coef_obj.field
            # 2D only. Eg. When 'sdf' has shape [n_x, 1], 'ext_values' should
            # be [1, n_y]. There are also cases that both have full shape
            # [n_x, n_y].
            ext_values = ext_values.reshape(self.sdf.T.shape)
            ext_field = ExtendedFunc(self.sdf, ext_values)
            if role not in field_by_role_dict:
                raise KeyError(f"Unknown dict key {role}. Please pre-specify "
                               "all possible keys in 'field_by_role_dict'.")
                # field_by_role_dict[role] = []
            field_by_role_dict[role].append(ext_field)

        append2key("1", self.beta2)
        if self.bc_type == self.ROBIN_D:
            append2key("un", self.alpha)
        elif self.bc_type == self.ROBIN_N:
            append2key("u", self.alpha)
        else:
            raise RuntimeError

    def record_field_locs_(self, field_loc_dict: Dict[str, str]) -> None:
        r"""Record subscripts of field-valued coefficients to a dictionary."""
        edge_location = self.edge_location
        if self.beta2.type_char == "f":
            field_loc_dict["1"] += edge_location
        if self.alpha.type_char == "f":
            if self.bc_type == self.ROBIN_D:
                field_loc_dict["un"] += edge_location
            elif self.bc_type == self.ROBIN_N:
                field_loc_dict["u"] += edge_location
            else:
                raise RuntimeError

    def assign_sdf(self, sdf_dict: Dict[str, NDArray[float]]) -> None:
        r"""Assign the signed-distance-function values of the current edge."""
        sdf = sdf_dict[self.key_name]
        self.sdf = np.abs(sdf)  # edges has no 'interior'


class FullBoundaryConditions(PDETermBase):
    r"""
    Non-periodic boundary condition of a PDE, including all edges (surfaces) of
    a square (cubic) domain (unless the domain is periodic along this axis), or
    the outer edge of a disk domain.

    For each edge or surface, the boundary condition takes the general form
    $B_iu(r)=0$ for $r\in\Gamma_i$.
    Each boundary operator $B_iu$ is taken to be Robin with D-type
    $B_iu = u + b_i(r)\partial u/\partial n + c_i(r)$, or Robin with N-type
    $B_iu = a_i(r)u + \partial u/\partial n + c_i(r)$, with equal probability.

    Each of the coefficient field $a_i(r),b_i(r),c_i(r)$ is taken to be zero,
    one, a random scalar, or a random field with certain probability. Note that
    when $a_i(r)$ or $b_i(r)$ equals zero, the boundary condition would
    degenerate to the Dirichlet type or the Neumann type. We may also set
    $c_i(r)$ to meet the initial condition.
    """
    edge_bc_cls: type = EdgeBoundaryConditionV2
    edges_by_type_dict: Dict[str, List[edge_bc_cls]]

    def __init__(self,
                 hdf5_group: h5py.Group,
                 idx_pde: Union[int, Tuple[int]],
                 keep_all_coef: bool) -> None:
        super().__init__(hdf5_group, idx_pde, keep_all_coef)
        edges_by_type_dict = {}
        for key_name in hdf5_group:
            edge_bc = self.edge_bc_cls(
                key_name, hdf5_group[key_name], idx_pde, keep_all_coef)
            edge_type = edge_bc.full_type
            if "c" in edge_type:
                edge_type = key_name  # do not merge, treat separately
            if edge_type not in edges_by_type_dict:
                edges_by_type_dict[edge_type] = []
            edges_by_type_dict[edge_type].append(edge_bc)

        self.edges_by_type_dict = edges_by_type_dict

    def nodes(self,  # pylint: disable=arguments-differ
              pde: PDENodesCollector,
              u_node: PDENode = None,
              domain: PDENode = None,
              coord_dict: Dict[str, NDArray[float]] = None,
              *,
              boundary_sdf_dict: Optional[Dict[str, PDENode]] = None) -> None:
        r"""
        Generate the node representation for the boundary conditions (BCs). The
        argument 'boundary_sdf_dict', when given, may be modified with
        additional entries during method call. When we need to specify the BCs
        for multiple variables, this functionality avoids repeated creation of
        the SDF node representation for the same boundary, and one SDF node can
        be used by different field variables $u_i$.
        """
        if None in [u_node, domain, coord_dict]:
            raise TypeError("Arguments 'u_node', 'domain', and 'coord_dict' "
                            "must be specified.")
        if boundary_sdf_dict is None:
            boundary_sdf_dict = {}

        # prepare field_coef_dict
        field_coef_dict = self._get_field_coef_dict()
        # dict values: type NDArray[float] -> PDENode
        field_coef_dict = {role: pde.new_coef_field(field, **coord_dict)
                           for role, field in field_coef_dict.items()}

        # proceed for each edge_type
        for edge_list in self.edges_by_type_dict.values():
            # location of boundary
            boundary_loc = ""
            for edge_obj in edge_list:
                boundary_loc += edge_obj.edge_location

            # prepare boundary_sdf
            if boundary_loc not in boundary_sdf_dict:
                boundary_sdf = [edge_obj.sdf for edge_obj in edge_list]
                # List[NDArray[float]] -> NDArray[float]
                boundary_sdf = np.stack(np.broadcast_arrays(*boundary_sdf))
                # [n_edge, n_x, n_y] -> [n_x, n_y]
                boundary_sdf = boundary_sdf.min(axis=0)
                # NDArray[float] -> PDENode
                boundary_sdf = pde.new_domain(boundary_sdf, **coord_dict)
                boundary_sdf_dict[boundary_loc] = boundary_sdf
            else:
                boundary_sdf = boundary_sdf_dict[boundary_loc]

            # Create BC nodes. Note that all edges in 'edge_list' has the same
            # type, thus having the same 'nodes' behavior. We use the first
            # edge for simplicity.
            edge_list[0].nodes(
                pde, u_node, domain, boundary_sdf, field_coef_dict)

    def add_latex(self,  # pylint: disable=arguments-differ
                  latex_sum: LaTeXSum,
                  symbol: str = "u",
                  bc_subscript: Union[str, int] = "") -> List[str]:
        if latex_sum.term_list:  # list non-empty
            raise RuntimeError("Expected empty 'latex_sum', but it has terms "
                               f"{latex_sum.term_list}.")

        # prepare field_loc_dict
        field_loc_dict = {role: str(bc_subscript) for role in ["u", "un", "1"]}
        for edge_list in self.edges_by_type_dict.values():
            for edge_obj in edge_list:
                edge_obj.record_field_locs_(field_loc_dict)

        # proceed for each edge_type
        bc_list = []  # List[str], one entry for each BC equation
        for edge_list in self.edges_by_type_dict.values():
            boundary_loc = ""
            for edge_obj in edge_list:
                boundary_loc += edge_obj.edge_location

            bc_latex = edge_list[0].add_latex(
                latex_sum, boundary_loc, field_loc_dict)
            bc_list.append(bc_latex)

        return bc_list

    def assign_sdf(self, sdf_dict: Dict[str, NDArray[float]]) -> None:
        r"""Assign the signed-distance-function values of all edges."""
        for edge_list in self.edges_by_type_dict.values():
            for edge_obj in edge_list:
                edge_obj.assign_sdf(sdf_dict)

    def _get_field_coef_dict(self) -> Dict[str, NDArray[float]]:
        r"""Obtain field_coef_dict for method 'nodes'."""
        field_by_role_dict = {"u": [], "un": [], "1": []}
        for edge_list in self.edges_by_type_dict.values():
            for edge_obj in edge_list:
                edge_obj.record_field_vals_(field_by_role_dict)
        # dict values: type List[ExtendedFunc] -> NDArray[float]
        field_coef_dict = {}
        for role, func_list in field_by_role_dict.items():
            if not func_list:  # list empty
                continue
            _, field = merge_extended_func(func_list)
            field_coef_dict[role] = field
        return field_coef_dict


class EdgeBCWithMurV2(EdgeBoundaryConditionV2):
    r"""
    Non-periodic boundary condition (BC) of a PDE, applied to one single edge
    (surface) of a square (cubic) domain (if the domain is non-periodic along
    this axis), or the outer edge of a disk domain. The BC takes the general
    form $Bu(r)=0$ for $r\in\Gamma$.

    The boundary operator $Bu$ is taken to be Robin with D-type
    $Bu = u + b(r)\partial u/\partial n + c(r)$, Robin with N-type
    $Bu = a(r)u + \partial u/\partial n + c(r)$, or (generalized) Mur type
    $Bu = u_t + a(r)u + b(r)\partial u/\partial n + c(r)$, with equal
    probability.
    """
    MUR_R = 2
    gamma: ConstOrField

    def __init__(self,
                 key_name: str,
                 hdf5_group: h5py.Group,
                 idx_pde: Union[int, Tuple[int]],
                 keep_all_coef: bool) -> None:
        super().__init__(key_name, hdf5_group, idx_pde, keep_all_coef)
        self.gamma = ConstOrField(hdf5_group["gamma"], idx_pde, keep_all_coef)

    @property
    def full_type(self) -> str:
        if self.bc_type != self.MUR_R:
            return super().full_type
        return ("1" + self.alpha.type_char + self.gamma.type_char
                + self.beta2.type_char)

    def nodes(self,
              pde: PDENodesCollector,
              u_node: PDENode = None,
              domain: PDENode = None,
              boundary_sdf: PDENode = None,
              field_coef_dict: Dict[str, PDENode] = None) -> None:
        if self.bc_type != self.MUR_R:
            super().nodes(pde, u_node, domain, boundary_sdf, field_coef_dict)
            return
        if None in [u_node, domain, boundary_sdf, field_coef_dict]:
            raise TypeError("Arguments 'u_node', 'domain', 'boundary_sdf' and "
                            "'field_coef_dict' must be specified.")

        def coef_node(role, coef_obj):
            if coef_obj.type_char == "f":
                return field_coef_dict[role]
            return coef_obj.nodes(pde)

        alpha = coef_node("u", self.alpha)
        gamma = coef_node("un", self.gamma)
        beta2 = coef_node("1", self.beta2)
        sum_list = [u_node.dt, alpha * u_node, beta2] + pde.dn_sum_list(
            u_node, domain, coef=gamma)
        pde.bc_sum_eq0(boundary_sdf, sum_list)

    def add_latex(self,
                  latex_sum: LaTeXSum,
                  boundary_loc: str = None,
                  field_loc_dict: Dict[str, str] = None,
                  symbol: str = "u") -> str:
        if self.bc_type != self.MUR_R:
            return super().add_latex(
                latex_sum, boundary_loc, field_loc_dict, symbol)

        if boundary_loc is None or field_loc_dict is None:
            raise TypeError("Arguments 'boundary_loc' and 'field_loc_dict' "
                            "must be specified.")
        if latex_sum.term_list:  # list non-empty
            raise RuntimeError("Expected empty 'latex_sum', but it has terms "
                               f"{latex_sum.term_list}.")

        def add_coef(coef_obj, role, coef_symbol, term_latex=""):
            if coef_obj.type_char == "f":
                subscript = field_loc_dict[role]
            else:
                subscript = boundary_loc
            if len(subscript) > 1:
                subscript = "{" + subscript + "}"
            subscript = "_" + subscript
            coef_obj.add_latex(latex_sum, coef_symbol + subscript, term_latex)

        # LHS
        latex_sum.add_term(r"\partial_t" + symbol)
        add_coef(self.alpha, "u", "a", symbol)
        add_coef(self.gamma, "un", "b", r"\partial_n" + symbol)
        add_coef(self.beta2, "1", "c")
        bc_latex = latex_sum.strip_sum("(", ")|_{" + boundary_loc + "}=0")
        return bc_latex

    def record_field_vals_(
            self, field_by_role_dict: Dict[str, List[ExtendedFunc]]) -> None:
        if self.bc_type != self.MUR_R:
            super().record_field_vals_(field_by_role_dict)
            return

        def append2key(role, coef_obj):
            if coef_obj.type_char != "f":
                return
            ext_values = coef_obj.field
            # 2D only. Eg. When 'sdf' has shape [n_x, 1], 'ext_values' should
            # be [1, n_y]. There are also cases that both has full shape
            # [n_x, n_y].
            ext_values = ext_values.reshape(self.sdf.T.shape)
            ext_field = ExtendedFunc(self.sdf, ext_values)
            field_by_role_dict[role].append(ext_field)

        append2key("u", self.alpha)
        append2key("un", self.gamma)
        append2key("1", self.beta2)

    def record_field_locs_(self, field_loc_dict: Dict[str, str]) -> None:
        if self.bc_type != self.MUR_R:
            super().record_field_locs_(field_loc_dict)
            return
        edge_location = self.edge_location
        if self.alpha.type_char == "f":
            field_loc_dict["u"] += edge_location
        if self.gamma.type_char == "f":
            field_loc_dict["un"] += edge_location
        if self.beta2.type_char == "f":
            field_loc_dict["1"] += edge_location


class FullBCsWithMur(FullBoundaryConditions):
    r"""
    Non-periodic boundary condition of a PDE, including all edges (surfaces) of
    a square (cubic) domain (unless the domain is periodic along this axis), or
    the outer edge of a disk domain.

    For each edge or surface, the boundary condition takes the general form
    $B_iu(r)=0$ for $r\in\Gamma_i$.
    Each boundary operator $B_iu$ is taken to be Robin with D-type
    $B_iu = u + b_i(r)\partial u/\partial n + c_i(r)$, Robin with N-type
    $B_iu = a_i(r)u + \partial u/\partial n + c_i(r)$, or (generalized) Mur type
    $B_iu = u_t + a_i(r)u + b_i(r)\partial u/\partial n + c_i(r)$, with equal
    probability.

    Each of the coefficient field $a_i(r),b_i(r),c_i(r)$ is taken to be zero,
    one, a random scalar, or a random field with certain probability. Note that
    when $a_i(r)$ or $b_i(r)$ equals zero, the boundary condition would
    degenerate to the Dirichlet type or the Neumann type. We may also set
    $c_i(r)$ to meet the initial condition.
    """
    edge_bc_cls: type = EdgeBCWithMurV2


class DOrNBoxDomainBCV2(DirichletOrNeumannBoxDomainBC):  # pylint: disable=missing-docstring
    __doc__ = DirichletOrNeumannBoxDomainBC.__doc__

    @staticmethod
    def _bc_node(pde: PDENodesCollector,
                 sum_list: List[PDENode],
                 sdf: NDArray[float],
                 beta: NDArray[float],
                 coord_dict: Dict[str, NDArray[float]]) -> None:
        sum_list = sum_list.copy()
        beta2 = -beta
        mean_beta2 = np.mean(beta2)
        if not np.allclose(beta2, mean_beta2, atol=1e-3):  # FIELD_COEF
            sum_list.append(pde.new_coef_field(beta2, **coord_dict))
        elif np.abs(mean_beta2) > 1e-3:  # SCALAR_COEF
            sum_list.append(mean_beta2)

        boundary = pde.new_domain(sdf, **coord_dict)
        pde.bc_sum_eq0(boundary, sum_list)

    @staticmethod
    def _bc_latex(latex_sum: LaTeXSum,
                  bc_value: ConstOrField,
                  lhs: str,
                  subscript: str) -> str:
        latex_sum.add_term(lhs)
        neg_bc_value = -bc_value
        neg_bc_value.add_latex(latex_sum, rf"\beta_{subscript}")
        return latex_sum.strip_sum("(", ")|_" + subscript + "=0")
