r"""
Generate DAG nodes as well as LaTeX representations for the boundary conditions
(BCs) involved in the custom multi_pde dataset. Employing the interface
`PDENodesCollector.set_bv`, the BCs are represented in the version 1 form
$Bu = \beta(r)$, in which $B$ is a linear operator with only homogeneous terms.
"""
from typing import Union, Tuple, List, Dict

import numpy as np
from numpy.typing import NDArray
import h5py

from ..pde_dag import PDENodesCollector, PDENode, ExtendedFunc, merge_extended_func
from .terms import PDETermBase, LaTeXSum, ConstOrField, ReNamedConstOrField


class EdgeBoundaryCondition(PDETermBase):
    r"""
    Non-periodic boundary condition of a PDE, applied to one single edge of the
    square domain, taking the general form $Bu(r)=\beta(r)$ for $r\in\Gamma_i$.

    The linear boundary operator $Bu$ is taken to be Robin with (D) type
    $Bu=u+\alpha(r)\partial u/\partial n$, and Robin with (N) type
    $Bu=\alpha(r)u+\partial u/\partial n$, with equal probability.

    The coefficient field $\alpha(r)$ as well as the term $\beta(r)$ are taken
    to be zero, one, a random scalar, or a random field with certain
    probability. Note that when $\alpha(r)$ equals zero, the boundary condition
    would degenerate to the Dirichlet type or the Neumann type. We may also set
    $\beta(r)$ to meet the initial condition.
    """
    ROBIN_D = 0
    ROBIN_N = 1
    bc_type: int
    alpha: ConstOrField
    beta: ConstOrField
    key_name: str
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
        self.beta = ConstOrField(hdf5_group["beta"], idx_pde, keep_all_coef)

    @property
    def full_type(self) -> str:
        r"""
        Return the string representation of the BC type.
        Possible returned values: D, N, R1, RDc, RNc, RDf, RNf.
        """
        alpha_type = self.alpha.coef_type
        if alpha_type == self.alpha.UNIT_COEF:
            return "R1"

        # type_char
        if self.bc_type == self.ROBIN_D:
            type_char = "D"
        elif self.bc_type == self.ROBIN_N:
            type_char = "N"
        else:
            raise RuntimeError(f"Unexpected 'bc_type' {self.bc_type}.")
        if alpha_type == self.alpha.ZERO_COEF:
            return type_char

        # alpha_char
        if alpha_type == self.alpha.SCALAR_COEF:
            alpha_char = "c"
        elif alpha_type == self.alpha.FIELD_COEF:
            alpha_char = "f"
        else:
            raise RuntimeError(f"Unexpected 'alpha_type' {alpha_type}.")
        return "R" + type_char + alpha_char

    @property
    def edge_subscript(self) -> str:
        r"""
        Subscript representing the current edge. Values include:
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
              coord_dict: Dict[str, NDArray[float]] = None) -> None:
        if None in [u_node, domain, coord_dict]:
            raise ValueError("Arguments 'u_node', 'domain', and 'coord_dict' "
                             "must be specified.")
        if self.alpha.coef_type != self.alpha.SCALAR_COEF:
            raise RuntimeError

        # sum_list
        alpha = self.alpha.nodes(pde, coord_dict)
        if self.bc_type == self.ROBIN_D:
            sum_list = [u_node] + pde.dn_sum_list(u_node, domain, coef=alpha)
        elif self.bc_type == self.ROBIN_N:
            sum_list = [alpha * u_node] + pde.dn_sum_list(u_node, domain)
        else:
            raise RuntimeError(f"Unexpected 'bc_type' {self.bc_type}.")

        # final BV node
        sdf, beta = self.get_ext_field()
        pde.set_bv(pde.sum(sum_list), sdf, beta, **coord_dict)

    def add_latex(self,  # pylint: disable=arguments-differ
                  latex_sum: LaTeXSum,
                  symbol: str = "u",
                  bc_subscript: Union[str, int] = "") -> str:
        if latex_sum.term_list:  # list non-empty
            raise RuntimeError("Expected empty 'latex_sum', but it has terms "
                               f"{latex_sum.term_list}.")
        subscript = self.edge_subscript
        if bc_subscript != "":
            subscript = "{" + str(bc_subscript) + subscript + "}"

        # LHS
        self.alpha.add_latex(latex_sum, rf"\alpha_{subscript}")
        _ = latex_sum.strip_sum()  # clear its term_list
        if self.bc_type == self.ROBIN_D:
            lhs = rf"({symbol}+\alpha_{subscript}\partial_n{symbol})"
        elif self.bc_type == self.ROBIN_N:
            lhs = rf"(\partial_n{symbol}+\alpha_{subscript}{symbol})"
        else:
            raise RuntimeError(f"Unexpected 'bc_type' {self.bc_type}.")
        lhs = lhs + "|_" + subscript

        # RHS
        self.beta.add_latex(latex_sum, rf"\beta_{subscript}")
        rhs = latex_sum.strip_sum()  # get the beta term
        if rhs == "":
            rhs = "0"
        return lhs + "=" + rhs

    def get_ext_field(self, coef_name: str = "beta") -> ExtendedFunc:
        r"""Get the boundary values, extended to the whole domain."""
        coef_obj = getattr(self, coef_name)
        ext_values = coef_obj.field
        # 2D only. Eg. When 'sdf' has shape [n_x, 1], 'ext_values' should
        # be [1, n_y]. There are also cases that both has full shape
        # [n_x, n_y].
        ext_values = ext_values.reshape(self.sdf.T.shape)
        return ExtendedFunc(self.sdf, ext_values)

    def assign_sdf(self, sdf_dict: Dict[str, NDArray[float]]) -> None:
        r"""Assign the signed-distance-function values of the current edge."""
        sdf = sdf_dict[self.key_name]
        self.sdf = np.abs(sdf)  # edges has no 'interior'


class BoxDomainBoundaryCondition(PDETermBase):
    r"""
    Non-periodic boundary condition of a PDE, including all edges (surfaces) of
    a square (cubic) domain, unless the domain is periodic along this axis.

    For each edge or surface, the boundary condition takes the general form
    $Bu(r)=\beta(r)$ for $r\in\Gamma_i$.
    The linear boundary operator $Bu$ is taken to be Robin with (D) type
    $Bu=u+\alpha(r)\partial u/\partial n$, and Robin with (N) type
    $Bu=\alpha(r)u+\partial u/\partial n$, with equal probability.

    The coefficient field $\alpha(r)$ as well as the term $\beta(r)$ are taken
    to be zero, one, a random scalar, or a random field with certain
    probability. Note that when $\alpha(r)$ equals zero, the boundary
    condition would degenerate to the Dirichlet type or the Neumann type. We
    may also set $\beta(r)$ to meet the initial condition.
    """
    SCALAR_AS_FIELD_THRES = 3
    edge_bc_cls: type = EdgeBoundaryCondition
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
            if edge_type not in edges_by_type_dict:
                edges_by_type_dict[edge_type] = []
            edges_by_type_dict[edge_type].append(edge_bc)

        self._merge_scalar_bc_(edges_by_type_dict, self.SCALAR_AS_FIELD_THRES)
        self.edges_by_type_dict = edges_by_type_dict

    def nodes(self,  # pylint: disable=arguments-differ
              pde: PDENodesCollector,
              u_node: PDENode = None,
              domain: PDENode = None,
              coord_dict: Dict[str, NDArray[float]] = None,
              *,
              ignored_types: Tuple[str] = ()) -> None:
        if None in [u_node, domain, coord_dict]:
            raise ValueError("Arguments 'u_node', 'domain', and 'coord_dict' "
                             "must be specified.")

        for edge_type, edge_list in self.edges_by_type_dict.items():
            if edge_type in ignored_types:
                continue
            if edge_type == "Xc":
                for edge_obj in edge_list:
                    edge_obj.nodes(pde, u_node, domain, coord_dict)
                continue

            # merged cases
            # bc_node
            if edge_type == "D":
                bc_node = u_node
            elif edge_type == "N":
                bc_node = pde.sum(pde.dn_sum_list(u_node, domain))
            elif edge_type == "R1":
                bc_node = pde.sum([u_node] + pde.dn_sum_list(u_node, domain))
            else:
                _, alpha = merge_extended_func([
                    edge_obj.get_ext_field("alpha")
                    for edge_obj in edge_list])  # NDArray[float]
                alpha = pde.new_coef_field(alpha, **coord_dict)
                if edge_type == "RDf":
                    bc_node = pde.sum([u_node] + pde.dn_sum_list(
                        u_node, domain, coef=alpha))
                elif edge_type == "RNf":
                    bc_node = pde.sum([alpha * u_node] + pde.dn_sum_list(
                        u_node, domain))
                else:
                    raise RuntimeError(f"Unexpected 'edge_type' {edge_type}.")

            # final BV node
            sdf, beta = merge_extended_func([
                edge_obj.get_ext_field() for edge_obj in edge_list])
            pde.set_bv(bc_node, sdf, beta, **coord_dict)

    def add_latex(self,  # pylint: disable=arguments-differ
                  latex_sum: LaTeXSum,
                  symbol: str = "u",
                  bc_subscript: Union[str, int] = "",
                  *,
                  ignored_types: Tuple[str] = ()) -> List[str]:
        if latex_sum.term_list:  # list non-empty
            raise RuntimeError("Expected empty 'latex_sum', but it has terms "
                               f"{latex_sum.term_list}.")
        bc_list = []
        for edge_type, edge_list in self.edges_by_type_dict.items():
            if edge_type in ignored_types:
                continue
            if edge_type == "Xc":  # non-merged cases
                for edge_obj in edge_list:
                    bc_list.append(edge_obj.add_latex(
                        latex_sum, symbol, bc_subscript))
                continue

            # merged cases
            # subscript
            subscript = "{" + str(bc_subscript)
            for edge_obj in edge_list:
                subscript += edge_obj.edge_subscript
            subscript += "}"

            # LHS
            if edge_type == "D":
                lhs = symbol
            elif edge_type == "N":
                lhs = r"\partial_n" + symbol
            elif edge_type == "R1":
                lhs = rf"(\partial_n{symbol}+{symbol})"
            elif edge_type == "RDf":
                lhs = rf"({symbol}+\alpha_{subscript}(r)\partial_n{symbol})"
            elif edge_type == "RNf":
                lhs = rf"(\partial_n{symbol}+\alpha_{subscript}(r){symbol})"
            else:
                raise RuntimeError(f"Unexpected 'edge_type' {edge_type}.")

            bc_list.append(rf"{lhs}|_{subscript}=\beta_{subscript}(r)")

        return bc_list

    @classmethod
    def _merge_scalar_bc_(cls,
                          edges_dict: Dict[str, List[EdgeBoundaryCondition]],
                          n_bc_thres: int) -> None:
        r"""
        When there are too many edge types, treat scalar-valued alpha as
        field-valued ones to allow merging, which simplifies the PDE form.
        """
        rdc_list = edges_dict.pop("RDc", [])
        rnc_list = edges_dict.pop("RNc", [])

        # non-merging case
        if len(edges_dict) + len(rdc_list) + len(rnc_list) <= n_bc_thres:
            # if rdc_list or rnc_list:  # list non-empty
            edges_dict["Xc"] = rdc_list + rnc_list  # allow empty list
            return

        # merging case
        edges_dict["Xc"] = []
        cls._merge_c_into_f_(edges_dict, rdc_list, "RDf")
        cls._merge_c_into_f_(edges_dict, rnc_list, "RNf")

    @staticmethod
    def _merge_c_into_f_(edges_dict: Dict[str, List],
                         c_list: List[EdgeBoundaryCondition],
                         f_key: str) -> None:
        r"""
        Private method used by `_merge_scalar_bc_`. Merging scalar-type
        boundaries (specified by `c_list`) into a field-type one (specified by
        `f_key`).
        """
        if f_key in edges_dict:  # merge 'c' into 'f'
            edges_dict[f_key].extend(c_list)
        elif len(c_list) >= 2:  # merge all 'c' as a single 'f'
            edges_dict[f_key] = c_list
        else:  # merging does not simplify the PDE, so do not merge
            edges_dict["Xc"].extend(c_list)

    def assign_sdf(self, sdf_dict: Dict[str, NDArray[float]]) -> None:
        r"""Assign the signed-distance-function values of all edges."""
        for edge_obj_list in self.edges_by_type_dict.values():
            for edge_obj in edge_obj_list:
                edge_obj.assign_sdf(sdf_dict)


class DirichletOrNeumannBoxDomainBC(PDETermBase):
    r"""
    Boundary condition for a PDE on a rectangular domain. Boundary condition on
    each side for each component is randomly chosen from Dirichlet and Neumann.
    So the total number of boundary conditions is 4*n_vars.
    """
    DIRICHLET = 0
    NEUMANN = 1
    SUPPORTED_BC_TYPES = [DIRICHLET, NEUMANN]

    def __init__(self,
                 hdf5_group: h5py.Group,
                 idx_pde: Union[int, Tuple[int]],
                 keep_all_coef: bool) -> None:
        super().__init__(hdf5_group, idx_pde, keep_all_coef)
        self.bc_types = {}  # Dict[str, List[int]], shape is (4, n_vars)
        self.bc_values = {}  # Dict[str, List[ConstOrField]], shape is (4, n_vars)

        # different edges, [bd_left, bd_right, bd_bottom, bd_top]
        for bd_name in hdf5_group:
            bc_types = []
            bc_values = []
            if bd_name not in ["bd_left", "bd_right", "bd_bottom", "bd_top"]:
                raise ValueError(f"Unexpected bd_name {bd_name}.")
            for var_idx in hdf5_group[bd_name]:
                hdf5_subgroup = hdf5_group[bd_name][var_idx]
                bc_type = hdf5_subgroup["bc_type"][idx_pde]
                if bc_type not in self.SUPPORTED_BC_TYPES:
                    raise ValueError(f"Unexpected bc_type {bc_type}.")
                bc_types.append(bc_type)
                # match the format of ConstOrField
                name_dict = {}
                if "bc_val_type" in hdf5_subgroup:
                    name_dict["coef_type_name"] = "bc_val_type"
                if "bc_val" in hdf5_subgroup:
                    name_dict["field_name"] = "bc_val"
                bc_values.append(ReNamedConstOrField(
                    hdf5_subgroup, idx_pde, keep_all_coef, **name_dict))
            self.bc_types[bd_name] = bc_types
            self.bc_values[bd_name] = bc_values

    def nodes(self,  # pylint: disable=arguments-differ
              pde: PDENodesCollector,
              u_list: List[PDENode],
              domain: PDENode,
              coord_dict: Dict[str, NDArray[float]]) -> None:
        n_vars = len(u_list)
        # collect same type of BCs together
        bv_dict = {}
        for bc_type in self.SUPPORTED_BC_TYPES:
            for var_idx in range(n_vars):
                bv_dict[(bc_type, var_idx)] = []

        for bd_name in ["bd_left", "bd_right", "bd_bottom", "bd_top"]:
            bc_types = self.bc_types[bd_name]
            bc_values = self.bc_values[bd_name]
            if bd_name in ["bd_left", "bd_right"]:  # x-axis
                sdf = coord_dict["x"]
            else:  # y-axis
                sdf = coord_dict["y"]
            if bd_name in ["bd_right", "bd_top"]:  # high side
                sdf = 1 - sdf

            for var_idx, (bc_type, bc_value) in enumerate(
                    zip(bc_types, bc_values)):
                bv_dict[(bc_type, var_idx)].append(ExtendedFunc(
                    sdf, bc_value.field.reshape(sdf.T.shape)))

        for (bc_type, var_idx), ext_func_list in bv_dict.items():
            if not ext_func_list:
                continue
            if bc_type == self.DIRICHLET:
                sum_list = [u_list[var_idx]]
            elif bc_type == self.NEUMANN:
                sum_list = pde.dn_sum_list(u_list[var_idx], domain)
            else:
                raise RuntimeError(f"Unexpected 'bc_type' {bc_type}.")

            sdf, beta = merge_extended_func(ext_func_list)
            self._bc_node(pde, sum_list, sdf, beta, coord_dict)

    def add_latex(self,  # pylint: disable=arguments-differ
                  latex_sum_list: List[LaTeXSum],
                  symbol: str = "u",
                  bc_subscript: Union[str, int] = "") -> List[List[str]]:
        bc_lists = [[] for _ in latex_sum_list]  # nested list, shape is (n_vars, 4)
        namemap = {"bd_left": "L", "bd_right": "R", "bd_bottom": "D",
                   "bd_top": "U"}
        for bd_name in ["bd_left", "bd_right", "bd_bottom", "bd_top"]:
            bc_types = self.bc_types[bd_name]
            bc_values = self.bc_values[bd_name]
            # subscript
            subscript = "{" + str(bc_subscript) + namemap[bd_name] + "}"

            for var_idx, (bc_type, bc_value) in enumerate(
                    zip(bc_types, bc_values)):
                latex_sum = latex_sum_list[var_idx]
                if latex_sum.term_list:  # list non-empty
                    raise RuntimeError("Expected empty 'latex_sum', but it has "
                                       f"terms {latex_sum.term_list}.")
                symbol = "uv"[var_idx]
                if bc_type == self.DIRICHLET:
                    lhs = symbol
                elif bc_type == self.NEUMANN:
                    lhs = rf"\partial_n{symbol}"
                else:
                    raise RuntimeError(f"Unexpected 'bc_type' {bc_type}.")
                bc_lists[var_idx].append(self._bc_latex(
                    latex_sum, bc_value, lhs, subscript))

        return bc_lists

    def _latex_v1(self,
                  latex_sum_list: List[LaTeXSum],
                  symbol: str = "u",
                  bc_subscript: Union[str, int] = "") -> List[List[str]]:
        r"""LaTeX representation with edges merged. Unifinshed."""
        n_vars = len(latex_sum_list)
        # collect same type of BCs together
        bv_dict = {}
        for bc_type in self.SUPPORTED_BC_TYPES:
            for var_idx in range(n_vars):
                bv_dict[(bc_type, var_idx)] = ""

        for bd_name, subscript in zip(
                ["bd_left", "bd_right", "bd_bottom", "bd_top"], "LRDU"):
            bc_types = self.bc_types[bd_name]
            for var_idx, bc_type in enumerate(bc_types):
                bv_dict[(bc_type, var_idx)] += subscript

        for (bc_type, var_idx), subscript in bv_dict.items():
            if not subscript:
                continue
        raise NotImplementedError

    @staticmethod
    def _bc_node(pde: PDENodesCollector,
                 sum_list: List[PDENode],
                 sdf: NDArray[float],
                 beta: NDArray[float],
                 coord_dict: Dict[str, NDArray[float]]) -> None:
        r"""Add DAG node for a specific boundary condition."""
        pde.set_bv(pde.sum(sum_list), sdf, beta, **coord_dict)

    @staticmethod
    def _bc_latex(latex_sum: LaTeXSum,
                  bc_value: ConstOrField,
                  lhs: str,
                  subscript: str) -> str:
        r"""Get LaTeX representation for a specific boundary condition."""
        lhs = lhs + "|_" + subscript
        # RHS
        bc_value.add_latex(latex_sum, rf"\beta_{subscript}")
        rhs = latex_sum.strip_sum()
        if rhs == "":
            rhs = "0"
        return lhs + "=" + rhs


class EdgeBCWithMur(EdgeBoundaryCondition):
    r"""
    Non-periodic boundary condition of a wave equation, applied to one single
    edge of the square domain, taking the general form $Bu(r)=\beta(r)$ for
    $r\in\Gamma_i$.

    The linear boundary operator $Bu$ is taken to be Robin with (D) type
    $Bu=u+\alpha(r)\partial u/\partial n$, Robin with (N) type
    $Bu=\alpha(r)u+\partial u/\partial n$, and (generalized) Mur type
    $Bu=u_t+\alpha(r)u+\gamma(r)\partial u/\partial n$ with equal probability.
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
        r"""
        Return the string representation of the BC type.
        Possible returned values: D, N, R1, RDc, RNc, RDf, RNf, Mc, Mf.
        """
        if self.bc_type != self.MUR_R:
            return super().full_type
        if (self.alpha.coef_type == self.alpha.FIELD_COEF
                and self.gamma.coef_type == self.gamma.FIELD_COEF):
            return "Mf"
        return "Mc"

    def nodes(self,
              pde: PDENodesCollector,
              u_node: PDENode = None,
              domain: PDENode = None,
              coord_dict: Dict[str, NDArray[float]] = None) -> None:
        if self.bc_type != self.MUR_R:
            super().nodes(pde, u_node, domain, coord_dict)
            return
        if None in [u_node, domain, coord_dict]:
            raise ValueError("Arguments 'u_node', 'domain', and 'coord_dict' "
                             "must be specified.")

        # sum_list
        alpha = self.alpha.nodes(pde, coord_dict)
        gamma = self.gamma.nodes(pde, coord_dict)
        sum_list = [u_node.dt, alpha * u_node] + pde.dn_sum_list(
            u_node, domain, coef=gamma)

        # final BV node
        sdf, beta = self.get_ext_field()
        pde.set_bv(pde.sum(sum_list), sdf, beta, **coord_dict)

    def add_latex(self,
                  latex_sum: LaTeXSum,
                  symbol: str = "u",
                  bc_subscript: Union[str, int] = "") -> str:
        if self.bc_type != self.MUR_R:
            return super().add_latex(latex_sum, symbol, bc_subscript)

        if latex_sum.term_list:  # list non-empty
            raise RuntimeError("Expected empty 'latex_sum', but it has terms "
                               f"{latex_sum.term_list}.")
        subscript = self.edge_subscript
        if bc_subscript != "":
            subscript = "{" + str(bc_subscript) + subscript + "}"

        # LHS
        latex_sum.add_term(r"\partial_t" + symbol)
        self.alpha.add_latex(latex_sum, rf"\alpha_{subscript}", symbol)
        self.gamma.add_latex(
            latex_sum, rf"\gamma_{subscript}", r"\partial_n" + symbol)
        lhs = latex_sum.strip_sum("(", rf")|_{subscript}")

        # RHS
        self.beta.add_latex(latex_sum, rf"\beta_{subscript}")
        rhs = latex_sum.strip_sum()  # get the beta term
        if rhs == "":
            rhs = "0"
        return lhs + "=" + rhs


class BoxDomainBCWithMur(BoxDomainBoundaryCondition):
    r"""
    Non-periodic boundary condition of a PDE, including all edges (surfaces) of
    a square (cubic) domain, unless the domain is periodic along this axis.

    For each edge or surface, the boundary condition takes the general form
    $Bu(r)=\beta(r)$ for $r\in\Gamma_i$.
    The linear boundary operator $Bu$ is taken to be Robin with (D) type
    $Bu=u+\alpha(r)\partial u/\partial n$, Robin with (N) type
    $Bu=\alpha(r)u+\partial u/\partial n$, and (generalized) Mur type
    $Bu=u_t+\alpha(r)u+\gamma(r)\partial u/\partial n$ with equal probability.

    The coefficient fields $\alpha(r),\gamma(r)$ as well as the term $\beta(r)$
    are taken to be zero, one, a random scalar, or a random field with certain
    probability. We may also set $\beta(r)$ to meet the initial condition.
    """
    edge_bc_cls: type = EdgeBCWithMur

    def nodes(self,
              pde: PDENodesCollector,
              u_node: PDENode = None,
              domain: PDENode = None,
              coord_dict: Dict[str, NDArray[float]] = None,
              *,
              ignored_types: Tuple[str] = ()) -> None:
        super().nodes(pde, u_node, domain, coord_dict,
                      ignored_types=("Mf",) + ignored_types)
        if "Mf" not in self.edges_by_type_dict:
            return
        edge_list = self.edges_by_type_dict["Mf"]
        if not edge_list:  # list empty
            return

        # coefs
        _, alpha = merge_extended_func([
            edge_obj.get_ext_field("alpha")
            for edge_obj in edge_list])  # NDArray[float]
        alpha = pde.new_coef_field(alpha, **coord_dict)
        _, gamma = merge_extended_func([
            edge_obj.get_ext_field("gamma")
            for edge_obj in edge_list])  # NDArray[float]
        gamma = pde.new_coef_field(gamma, **coord_dict)

        # final BV node
        sum_list = [u_node.dt, alpha * u_node] + pde.dn_sum_list(
            u_node, domain, coef=gamma)
        sdf, beta = merge_extended_func([
            edge_obj.get_ext_field() for edge_obj in edge_list])
        pde.set_bv(pde.sum(sum_list), sdf, beta, **coord_dict)

    def add_latex(self,
                  latex_sum: LaTeXSum,
                  symbol: str = "u",
                  bc_subscript: Union[str, int] = "",
                  *,
                  ignored_types: Tuple[str] = ()) -> List[str]:
        bc_list = super().add_latex(latex_sum, symbol, bc_subscript,
                                    ignored_types=("Mf",) + ignored_types)
        if "Mf" not in self.edges_by_type_dict:
            return bc_list
        edge_list = self.edges_by_type_dict["Mf"]
        if not edge_list:  # list empty
            return bc_list

        # Adding "Mf" edge type
        subscript = "{" + str(bc_subscript)
        for edge_obj in edge_list:
            subscript += edge_obj.edge_subscript
        subscript += "}"
        bc_list.append(
            rf"(\partial_t{symbol}+\alpha_{subscript}(r){symbol}"
            rf"+\gamma_{subscript}(r)\partial_n{symbol})"
            rf"|_{subscript}=\beta_{subscript}(r)")
        return bc_list

    @classmethod
    def _merge_scalar_bc_(cls,
                          edges_dict: Dict[str, List[EdgeBoundaryCondition]],
                          n_bc_thres: int) -> None:
        rdc_list = edges_dict.pop("RDc", [])
        rnc_list = edges_dict.pop("RNc", [])
        mc_list = edges_dict.pop("Mc", [])

        # non-merging case
        sum_len = len(edges_dict) + len(rdc_list) + len(rnc_list) + len(mc_list)
        if sum_len <= n_bc_thres:
            # allow empty list
            edges_dict["Xc"] = rdc_list + rnc_list + mc_list
            return

        # merging case
        edges_dict["Xc"] = []
        cls._merge_c_into_f_(edges_dict, rdc_list, "RDf")
        cls._merge_c_into_f_(edges_dict, rnc_list, "RNf")
        cls._merge_c_into_f_(edges_dict, mc_list, "Mf")


class DiskDomain(PDETermBase):
    r"""A random disk-shaped computational domain."""
    min_diameter: float
    radius: float
    center: NDArray[float]

    def __init__(self,
                 hdf5_group: h5py.Group,
                 idx_pde: Union[int, Tuple[int]],
                 keep_all_coef: bool) -> None:
        super().__init__(hdf5_group, idx_pde, keep_all_coef)
        self.radius = hdf5_group["radius"][idx_pde]
        self.center = hdf5_group["center"][idx_pde]

    def nodes(self,  # pylint: disable=arguments-differ
              pde: PDENodesCollector,
              coord_dict: Dict[str, NDArray[float]] = None) -> Tuple:
        if coord_dict is None:
            raise TypeError("Arguments 'coord_dict' must be given.")
        x_ext = coord_dict["x"]
        y_ext = coord_dict["y"]
        center = self.center
        center_distance = np.sqrt(
            (x_ext - center[0])**2 + (y_ext - center[1])**2)
        sdf = center_distance - self.radius
        domain = pde.new_domain(sdf, x=x_ext, y=y_ext)
        sdf_dict = {"outer": sdf}
        return domain, sdf_dict  # PDENode, Dict[str, NDArray[float]]

    def add_latex(self, latex_sum: LaTeXSum, symbol: str) -> None:
        pass  # nothing to do
