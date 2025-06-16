r"""
Generate DAG as well as LaTeX representations for the PDE types involved in the
custom multi_pde dataset.
"""
import os
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Callable, Union, Tuple, List, Dict
from functools import reduce

import numpy as np
import h5py
from omegaconf import DictConfig, ListConfig

from ..env import float_dtype, int_dtype, DAG_BC_VERSION
from ..pde_dag import PDENodesCollector
from . import terms, boundary, boundary_v2


DAG_INFO_DIR = "dag_info_v5"
DYN_DSET_COMM_DIR = "dyn_dset_comm"
FileListType = Union[str, List[str], DictConfig]
pde_info_cls_dict = {}


def record_pde_info(*pde_type_all):
    r"""Register the current class with a specific name."""
    def add_class(cls):
        for pde_type in pde_type_all:
            pde_info_cls_dict[pde_type] = cls
        return cls
    return add_class


def get_pde_info_cls(pde_type: str) -> type:
    r"""Get the class to provide information of a specific PDE type."""
    pde_type = pde_type.split("_")[0]
    return pde_info_cls_dict[pde_type]


class PDEInfoBase(ABC):
    r"""Basic terms involved in the custom multi_pde dataset."""

    @staticmethod
    @abstractmethod
    def var_latex(idx_var: int) -> str:
        r"""
        Get the name of the variable indexed by `idx_var`.
        """

    @classmethod
    def dag_file_suffix(cls,
                        config: DictConfig,
                        file_ext: str = ".hdf5") -> str:
        r"""
        The custom multi_pde dataset requires preprocessing, in which we construct
        the DAG representation of each PDE in the dataset, and save the graph data
        into a separate HDF5 file. This function returns the suffix of this
        file containing DAG information.
        """
        if config.model.multi_inr.enable and config.model.multi_inr.separate_latent:
            uf_num_mod = config.model.inr.num_layers + config.model.inr2.num_layers - 2
        else:
            uf_num_mod = config.model.inr.num_layers - 1
        max_n_scalar_nodes = config.data.pde_dag.max_n_scalar_nodes
        max_n_function_nodes = config.data.pde_dag.max_n_function_nodes
        function_num_branches = config.model.function_encoder.num_branches
        suffix = (f"_inr{uf_num_mod}sc{max_n_scalar_nodes}"
                  f"func{max_n_function_nodes}x{function_num_branches}")
        suffix += file_ext
        # return "_ModSeq" + suffix  # ModSeq
        return suffix

    @classmethod
    def dag_info_file_path(cls,
                           config: DictConfig,
                           filename: str,
                           file_ext: str = ".hdf5") -> str:
        r"""
        The custom multi_pde dataset requires preprocessing, in which we construct
        the DAG representation of each PDE in the dataset, and save the graph data
        into a separate HDF5 file. This function returns the path of this file
        containing DAG information.
        """
        suffix = cls.dag_file_suffix(config, file_ext=file_ext)
        return os.path.join(config.data.path, DAG_INFO_DIR, filename + suffix)

    @classmethod
    @abstractmethod
    def pde_nodes(cls,
                  h5file_in: h5py.File,
                  idx_pde: Union[int, Tuple[int]],
                  keep_all_coef: bool) -> PDENodesCollector:
        r"""Generate DAG nodes for the PDE."""

    @classmethod
    @abstractmethod
    def pde_latex(cls,
                  h5file_in: h5py.File,
                  idx_pde: Union[int, Tuple[int]],
                  keep_all_coef: bool) -> Tuple:
        r"""
        Generate the mathematical representation as well as a dictionary of
        coefficient values for the current PDE.

        Returns:
            pde_latex (str): LaTeX expression of the PDE.
            coef_dict (OrderedDict): Dictionary of the coefficient values.
        """

    @classmethod
    def gen_dag_info(cls,
                     filename: str,
                     config: DictConfig,
                     print_fn: Callable[[str], None] = print) -> None:
        r"""
        Precompute Graphormer input (PDE DAG info) for a single data file, and
        save the results to a HDF5 file.
        """
        # target DAG data file
        dag_filepath = cls.dag_info_file_path(config, filename)
        if os.path.exists(dag_filepath):
            return  # no need to (re)generate DAG file
        print_fn("generating " + dag_filepath)

        n_scalar_nodes = config.data.pde_dag.max_n_scalar_nodes
        n_function_nodes = config.data.pde_dag.max_n_function_nodes
        function_num_branches = config.model.function_encoder.num_branches
        n_node = n_scalar_nodes + n_function_nodes * function_num_branches

        # h5file_in
        u_filepath = os.path.join(config.data.path, filename + ".hdf5")
        h5file_in = h5py.File(u_filepath, "r")

        # data to be saved
        n_pde = h5file_in["args/num_pde"][()]
        # n_x = np.size(h5file_in["coord/x"])
        # n_y = np.size(h5file_in["coord/y"])
        node_type = np.zeros([n_pde, n_node, 1], dtype=int_dtype)
        node_scalar = np.zeros([n_pde, n_scalar_nodes, 1], dtype=float_dtype)
        node_func_t = np.zeros([n_pde, n_function_nodes, 1, 1],
                               dtype=float_dtype)
        # node_func_f = np.zeros([n_pde, n_function_nodes, n_x * n_y, 1],
        node_func_f = np.zeros([n_pde, n_function_nodes, 128**2, 1],
                               dtype=float_dtype)
        spatial_pos = np.zeros([n_pde, n_node, n_node], dtype=np.uint8)
        in_degree = np.zeros([n_pde, n_node], dtype=int_dtype)
        out_degree = np.zeros([n_pde, n_node], dtype=int_dtype)

        for idx_pde in range(n_pde):
            pde = cls.pde_nodes(h5file_in, idx_pde, keep_all_coef=False)
            pde_dag = pde.gen_dag(config)

            node_type[idx_pde] = pde_dag.node_type
            node_scalar[idx_pde] = pde_dag.node_scalar
            node_func_t[idx_pde] = pde_dag.node_function[:, :1, :1]
            node_func_f[idx_pde] = pde_dag.node_function[:, :, -1:]
            spatial_pos[idx_pde] = pde_dag.spatial_pos
            in_degree[idx_pde] = pde_dag.in_degree
            out_degree[idx_pde] = pde_dag.out_degree

        h5file_in.close()
        with h5py.File(dag_filepath, "w") as h5file_out:
            h5file_out.create_dataset("node_type", data=node_type)
            h5file_out.create_dataset("node_scalar", data=node_scalar)
            h5file_out.create_dataset("node_func_t", data=node_func_t)
            h5file_out.create_dataset("node_func_f", data=node_func_f)
            h5file_out.create_dataset("spatial_pos", data=spatial_pos)
            h5file_out.create_dataset("in_degree", data=in_degree)
            h5file_out.create_dataset("out_degree", data=out_degree)


def _term_cls_to_obj(term_cls_dict: Dict[str, type],
                     h5file_in: h5py.File,
                     idx_pde: Union[int, Tuple[int]],
                     keep_all_coef: bool) -> Dict[str, terms.PDETermBase]:
    return {key: term_cls(h5file_in["coef"][key], idx_pde, keep_all_coef)
            for key, term_cls in term_cls_dict.items()}


@record_pde_info("diffConvecReac", "dcr")
class DiffConvecReac2DInfo(PDEInfoBase):
    r"""
    ======== Diffusion-Convection-Reaction Equation ========
    The PDE takes the form $u_t+Lu+f_0(u)+s(r)+f_1(u)_x+f_2(u)_y=0$,
    $u(0,r)=g(r)$, $t\in[0,1]$, $r=(x,y)\in[0,1]^2$.
    On edge $\Gamma_i$, the boundary condition imposed is either periodic or of
    the general form $B_iu(r)=0$ for $r\in\Gamma_i$.

    Here, the spatial second-order term $Lu$ is randomly selected from
    the non-divergence form $Lu=-a(r)\Delta u$, the factored form
    $Lu=-\sqrt a(r)\nabla\cdot(\sqrt a(r)\nabla u)$, and the divergence form
    $Lu=-\nabla\cdot(a(r)\nabla u)$ with equal probability, where $a(r)$ is
    taken to be a random scalar or a random field, and $r=(x,y,z)$ denotes the
    spatial coordinates.
    We take $f_i(u) = \sum_{k=1}^3c_{i0k}u^k
                     + \sum_{j=1}^{J_i}c_{ij0}h_{ij}(c_{ij1}u+c_{ij2}u^2)$
    for $i=0,1,2$, where $J_0+J_1+J_2\le J$ are randomly generated.

    Each boundary operator $B_iu$ is taken to be Robin with D-type
    $B_iu = u + b_i(r)\partial u/\partial n + c_i(r)$, or Robin with N-type
    $B_iu = a_i(r)u + \partial u/\partial n + c_i(r)$, with equal probability.
    Each of the coefficient field $a_i(r),b_i(r),c_i(r)$ is taken to be zero,
    one, a random scalar, or a random field with certain probability. Note that
    when $a_i(r)$ or $b_i(r)$ equals zero, the boundary condition would
    degenerate to the Dirichlet type or the Neumann type. We may also set
    $c_i(r)$ to meet the initial condition.
    """
    IS_WAVE: bool = False
    LG_KAPPA: bool = False
    N_BC_PER_LINE = 2

    @staticmethod
    def var_latex(idx_var: int) -> str:
        return "u"

    @classmethod
    def dag_file_suffix(cls,
                        config: DictConfig,
                        file_ext: str = ".hdf5") -> str:
        bc_str = "_bc2" if DAG_BC_VERSION == 2 else ""
        return bc_str + super().dag_file_suffix(config, file_ext=file_ext)

    @classmethod
    def pde_nodes(cls,
                  h5file_in: h5py.File,
                  idx_pde: Union[int, Tuple[int]],
                  keep_all_coef: bool) -> PDENodesCollector:
        # x_ext = h5file_in["coord/x"][:]
        # y_ext = h5file_in["coord/y"][:]
        # coef values always given on regular grid points
        x_ext = np.linspace(0, 1, 129)[:-1].reshape(-1, 1)
        y_ext = np.linspace(0, 1, 129)[:-1].reshape(1, -1)
        coord_dict = {"x": x_ext, "y": y_ext}
        pde = PDENodesCollector(dim=2)
        term_obj_dict = cls._get_term_dict(h5file_in, idx_pde, keep_all_coef)

        # Domain and unknown variable
        periodic = h5file_in.get("args/periodic", True)
        if "domain" in term_obj_dict:
            domain, sdf_dict = term_obj_dict["domain"].nodes(pde, coord_dict)
            term_obj_dict["bc"].assign_sdf(sdf_dict)
        elif np.all(periodic):
            domain = None
        else:
            sdf_dict = {}
            for i_ax, ax_name in enumerate("xy"):
                if periodic[i_ax]:
                    continue
                sdf_dict[ax_name + "_low"] = -coord_dict[ax_name]
                sdf_dict[ax_name + "_high"] = coord_dict[ax_name] - 1
            domain = pde.new_domain(
                reduce(np.maximum, sdf_dict.values()), x=x_ext, y=y_ext)
            term_obj_dict["bc"].assign_sdf(sdf_dict)
        u_node = pde.new_uf(domain)

        # Initial condition
        pde.set_ic(u_node, h5file_in["coef/u_ic"][idx_pde], x=x_ext, y=y_ext)
        if cls.IS_WAVE:
            pde.set_ic(u_node.dt, h5file_in["coef/ut_ic"][idx_pde], x=x_ext, y=y_ext)

        # PDE Terms
        if h5file_in["args/inhom_diff_u"][()]:
            sum_list = [term_obj_dict["Lu"].nodes(pde, u_node, coord_dict)]
        else:
            sum_list = [term_obj_dict["Lu"].nodes(pde, u_node)]

        sum_list.extend(term_obj_dict["f0"].nodes(pde, u_node))
        sum_list.append(pde.dx(pde.sum(term_obj_dict["f1"].nodes(pde, u_node))))
        sum_list.append(pde.dy(pde.sum(term_obj_dict["f2"].nodes(pde, u_node))))
        sum_list.append(term_obj_dict["s"].nodes(pde, coord_dict))

        if cls.IS_WAVE:
            sum_list.append(term_obj_dict["mu"].nodes(pde, coord_dict) * u_node.dt)
            sum_list.append(u_node.dt.dt)
        else:
            sum_list.append(u_node.dt)
        pde.sum_eq0(sum_list)

        # Boundary Condition
        if "bc" in term_obj_dict:
            term_obj_dict["bc"].nodes(pde, u_node, domain, coord_dict)
        return pde

    @classmethod
    def pde_latex(cls,
                  h5file_in: h5py.File,
                  idx_pde: Union[int, Tuple[int]],
                  keep_all_coef: bool) -> Tuple:
        coef_dict = OrderedDict()
        latex_sum = terms.LaTeXSum(keep_all_coef, coef_dict)
        term_obj_dict = cls._get_term_dict(h5file_in, idx_pde, keep_all_coef)

        if cls.IS_WAVE:
            latex_sum.add_term("u_{tt}")
            term_obj_dict["mu"].add_latex(latex_sum, r"\mu", "u_t")
        else:
            latex_sum.add_term("u_t")

        # PDE Terms
        term_obj_dict["Lu"].add_latex(latex_sum)
        term_obj_dict["s"].add_latex(latex_sum, "s")

        # Nonlinear Terms
        term_obj_dict["f0"].add_latex(latex_sum, i=0)
        fi_sum = terms.LaTeXSum(keep_all_coef, coef_dict)
        term_obj_dict["f1"].add_latex(fi_sum, i=1)
        latex_sum.add_term(fi_sum.strip_sum("(", ")_x"))
        term_obj_dict["f2"].add_latex(fi_sum, i=2)
        latex_sum.add_term(fi_sum.strip_sum("(", ")_y"))
        # latex_sum.merge_coefs(fi_sum)

        pde_latex = latex_sum.strip_sum("", "=0")
        # Boundary Condition
        if "bc" in term_obj_dict:
            bc_list = term_obj_dict["bc"].add_latex(latex_sum)
            eqn_list = [pde_latex]
            for i in range(0, len(bc_list), cls.N_BC_PER_LINE):
                eqn_list.append(r",\ ".join(bc_list[i:i+cls.N_BC_PER_LINE]))
            pde_latex = "$\n$".join(eqn_list)
        pde_latex = "$" + pde_latex + "$"
        return pde_latex, coef_dict

    @classmethod
    def _get_term_dict(cls,
                       h5file_in: h5py.File,
                       idx_pde: Union[int, Tuple[int]],
                       keep_all_coef: bool) -> Dict[str, terms.PDETermBase]:
        r"""Get dictionary of term objectives in a PDE."""
        term_cls_dict = {"f0": terms.PolySinusNonlinTerm,
                         "f1": terms.PolySinusNonlinTerm,
                         "f2": terms.PolySinusNonlinTerm,
                         "s": terms.ConstOrField}
        if cls.IS_WAVE:
            term_cls_dict["mu"] = terms.ConstOrField

        # spatial 2nd-order term
        if h5file_in["args/inhom_diff_u"][()]:
            if cls.LG_KAPPA:
                term_cls_dict["Lu"] = terms.LgCoefInhomSpatialOrder2Term
            else:
                term_cls_dict["Lu"] = terms.InhomSpatialOrder2Term
        elif cls.LG_KAPPA:
            term_cls_dict["Lu"] = terms.LgCoefHomSpatialOrder2Term
        else:
            term_cls_dict["Lu"] = terms.HomSpatialOrder2Term

        # boundary conditions
        if "bc" not in h5file_in["coef"]:
            pass
        elif DAG_BC_VERSION == 1:
            if cls.IS_WAVE:
                term_cls_dict["bc"] = boundary.BoxDomainBCWithMur
            else:
                term_cls_dict["bc"] = boundary.BoxDomainBoundaryCondition
        elif DAG_BC_VERSION == 2:
            if cls.IS_WAVE:
                term_cls_dict["bc"] = boundary_v2.FullBCsWithMur
            else:
                term_cls_dict["bc"] = boundary_v2.FullBoundaryConditions
        else:
            raise NotImplementedError

        if "domain" in h5file_in["coef"]:
            term_cls_dict["domain"] = boundary.DiskDomain

        return _term_cls_to_obj(term_cls_dict, h5file_in, idx_pde, keep_all_coef)

    @classmethod
    def _unit_test(cls, config: DictConfig, keep_all_coef: bool = False):
        r"""unit test"""
        field_sim = np.random.random((1, 2, 3))
        coef_sim = {"f0": np.array([[[0, 0, 0, 0]]]),
                    "f1": np.array([[[0, 0, 0, 0]]]),
                    "f2": np.array([[[0, 0, 0, 0]]]),
                    "Lu": {"diff_type": [0], "value": [0.5]},
                    "s": {"coef_type": [0], "field": field_sim},
                    "mu": {"coef_type": [0], "field": field_sim}}
        h5file_sim = {"coef": coef_sim,
                      "coef/u_ic": field_sim,
                      "coef/ut_ic": field_sim,
                      "args/inhom_diff_u": False,
                      "coord/x": np.random.random((2, 1)),
                      "coord/y": np.random.random((1, 3))}
        print(cls.pde_latex(h5file_sim, 0, keep_all_coef))
        pde = cls.pde_nodes(h5file_sim, 0, keep_all_coef)
        pde_dag = pde.gen_dag(config)
        pde_dag.plot()


@record_pde_info("dcrLgK")
class DCRLgKappa2DInfo(DiffConvecReac2DInfo):
    r"""
    Same as standard diffusion-convection-reaction equation, but with the
    coefficients written in the $10^a$ form.
    """
    LG_KAPPA: bool = True

    @classmethod
    def dag_file_suffix(cls,
                        config: DictConfig,
                        file_ext: str = ".hdf5") -> str:
        return "_lgK" + super().dag_file_suffix(config, file_ext=file_ext)


@record_pde_info("wave")
class Wave2DInfo(DiffConvecReac2DInfo):
    r"""
    ======== Wave Equation ========
    The PDE takes the form
        $$u_{tt}+\mu(r)u_t+Lu+f_0(u)+s(r)+f_1(u)_x+f_2(u)_y=0,$$
    $u(0,r)=g(r)$, $u_t(0,r)=h(r)$, $t\in[0,1]$, $r=(x,y)\in[0,1]^2$.
    On edge $\Gamma_i$, the boundary condition imposed is either periodic or of
    the general form $B_iu(r)=0$ for $r\in\Gamma_i$.

    Here, the spatial second-order term $Lu$ is randomly selected from
    the non-divergence form $Lu=-a(r)\Delta u$, the factored form
    $Lu=-\sqrt a(r)\nabla\cdot(\sqrt a(r)\nabla u)$, and the divergence form
    $Lu=-\nabla\cdot(a(r)\nabla u)$ with equal probability, where $a(r)$ is
    taken to be a random scalar or a random field, and $r=(x,y,z)$ denotes the
    spatial coordinates.
    We take $f_i(u) = \sum_{k=1}^3c_{i0k}u^k
                     + \sum_{j=1}^{J_i}c_{ij0}h_{ij}(c_{ij1}u+c_{ij2}u^2)$
    for $i=0,1,2$, where $J_0+J_1+J_2\le J$ are randomly generated.

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
    IS_WAVE: bool = True


@record_pde_info("mcompn", "multiComponent", "mcdcr", "mvdcr")
class MultiComponent2DInfo(PDEInfoBase):
    r"""
    ======== PDE with Multiple Components ========
    The PDE takes the form
        $$\partial_tu_i + L_iu_i + f_0(u)_i + s_i(r) + \partial_xf_1(u)_i
            + \partial_yf_2(u)_i = 0,$$
    $u_i(0,r)=g_i(r)$, $t\in[0,1]$, $r=(x,y)\in[0,1]^2$,
    $0 \le i,j,k \le d_u-1$, $j \le k$.
    Periodic boundary conditions are employed for simplicity.

    Here, each spatial second-order term $L_iu_i$ is randomly selected from
    the non-divergence form $L_iu_i=-a_i(r)\Delta u_i$, the factored form
    $L_iu_i=-\sqrt a_i(r)\nabla\cdot(\sqrt a_i(r)\nabla u_i)$, and the
    divergence form $L_iu_i=-\nabla\cdot(a_i(r)\nabla u_i)$ with equal
    probability, where $a_i(r)=a_i$ is taken to be a random scalar for
    simplicity, and $r=(x,y)$ denotes the spatial coordinates.

    We take $f_l(u)_i = \sum_ja_{lij}u_j + \sum_{j,k}b_{lijk}u_ju_k$.
    The coefficients $a,b$ are sparse arrays, with a total of at most $3d$
    non-zero entries.
    """
    IS_WAVE: bool = False
    LG_KAPPA: bool = False
    DIV_CONSTRAINT: bool = False

    @staticmethod
    def var_latex(idx_var: int) -> str:
        return f"u_{idx_var}"

    @classmethod
    def pde_nodes(cls,
                  h5file_in: h5py.File,
                  idx_pde: Union[int, Tuple[int]],
                  keep_all_coef: bool) -> PDENodesCollector:
        n_vars = h5file_in["args/n_vars"][()]
        pde = PDENodesCollector()
        u_list = [pde.new_uf() for _ in range(n_vars)]

        # Basic info
        x_ext = h5file_in["coord/x"]
        y_ext = h5file_in["coord/y"]
        coord_dict = {"x": x_ext, "y": y_ext}
        term_obj_dict = cls._get_term_dict(h5file_in, idx_pde, keep_all_coef)

        # PDE Terms
        if cls.DIV_CONSTRAINT:
            p_node = pde.new_uf()
            c_values = term_obj_dict["c"].value  # ignoring keep_all_coef
            pde.sum_eq0(u_list[0].dx,
                        u_list[1].dy,
                        c_values[0] * u_list[0],
                        c_values[1] * u_list[1],
                        c_values[2])
        f0_lists = term_obj_dict["f0"].nodes(pde, u_list)
        f1_lists = term_obj_dict["f1"].nodes(pde, u_list)
        f2_lists = term_obj_dict["f2"].nodes(pde, u_list)
        for i in range(n_vars):
            # Initial condition
            pde.set_ic(u_list[i], h5file_in[f"coef/u_ic/{i}"][idx_pde],
                       x=x_ext, y=y_ext)
            if cls.IS_WAVE:
                pde.set_ic(u_list[i].dt, h5file_in[f"coef/ut_ic/{i}"][idx_pde],
                           x=x_ext, y=y_ext)

            sum_i_list = f0_lists[i]
            sum_i_list.append(pde.dx(pde.sum(f1_lists[i])))
            sum_i_list.append(pde.dy(pde.sum(f2_lists[i])))
            sum_i_list.append(term_obj_dict[f"s/{i}"].nodes(pde, coord_dict))

            # Second-order term
            if h5file_in["args/inhom_diff_u"][()]:
                sum_i_list.append(term_obj_dict[f"Lu/{i}"].nodes(
                    pde, u_list[i], coord_dict))
            else:
                sum_i_list.append(term_obj_dict[f"Lu/{i}"].nodes(pde, u_list[i]))

            # pressure terms for divergence constraints
            if cls.DIV_CONSTRAINT:
                sum_i_list.append((-c_values[i]) * p_node)
                if i == 0:
                    sum_i_list.append(p_node.dx)
                elif i == 1:
                    sum_i_list.append(p_node.dy)
                elif i == 2:
                    sum_i_list.append(p_node.dz)
                    raise RuntimeError

            if cls.IS_WAVE:
                sum_i_list.append(
                    term_obj_dict[f"mu/{i}"].nodes(pde, coord_dict)
                    * u_list[i].dt)
                sum_i_list.append(u_list[i].dt.dt)
            else:
                sum_i_list.append(u_list[i].dt)
            pde.sum_eq0(sum_i_list)

        return pde

    @classmethod
    def pde_latex(cls,
                  h5file_in: h5py.File,
                  idx_pde: Union[int, Tuple[int]],
                  keep_all_coef: bool) -> Tuple:
        coef_dict = OrderedDict()
        term_obj_dict = cls._get_term_dict(h5file_in, idx_pde, keep_all_coef)
        n_vars = h5file_in["args/n_vars"][()]
        latex_sum_list = [terms.LaTeXSum(keep_all_coef, coef_dict)
                          for _ in range(n_vars)]

        # PDE Terms
        if cls.IS_WAVE:
            for i in range(n_vars):
                latex_sum_list[i].add_term(rf"\partial_{{tt}}u_{i}")
                term_obj_dict[f"mu/{i}"].add_latex(
                    latex_sum_list[i], rf"\mu_{i}", rf"\partial_tu_{i}")
        else:
            for i in range(n_vars):
                latex_sum_list[i].add_term(rf"\partial_tu_{i}")

        term_obj_dict["f0"].add_latex(latex_sum_list, l=0)
        fi_sum_list = [terms.LaTeXSum(keep_all_coef, coef_dict)
                       for _ in range(n_vars)]
        term_obj_dict["f1"].add_latex(fi_sum_list, l=1)
        for i in range(n_vars):
            latex_sum_list[i].add_term(
                fi_sum_list[i].strip_sum(r"\partial_x(", ")"))
        term_obj_dict["f2"].add_latex(fi_sum_list, l=2)
        for i in range(n_vars):
            latex_sum_list[i].add_term(
                fi_sum_list[i].strip_sum(r"\partial_y(", ")"))
            term_obj_dict[f"s/{i}"].add_latex(latex_sum_list[i], f"s_{i}")
            term_obj_dict[f"Lu/{i}"].add_latex(
                latex_sum_list[i], f"a_{i}", f"u_{i}")
            if cls.DIV_CONSTRAINT:
                latex_sum_list[i].add_term(r"\partial_" + "xyzw"[i] + "p")
                latex_sum_list[i].add_term_with_coef(
                    -term_obj_dict["c"].value[i], f"(-c_{i})", "p")

        if cls.DIV_CONSTRAINT:
            latex_sum_div = terms.LaTeXSum(keep_all_coef, coef_dict)
            latex_sum_div.add_term(r"\partial_xu_0")
            latex_sum_div.add_term(r"\partial_yu_1")
            term_obj_dict["c"].add_latex(latex_sum_div, "c_0", "u_0", 0)
            term_obj_dict["c"].add_latex(latex_sum_div, "c_1", "u_1", 1)
            term_obj_dict["c"].add_latex(latex_sum_div, "c_2", "", 2)
            latex_sum_list.append(latex_sum_div)

        pde_latex_list = [latex_sum.strip_sum("$", "=0$")
                          for latex_sum in latex_sum_list]
        pde_latex = "\n".join(pde_latex_list)
        return pde_latex, coef_dict

    @classmethod
    def _get_term_dict(cls,
                       h5file_in: h5py.File,
                       idx_pde: Union[int, Tuple[int]],
                       keep_all_coef: bool) -> Dict[str, terms.PDETermBase]:
        r"""Get dictionary of term objectives in a PDE."""
        term_cls_dict = {"f0": terms.MultiComponentMixedTerm,
                         "f1": terms.MultiComponentMixedTerm,
                         "f2": terms.MultiComponentMixedTerm}
        n_vars = h5file_in["args/n_vars"][()]
        if not np.all(h5file_in.get("args/periodic", True)):
            raise NotImplementedError("Dataset 'MCompn2D' only supports "
                                      "periodic BC.")
        if cls.DIV_CONSTRAINT:
            term_cls_dict["c"] = terms.CoefArray
        for i in range(n_vars):
            term_cls_dict[f"s/{i}"] = terms.ConstOrField
            if cls.IS_WAVE:
                term_cls_dict[f"mu/{i}"] = terms.ConstOrField

            # spatial 2nd-order term
            if h5file_in["args/inhom_diff_u"][()]:
                raise NotImplementedError(
                    "Dataset 'MCompn2D' only supports homogeneous diffusion "
                    "coefficients.")  # change below to 'elif' after finished
            if cls.LG_KAPPA:
                term_cls_dict[f"Lu/{i}"] = terms.LgCoefHomSpatialOrder2Term
            else:
                term_cls_dict[f"Lu/{i}"] = terms.HomSpatialOrder2Term

            # boundary conditions
            if "bc" not in h5file_in["coef"]:
                pass
            elif cls.IS_WAVE:
                term_cls_dict[f"bc/{i}"] = boundary.BoxDomainBCWithMur
            else:
                term_cls_dict[f"bc/{i}"] = boundary.BoxDomainBoundaryCondition

        return _term_cls_to_obj(term_cls_dict, h5file_in, idx_pde, keep_all_coef)

    @classmethod
    def _unit_test(cls, config: DictConfig, keep_all_coef: bool = False):
        r"""unit test"""
        field_sim = np.random.random((1, 2, 3))
        coo_zero = {"coo_len": np.array([0]),
                    "coo_i": np.array([[0]]),
                    "coo_j": np.array([[0]]),
                    "coo_k": np.array([[0]]),
                    "coo_vals": np.array([[1.]]),
                    }
        coo_test = {"coo_len": np.array([2]),
                    "coo_i": np.array([[0, 1]]),
                    "coo_j": np.array([[1, 1]]),
                    "coo_k": np.array([[1, 1]]),
                    "coo_vals": np.array([[1., -1]]),
                    }
        fi_zero = {"lin": coo_zero, "deg2": coo_zero}
        fi_test = {"lin": coo_zero, "deg2": coo_test}
        coef_sim = {"f0": fi_zero, "f1": fi_zero, "f2": fi_test,
                    "Lu/0": {"diff_type": [0], "value": [0.]},
                    "Lu/1": {"diff_type": [1], "value": [0.]},
                    "s/0": {"coef_type": [0], "field": field_sim},
                    "s/1": {"coef_type": [0], "field": field_sim}}
        h5file_sim = {"coef": coef_sim,
                      "coef/u_ic/0": field_sim,
                      "coef/u_ic/1": field_sim,
                      "args/inhom_diff_u": False,
                      "args/n_vars": 2,
                      "coord/x": np.random.random((2, 1)),
                      "coord/y": np.random.random((1, 3))}
        print(cls.pde_latex(h5file_sim, 0, keep_all_coef))
        pde = cls.pde_nodes(h5file_sim, 0, keep_all_coef)
        pde_dag = pde.gen_dag(config)
        pde_dag.plot()


@record_pde_info("mcLgK", "mcdcrLgK", "mvdcrLgK")
class MCompnLgKappa2DInfo(MultiComponent2DInfo):
    r"""
    Same as standard multi-component diffusion-convection-reaction equation,
    but with the coefficients written in the $10^a$ form.
    """
    LG_KAPPA: bool = True

    @classmethod
    def dag_file_suffix(cls,
                        config: DictConfig,
                        file_ext: str = ".hdf5") -> str:
        return "_lgK" + super().dag_file_suffix(config, file_ext=file_ext)


@record_pde_info("mcwave", "mcWave", "MCWave", "mvwave")
class MCWave2DInfo(MultiComponent2DInfo):
    r"""
    ======== Wave Equation with Multiple Components ========
    The PDE takes the form
        $$\partial_{tt}u_i + \mu_i(r)\partial_tu_i + L_iu_i + f_0(u)_i + s_i(r)
            + \partial_xf_1(u)_i + \partial_yf_2(u)_i = 0,$$
    $u_i(0,r)=g_i(r)$, $\partial_tu_i(0,r)=g_i(r)$, $t\in[0,1]$,
    $r=(x,y)\in[0,1]^2$, $0 \le i,j,k \le d_u-1$, $j \le k$.

    Here, each spatial second-order term $L_iu_i$ is randomly selected from
    the non-divergence form $L_iu_i=-a_i(r)\Delta u_i$, the factored form
    $L_iu_i=-\sqrt a_i(r)\nabla\cdot(\sqrt a_i(r)\nabla u_i)$, and the
    divergence form $L_iu_i=-\nabla\cdot(a_i(r)\nabla u_i)$ with equal
    probability, where $a_i(r)=a_i$ is taken to be a random scalar for
    simplicity, and $r=(x,y)$ denotes the spatial coordinates.

    We take $f_l(u)_i = \sum_ja_{lij}u_j + \sum_{j,k}b_{lijk}u_ju_k$.
    The coefficients $a,b$ are sparse arrays, with a total of at most $3d_u$
    non-zero entries.
    """
    IS_WAVE: bool = True


@record_pde_info("divConstrDCR", "dcdcr")
class DivConstraintDCR2DInfo(MultiComponent2DInfo):
    r"""
    ======== Diffusion-Convection-Reaction PDE with Divergence-Constraint ========
    The PDE takes the form
        $$\partial_tu_i + L_iu_i + f_0(u)_i + s_i(r) + \partial_xf_1(u)_i
            + \partial_yf_2(u)_i + (-c_i)p + (\nabla p)_i = 0,$$
        $$\partial_xu_0 + \partial_yu_1 + c_0u_0 + c_1u_1 + c_2 = 0,$$
    $t\in[0,1]$, $r=(x,y)\in[0,1]^2$, $0 \le i,j,k \le 1$, $j \le k$.
    When the initial value is required to comply with the divergence
    constraint, the initial condition is taken as
    $u(0,r)=(c+\nabla)\times\psi(r)$.
    Periodic boundary conditions are employed for simplicity.

    Here, each spatial second-order term $L_iu_i$ is randomly selected from
    the non-divergence form $L_iu_i=-a_i(r)\Delta u_i$, the factored form
    $L_iu_i=-\sqrt a_i(r)\nabla\cdot(\sqrt a_i(r)\nabla u_i)$, and the
    divergence form $L_iu_i=-\nabla\cdot(a_i(r)\nabla u_i)$ with equal
    probability, where $a_i(r)=a_i$ is taken to be a random scalar for
    simplicity, and $r=(x,y)$ denotes the spatial coordinates.

    We take $f_l(u)_i = \sum_ja_{lij}u_j + \sum_{j,k}b_{lijk}u_ju_k$.
    The coefficients $a,b$ are sparse arrays, with a total of at most $6$
    non-zero entries.
    """
    DIV_CONSTRAINT: bool = True

    @staticmethod
    def var_latex(idx_var: int) -> str:
        if idx_var == 2:
            return "p"
        return f"u_{idx_var}"


@record_pde_info("divConstrDCRLgK", "dcdcrLgK")
class DivConstraintDCRLgKappa2DInfo(DivConstraintDCR2DInfo):
    r"""
    Same as the standard diffusion-convection-reaction PDE with
    divergence-constraint, but with the coefficients written in the $10^a$ form.
    """
    LG_KAPPA: bool = True

    @classmethod
    def dag_file_suffix(cls,
                        config: DictConfig,
                        file_ext: str = ".hdf5") -> str:
        return "_lgK" + super().dag_file_suffix(config, file_ext=file_ext)


@record_pde_info("divConstrWave", "dcwave")
class DivConstraintWave2DInfo(DivConstraintDCR2DInfo):
    r"""
    ======== Wave Equation with Divergence-Constraint ========
    The PDE takes the form
        $$\partial_{tt}u_i + \mu_i(r)\partial_tu_i + L_iu_i + f_0(u)_i + s_i(r)
            + \partial_xf_1(u)_i + \partial_yf_2(u)_i
            + (-c_i)p + (\nabla p)_i = 0,$$
        $$\partial_xu_0 + \partial_yu_1 + c_0u_0 + c_1u_1 + c_2 = 0,$$
    $t\in[0,1]$, $r=(x,y)\in[0,1]^2$, $0 \le i,j,k \le 1$, $j \le k$.
    When the initial value is required to comply with the divergence
    constraint, the initial condition is taken as
    $u(0,r)=(c+\nabla)\times\psi_1(r)$, $u_t(0,r)=(c+\nabla)\times\psi_2(r)$.
    Periodic boundary conditions are employed for simplicity.

    Here, each spatial second-order term $L_iu_i$ is randomly selected from
    the non-divergence form $L_iu_i=-a_i(r)\Delta u_i$, the factored form
    $L_iu_i=-\sqrt a_i(r)\nabla\cdot(\sqrt a_i(r)\nabla u_i)$, and the
    divergence form $L_iu_i=-\nabla\cdot(a_i(r)\nabla u_i)$ with equal
    probability, where $a_i(r)=a_i$ is taken to be a random scalar for
    simplicity, and $r=(x,y)$ denotes the spatial coordinates.

    We take $f_l(u)_i = \sum_ja_{lij}u_j + \sum_{j,k}b_{lijk}u_ju_k$.
    The coefficients $a,b$ are sparse arrays, with a total of at most $6$
    non-zero entries.
    """
    IS_WAVE: bool = True


@record_pde_info("swe", "shallowWater", "gswe")
class ShallowWater2DInfo(PDEInfoBase):
    r"""
    ======== Shallow Water Equation ========
    The PDE takes the form
        $$h_t + L_hh + f_h + s_h(r) + ((h+H(r))u)_x + ((h+H(r))v)_y = 0,$$
        $$u_t + L_uu + f_u + s_u(r) + uu_x + vu_y + g_1h_x = 0,$$
        $$v_t + L_vv + f_v + s_v(r) + uv_x + vv_y + g_2h_y = 0,$$
    $\eta(0,r)=g_\eta(r)$ for $\eta\in\{h,u,v\}$, $t\in[0,1]$,
    $r=(x,y)\in[0,1]^2$.
    Periodic boundary conditions are employed for simplicity.
    We take $[f_h;f_u;f_v] = f_0([h;u;v])$ with $f_0$ being the same as that of
    multi-component DCR/Wave equations.
    The initial water height $g_h(r)$ is taken to be a non-negative random
    field. The base height of the water $H(r)$ is also non-negative in the
    non-zero case.
    """
    LG_KAPPA: bool = False

    @staticmethod
    def var_latex(idx_var: int) -> str:
        return "huv"[idx_var]

    @classmethod
    def pde_nodes(cls,
                  h5file_in: h5py.File,
                  idx_pde: Union[int, Tuple[int]],
                  keep_all_coef: bool) -> PDENodesCollector:
        pde = PDENodesCollector()
        h_node, u_node, v_node = pde.new_uf(), pde.new_uf(), pde.new_uf()
        var_list = [h_node, u_node, v_node]

        # Basic info
        x_ext = h5file_in["coord/x"]
        y_ext = h5file_in["coord/y"]
        coord_dict = {"x": x_ext, "y": y_ext}
        term_obj_dict = cls._get_term_dict(h5file_in, idx_pde, keep_all_coef)

        f0_lists = term_obj_dict["f0"].nodes(pde, var_list)
        h_total = h_node + term_obj_dict["h_base"].nodes(pde, coord_dict)
        for i, var_name in enumerate("huv"):
            # Initial condition
            pde.set_ic(var_list[i], h5file_in[f"coef/ic/{var_name}"][idx_pde],
                       x=x_ext, y=y_ext)

            sum_i_list = f0_lists[i]
            sum_i_list.append(term_obj_dict[f"s/{var_name}"].nodes(pde, coord_dict))
            if var_name == "h":
                sum_i_list.append(pde.dx(h_total * u_node))
                sum_i_list.append(pde.dy(h_total * v_node))
            elif var_name == "u":
                sum_i_list.append(u_node * u_node.dx)
                sum_i_list.append(v_node * u_node.dy)
                g_value = term_obj_dict["g1"].nodes(pde)
                sum_i_list.append(g_value * h_node.dx)
            elif var_name == "v":
                sum_i_list.append(u_node * v_node.dx)
                sum_i_list.append(v_node * v_node.dy)
                g_value = term_obj_dict["g2"].nodes(pde)
                sum_i_list.append(g_value * h_node.dy)

            # Second-order term
            if h5file_in["args/inhom_diff_u"][()]:
                sum_i_list.append(term_obj_dict[f"L/{var_name}"].nodes(
                    pde, var_list[i], coord_dict))
            else:
                sum_i_list.append(term_obj_dict[f"L/{var_name}"].nodes(pde, var_list[i]))

            sum_i_list.append(var_list[i].dt)
            pde.sum_eq0(sum_i_list)

        return pde

    @classmethod
    def pde_latex(cls,
                  h5file_in: h5py.File,
                  idx_pde: Union[int, Tuple[int]],
                  keep_all_coef: bool) -> Tuple:
        coef_dict = OrderedDict()
        term_obj_dict = cls._get_term_dict(h5file_in, idx_pde, keep_all_coef)
        latex_sum_list = [terms.LaTeXSum(keep_all_coef, coef_dict)
                          for _ in "huv"]

        # h_total_sum
        h_total_sum = terms.LaTeXSum(keep_all_coef, coef_dict)
        h_total_sum.add_term("h")
        term_obj_dict["h_base"].add_latex(h_total_sum, "H")
        h_total_sum = h_total_sum.strip_sum()  # LaTeXSum -> str
        if "+" in h_total_sum:
            h_total_sum = "(" + h_total_sum + ")"

        # basic SWE
        for i, var_name in enumerate("huv"):
            latex_sum_list[i].add_term(var_name + "_t")
        latex_sum_list[0].add_term("(" + h_total_sum + "u)_x")
        latex_sum_list[0].add_term("(" + h_total_sum + "v)_y")
        latex_sum_list[1].add_term("uu_x")
        latex_sum_list[1].add_term("vu_y")
        term_obj_dict["g1"].add_latex(latex_sum_list[1], "g_1", "h_x")
        latex_sum_list[2].add_term("uv_x")
        latex_sum_list[2].add_term("vv_y")
        term_obj_dict["g2"].add_latex(latex_sum_list[2], "g_2", "h_y")

        # additional terms
        for i, var_name in enumerate("huv"):
            term_obj_dict[f"L/{var_name}"].add_latex(
                latex_sum_list[i], f"a^{var_name}", var_name)
            term_obj_dict[f"s/{var_name}"].add_latex(
                latex_sum_list[i], f"s^{var_name}")
        term_obj_dict["f0"].add_latex(latex_sum_list, "huv")

        pde_latex_list = [latex_sum.strip_sum("$", "=0$")
                          for latex_sum in latex_sum_list]
        pde_latex = "\n".join(pde_latex_list)
        return pde_latex, coef_dict

    @classmethod
    def _get_term_dict(cls,
                       h5file_in: h5py.File,
                       idx_pde: Union[int, Tuple[int]],
                       keep_all_coef: bool) -> Dict[str, terms.PDETermBase]:
        r"""Get dictionary of term objectives in a PDE."""
        term_cls_dict = {"h_base": terms.ConstOrField,
                         "f0": terms.MultiComponentMixedTerm,
                         "g1": terms.ScalarCoef,
                         "g2": terms.ScalarCoef}
        if not np.all(h5file_in["args/periodic"]):
            raise NotImplementedError(
                "Dataset 'SWE2D' only supports periodic BC.")
        for var_name in "huv":
            term_cls_dict[f"s/{var_name}"] = terms.ConstOrField

            # spatial 2nd-order term
            if h5file_in["args/inhom_diff_u"][()]:
                raise NotImplementedError(
                    "Dataset 'SWE2D' only supports homogeneous diffusion "
                    "coefficients.")  # change below to 'elif' after finished
            if cls.LG_KAPPA:
                term_cls_dict[f"L/{var_name}"] = terms.LgCoefHomSpatialOrder2Term
            else:
                term_cls_dict[f"L/{var_name}"] = terms.HomSpatialOrder2Term

        return _term_cls_to_obj(term_cls_dict, h5file_in, idx_pde, keep_all_coef)


@record_pde_info("sweLgK", "shallowWaterLgK", "gsweLgK")
class ShallowWaterLgKappa2DInfo(ShallowWater2DInfo):
    r"""
    Same as the standard shallow-water equation, but with the coefficients
    written in the $10^a$ form.
    """
    LG_KAPPA: bool = True

    @classmethod
    def dag_file_suffix(cls,
                        config: DictConfig,
                        file_ext: str = ".hdf5") -> str:
        return "_lgK" + super().dag_file_suffix(config, file_ext=file_ext)


@record_pde_info("elasticwave", "elasticWave", "ElasticWave")
class ElasticWave2DInfo(PDEInfoBase):
    r"""
    ======== Elastic Wave Equation ========
    The PDE takes the form
        $\rho(r)\partial_{tt}u_i-\sigma_{ji,j}-fr_i(r)ft_i(t)-f_i(r)=0,$
    $u_i(0,r)=g_i(r)$, $\partial_t u_i(0,r)=h_i(r)$, $t\in[0,1]$,
    $r=(x,y)\in[0,1]^2$.

    Here, $\sigma_{ij} = \sigma_{ji}$ is the stress tensor, $\rho(r)$ is the
    density, $f(r)$ is the time-independent external force, and $fr_i(r)ft_i(t)$
    is the time-dependent external force factorized into spatial and temporal
    components. The stress (\sigma_{11}, \sigma_{22}, \sigma_{12})^T is
    determined by the strain (\epsilon_{11}, \epsilon_{22}, \epsilon_{12})^T
    through a 3x3 matrix $C$:
        $\sigma_{ij} = C_{ijkl}(r)\epsilon_{kl}.$
    The strain is given by
        $\epsilon_{ij}=\frac{1}{2}(\partial_i u_j+\partial_j u_i)$.

    ======== Detailed Description ========
    - Boundary condition on each side is randomly chosen from Dirichlet,
      Neumann, and Robin. The values of boundary conditions are randomly chosen
      from zero, random constant, and random 1D function.
    - The initial displacement $g_i(r)$ is set to be the numerical solution of
      the Steady-State equation $\sigma_{ji,j}+f_i(r)=0$ with specific boundary
      conditions. These boundary conditions are either the same as the ones used
      in the time-dependent equation, or chosen randomly again.
      When introduce motion by initial displacement, a gaussian function will
      be added to the initial displacement.
    - The initial velocity $h_i(r)$ is set to zero.
    - The density $\rho(r)$ and elements of the stiffness tensor $C_{ijkl}(r)$
      are randomly chosen from random positive constants and random 2D
      functions.
    - The time-independent external force $f_i(r)$ is randomly chosen from zero,
      random constant, random constant times $\rho(r)$ and random 2D functions.
    - The time-dependent external force can be factorized into spatial and
      temporal components $fr_i(r)$ and $ft_i(t)$. The spatial component is
      set to be a gaussian function, and the temporal component is set to be a
      Ricker wavelet.
    - The motion is either introduced by the time-dependent external force,
      change of boundary condition, change of initial displacement, randomly
      generated initial displacement, or randomly generated initial velocity,
      each with equal probability.
    """
    LG_RHO: bool = False  # represent density in 10^a form
    N_BC_PER_LINE = 4
    n_vars: int = 2

    TIMEDEP_FORCE = 0
    BC_CHANGE = 1
    IC_CHANGE = 2
    RANDOM_IC = 3
    RANDOM_IVEL = 4
    SUPPORTED_MOTION_TYPES = [
        TIMEDEP_FORCE, BC_CHANGE, IC_CHANGE, RANDOM_IC, RANDOM_IVEL]
    # SUPPORTED_MOTION_TYPES = [BC_CHANGE, IC_CHANGE, RANDOM_IC, RANDOM_IVEL]

    @staticmethod
    def var_latex(idx_var: int) -> str:
        return f"u_{{{idx_var}}}"

    @classmethod
    def dag_file_suffix(cls,
                        config: DictConfig,
                        file_ext: str = ".hdf5") -> str:
        if DAG_BC_VERSION != 2:
            raise RuntimeError("'DAG_BC_VERSION' only supports 2, but got "
                               f"{DAG_BC_VERSION}.")
        return "_v2" + super().dag_file_suffix(config, file_ext=file_ext)

    @classmethod
    def pde_nodes(cls,
                  h5file_in: h5py.File,
                  idx_pde: Union[int, Tuple[int]],
                  keep_all_coef: bool) -> PDENodesCollector:
        n_vars = h5file_in.get("args/n_vars", cls.n_vars)
        if n_vars != cls.n_vars:
            raise NotImplementedError(
                f"ElasticWave2D only supports {cls.n_vars} components.")
        motion_type = cls._get_motion_type(h5file_in, idx_pde)
        pde = PDENodesCollector(dim=2)

        # Basic info
        x_ext = h5file_in["coord/x"][:].reshape(-1, 1)  # Shape is (n_x, 1)
        y_ext = h5file_in["coord/y"][:].reshape(1, -1)  # Shape is (1, n_y)
        resolution = (len(x_ext), len(y_ext))        # (n_x, n_y)
        t_coord = h5file_in["coord/t"][:].reshape(-1)   # Shape is (n_t,)
        coord_dict = {"x": x_ext, "y": y_ext}
        term_obj_dict = cls._get_term_dict(h5file_in, idx_pde, keep_all_coef)

        # Domain and unknown variables
        x_neg_sdf = np.maximum(-x_ext, x_ext - 1)   # Shape is (n_x, 1)
        y_neg_sdf = np.maximum(-y_ext, y_ext - 1)   # Shape is (1, n_y)
        neg_sdf = np.maximum(x_neg_sdf, y_neg_sdf)  # Shape is (n_x, n_y)
        domain = pde.new_domain(neg_sdf, x=x_ext, y=y_ext)
        u_list = [pde.new_uf(domain) for _ in range(n_vars)]

        # stress tensor
        s_matrix = term_obj_dict["C"].nodes(pde, u_list, coord_dict)  # 2x2 nested list

        rho = term_obj_dict["rho"].nodes(pde, coord_dict)
        for i in range(n_vars):
            # Initial condition
            pde.set_ic(u_list[i], h5file_in["sol/ic"][idx_pde, ..., i],
                       x=x_ext, y=y_ext)
            if motion_type == cls.RANDOM_IVEL:
                ut_ic_field = h5file_in["coef/ut_ic/field"][idx_pde, ..., i]
            else:
                ut_ic_field = np.zeros(resolution)
            pde.set_ic(u_list[i].dt, ut_ic_field, x=x_ext, y=y_ext)

            sum_i_list = []
            # spatial 2nd-order term (divergence of stress tensor)
            sum_i_list.extend([s_matrix[i][0].dx, s_matrix[i][1].dy])
            # source term (external force)
            sum_i_list.append(term_obj_dict["f"].nodes(
                pde, i, coord_dict, neg=True))
            if motion_type == cls.TIMEDEP_FORCE:
                sum_i_list.append(term_obj_dict["f_rt"].nodes(
                    pde, i, coord_dict, t_coord, neg=True))
            # temporal 2nd-order term (acceleration)
            sum_i_list.append(rho * u_list[i].dt.dt)
            pde.sum_eq0(sum_i_list)

        # Boundary Condition
        term_obj_dict["u_bc"].nodes(pde, u_list, domain, coord_dict)
        return pde

    @classmethod
    def pde_latex(cls,
                  h5file_in: h5py.File,
                  idx_pde: Union[int, Tuple[int]],
                  keep_all_coef: bool) -> Tuple:
        coef_dict = OrderedDict()
        latex_sum_list = [terms.LaTeXSum(keep_all_coef, coef_dict)
                          for _ in range(cls.n_vars)]
        term_obj_dict = cls._get_term_dict(h5file_in, idx_pde, keep_all_coef)
        # TODO: add latex for PDE Terms
        motion_type = cls._get_motion_type(h5file_in, idx_pde)
        if motion_type == cls.TIMEDEP_FORCE:
            pde_str = "Time-dependent force"
        elif motion_type == cls.BC_CHANGE:
            pde_str = "Boundary condition change"
        elif motion_type == cls.IC_CHANGE:
            pde_str = "Initial condition change"
        elif motion_type == cls.RANDOM_IC:
            pde_str = "Random initial condition"
        elif motion_type == cls.RANDOM_IVEL:
            pde_str = "Random initial velocity"
        else:
            raise ValueError("Unsupported motion_type.")
        bc_lists = term_obj_dict["u_bc"].add_latex(latex_sum_list)
        eqn_list = [pde_str]
        for i in range(cls.n_vars):
            bc_list = bc_lists[i]
            for j in range(0, len(bc_list), cls.N_BC_PER_LINE):
                eqn_list.append(r",\ ".join(bc_list[j:j+cls.N_BC_PER_LINE]))
        pde_latex = "$\n$".join(eqn_list)
        pde_latex = "$" + pde_latex + "$"
        return pde_latex, coef_dict

    @classmethod
    def _get_term_dict(cls,
                       h5file_in: h5py.File,
                       idx_pde: Union[int, Tuple[int]],
                       keep_all_coef: bool) -> Dict[str, terms.PDETermBase]:
        r"""Get dictionary of term objectives in a PDE."""
        term_cls_dict = {"C": terms.StiffnessMatrix2D2Comp,
                         "f": terms.TimeIndepForce,
                         "f_rt": terms.TimeDepForce}
        # term_cls_dict = {"C": terms.StiffnessMatrix2D2Comp,
        #                  "f": terms.TimeIndepForce}
        if cls.LG_RHO:
            term_cls_dict["rho"] = terms.LgNonNegConstOrField
        else:
            term_cls_dict["rho"] = terms.ConstOrField
        # TODO: change the saving format of boundary conditions to reuse code.
        if DAG_BC_VERSION == 1:
            term_cls_dict["u_bc"] = boundary.DirichletOrNeumannBoxDomainBC
        elif DAG_BC_VERSION == 2:
            term_cls_dict["u_bc"] = boundary_v2.DOrNBoxDomainBCV2
        else:
            raise NotImplementedError

        return _term_cls_to_obj(term_cls_dict, h5file_in, idx_pde, keep_all_coef)

    @classmethod
    def _get_motion_type(cls, h5file_in: h5py.File, idx_pde: int) -> int:
        motion_type = h5file_in["coef/motion_type"][idx_pde]
        if motion_type not in cls.SUPPORTED_MOTION_TYPES:
            raise ValueError("Unsupported motion_type.")
        return motion_type

    @classmethod
    def gen_dag_info(cls,
                     filename: str,
                     config: DictConfig,
                     print_fn: Callable[[str], None] = print) -> None:
        r"""
        Precompute Graphormer input (PDE DAG info) for a single data file, and
        save the results to a HDF5 file.
        """
        # target DAG data file
        dag_filepath = cls.dag_info_file_path(config, filename)
        if os.path.exists(dag_filepath):
            return  # no need to (re)generate DAG file
        print_fn(" generating " + dag_filepath)

        n_scalar_nodes = config.data.pde_dag.max_n_scalar_nodes
        n_function_nodes = config.data.pde_dag.max_n_function_nodes
        function_num_branches = config.model.function_encoder.num_branches
        n_node = n_scalar_nodes + n_function_nodes * function_num_branches

        # h5file_in
        u_filepath = os.path.join(config.data.path, filename + ".hdf5")
        h5file_in = h5py.File(u_filepath, "r")

        # data to be saved
        # # Shape is [n_pde, n_t, n_x, n_y].
        # n_pde, _, n_x, n_y = h5file_in["sol/u"].shape
        n_pde = h5file_in["args/num_pde"][()]
        n_x = np.size(h5file_in["coord/x"])
        n_y = np.size(h5file_in["coord/y"])
        node_type = np.zeros([n_pde, n_node, 1], dtype=int_dtype)
        node_scalar = np.zeros([n_pde, n_scalar_nodes, 1], dtype=float_dtype)
        node_func_t = np.zeros([n_pde, n_function_nodes, 1, 1],
                               dtype=float_dtype)
        node_func_f = np.zeros([n_pde, n_function_nodes, n_x * n_y, 1],
                               dtype=float_dtype)
        spatial_pos = np.zeros([n_pde, n_node, n_node], dtype=np.uint8)
        in_degree = np.zeros([n_pde, n_node], dtype=int_dtype)
        out_degree = np.zeros([n_pde, n_node], dtype=int_dtype)

        is_valid = np.zeros(n_pde, dtype=bool)
        for idx_pde in range(n_pde):
            try:
                pde = cls.pde_nodes(h5file_in, idx_pde, keep_all_coef=False)
                pde_dag = pde.gen_dag(config)
                is_valid[idx_pde] = True
            except Exception as err:  # pylint: disable=broad-except
                is_valid[idx_pde] = False
                print_fn(f"Error in idx_pde={idx_pde}: {err}")
                continue

            node_type[idx_pde] = pde_dag.node_type
            node_scalar[idx_pde] = pde_dag.node_scalar
            node_func_t[idx_pde] = pde_dag.node_function[:, :1, :1]
            node_func_f[idx_pde] = pde_dag.node_function[:, :, -1:]
            spatial_pos[idx_pde] = pde_dag.spatial_pos
            in_degree[idx_pde] = pde_dag.in_degree
            out_degree[idx_pde] = pde_dag.out_degree
        n_pde_valid = np.sum(is_valid)
        valid_idx = np.where(is_valid)[0]

        h5file_in.close()
        with h5py.File(dag_filepath, "w") as h5file_out:
            h5file_out.create_dataset("node_type", data=node_type)
            h5file_out.create_dataset("node_scalar", data=node_scalar)
            h5file_out.create_dataset("node_func_t", data=node_func_t)
            h5file_out.create_dataset("node_func_f", data=node_func_f)
            h5file_out.create_dataset("spatial_pos", data=spatial_pos)
            h5file_out.create_dataset("in_degree", data=in_degree)
            h5file_out.create_dataset("out_degree", data=out_degree)
            h5file_out.create_dataset("valid_idx", data=valid_idx)
            h5file_out.create_dataset("num_pde_valid", data=n_pde_valid)


@record_pde_info("elasticsteady", "elasticSteady", "ElasticSteady",
                 "ElasticStatic", "ElasticSteadyState")
class ElasticSteady2DInfo(ElasticWave2DInfo):
    r"""
    ======== Elastic Steady State Equation ========
    The PDE takes the form
        $\sigma_{ji,j}+f_i(r)=0,$
    $r=(x,y)\in[0,1]^2$.

    Here, $\sigma_{ij} = \sigma_{ji}$ is the stress tensor, $f(r)$ is the external
    force. The stress (\sigma_{11}, \sigma_{22}, \sigma_{12})^T is determined by the
    strain (\epsilon_{11}, \epsilon_{22}, \epsilon_{12})^T through a 3x3 matrix $C$:
        $\sigma_{ij} = C_{ijkl}(r)\epsilon_{kl}.$
    The strain is given by
        $\epsilon_{ij}=\frac{1}{2}(\partial_i u_j+\partial_j u_i)$.

    ======== Detailed Description ========
    - Boundary condition on each side is randomly chosen from Dirichlet, Neumann,
      and Robin. The values of boundary conditions are randomly chosen from zero,
      random constant, and random 1D function.
    - The density $\rho(r)$ and elements of the stiffness tensor $C_{ijkl}(r)$ are
      randomly chosen from random positive constants and random 2D functions.
    - The external force $f_i(r)$ is randomly chosen from zero, random constant,
      random constant times $\rho(r)$ and random 2D functions.
    """
    n_vars: int = 2
    LATEX_VECTORIZED = False

    @classmethod
    def pde_nodes(cls,
                  h5file_in: h5py.File,
                  idx_pde: Union[int, Tuple[int]],
                  keep_all_coef: bool) -> PDENodesCollector:
        n_vars = h5file_in.get("args/n_vars", cls.n_vars)
        if n_vars != cls.n_vars:
            raise NotImplementedError(
                f"ElasticSteady2D only supports {cls.n_vars} components.")
        pde = PDENodesCollector(dim=2)

        # Basic info
        x_ext = h5file_in["coord/x"][:].reshape(-1, 1)  # Shape is (n_x, 1)
        y_ext = h5file_in["coord/y"][:].reshape(1, -1)  # Shape is (1, n_y)
        coord_dict = {"x": x_ext, "y": y_ext}
        term_obj_dict = cls._get_term_dict(h5file_in, idx_pde, keep_all_coef)

        # Domain and unknown variables
        x_neg_sdf = np.maximum(-x_ext, x_ext - 1)  # Shape is (n_x, 1)
        y_neg_sdf = np.maximum(-y_ext, y_ext - 1)  # Shape is (1, n_y)
        neg_sdf = np.maximum(x_neg_sdf, y_neg_sdf)
        domain = pde.new_domain(neg_sdf, x=x_ext, y=y_ext)
        u_list = [pde.new_uf(domain) for _ in range(n_vars)]

        # stress tensor
        s_matrix = term_obj_dict["C"].nodes(pde, u_list, coord_dict)

        for i in range(n_vars):
            sum_i_list = []
            # spatial 2nd-order term (divergence of stress tensor)
            sum_i_list.extend([s_matrix[i][0].dx, s_matrix[i][1].dy])
            # source term (external force)
            sum_i_list.append(term_obj_dict["f"].nodes(
                pde, i, coord_dict, neg=True))
            pde.sum_eq0(sum_i_list)

        # Boundary Condition
        term_obj_dict["u_bc"].nodes(pde, u_list, domain, coord_dict)
        return pde

    @classmethod
    def pde_latex(cls,
                  h5file_in: h5py.File,
                  idx_pde: Union[int, Tuple[int]],
                  keep_all_coef: bool) -> Tuple:
        coef_dict = OrderedDict()
        latex_sum_list = [terms.LaTeXSum(keep_all_coef, coef_dict)
                          for _ in range(cls.n_vars)]
        term_obj_dict = cls._get_term_dict(h5file_in, idx_pde, keep_all_coef)
        bc_lists = term_obj_dict["u_bc"].add_latex(latex_sum_list)
        if cls.LATEX_VECTORIZED:
            eqn_list = [(
                r"\nabla\cdot\left(\lambda(r)(\nabla\cdot\mathbf{u})\mathbf{I}"
                r"+\mu(r)(\nabla\mathbf{u}+\nabla\mathbf{u}^\mathrm{T})\right)"
                r"+\mathbf{f}(r)=0,\ \mathbf{u}=(u,v)")]
        else:
            eqn_list = [(r"(\lambda(r)(u_x+v_y)+\mu_2(r)u_x)_x"
                         r"+(\frac{1}{2}\mu_2(r)(u_y+v_x))_y+f_1(r)=0"),
                        (r"(\frac{1}{2}\mu_2(r)(u_y+v_x))_x"
                         r"+(\lambda(r)(u_x+v_y)+\mu_2(r)v_y)_y+f_2(r)=0")]
        for i in range(cls.n_vars):
            bc_list = bc_lists[i]
            for j in range(0, len(bc_list), cls.N_BC_PER_LINE):
                eqn_list.append(r",\ ".join(bc_list[j:j+cls.N_BC_PER_LINE]))
        pde_latex = "$\n$".join(eqn_list)
        pde_latex = "$" + pde_latex + "$"
        return pde_latex, coef_dict

    @classmethod
    def _get_term_dict(cls,
                       h5file_in: h5py.File,
                       idx_pde: Union[int, Tuple[int]],
                       keep_all_coef: bool) -> Dict[str, terms.PDETermBase]:
        term_cls_dict = {"C": terms.StiffnessMatrix2D2Comp,
                         "f": terms.TimeIndepForce}
        if DAG_BC_VERSION == 1:
            term_cls_dict["u_bc"] = boundary.DirichletOrNeumannBoxDomainBC
        elif DAG_BC_VERSION == 2:
            term_cls_dict["u_bc"] = boundary_v2.DOrNBoxDomainBCV2
        else:
            raise NotImplementedError
        return _term_cls_to_obj(term_cls_dict, h5file_in, idx_pde, keep_all_coef)


def gen_file_list(data_files: FileListType) -> List[str]:
    r"""Create a file list based on prescribed data format."""
    if isinstance(data_files, str):
        return [data_files]
    if isinstance(data_files, ListConfig):
        return data_files
    if "indices" in data_files:
        indices = data_files.indices
    else:
        begin = data_files.begin
        step = data_files.get("step", 1)
        indices = range(begin, begin + data_files.num * step, step)
    file_list = [data_files.format % ind for ind in indices]
    return file_list


def preprocess_dag_info(config: DictConfig,
                        print_fn: Callable[[str], None] = print) -> None:
    r"""
    Precompute Graphormer input (PDE DAG info) for all custom multi_pde data
    files, and save the results to the corresponding HDF5 files.
    """
    os.makedirs(os.path.join(config.data.path, DAG_INFO_DIR), exist_ok=True)

    def process_file_dict(file_dict: Dict[str, FileListType]) -> None:
        for pde_type, file_list in file_dict.items():
            pde_info_cls = get_pde_info_cls(pde_type)
            file_list = gen_file_list(file_list)
            for filename in file_list:
                pde_info_cls.gen_dag_info(filename, config, print_fn)

    process_file_dict(config.data.multi_pde.train)
    process_file_dict(config.data.multi_pde.get("test", {}))
    print_fn("All auxiliary DAG data generated.")
