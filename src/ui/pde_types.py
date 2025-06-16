r"""Python classes for PDEformer inference on various PDE types."""
from typing import Union, Dict, Optional, Tuple
from abc import ABC, abstractmethod
from collections import OrderedDict
from argparse import Namespace
from functools import reduce

import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig

from src.cell import get_model
from src.data.env import DAG_BC_VERSION
from src.data.pde_dag import PDEAsDAG, PDENodesCollector
from src.data.multi_pde import pde_types
from src.data.multi_pde import terms_from_dict as terms
from src.data.multi_pde import boundary_v2_from_dict as boundary_v2
from src.inference import inference_cartesian


pdeformer_solver_cls_dict = {}


def record_pdeformer_solver(*pde_type_all):
    r"""Register the current PDEformer solver class with specific names."""
    def add_class(cls):
        for pde_type in pde_type_all:
            pdeformer_solver_cls_dict[pde_type] = cls
        return cls
    return add_class


def get_pde_solver_cls(pde_type: str,
                       args: Union[Namespace, Dict]) -> type:
    r"""Get the class to solve the given PDE type."""
    pde_type = pde_type.split("_")[0]
    if args.get("solver_type", "PDEformer").lower() == "pdeformer":
        return pdeformer_solver_cls_dict[pde_type]
    raise ValueError(f"Unsupported solver type: {args.solver_type}.")


class PDEformerSolverBase(ABC):
    r"""Base class for solving PDEs using pre-trained PDEformer models."""

    def __init__(self,
                 config: DictConfig) -> None:
        self.config = config
        self.model = get_model(config)

    def solve(self,
              term_dict: Dict,
              arg_dict: Dict) -> NDArray[float]:
        r"""
        Solve the PDE specified by the given terms and arguments.
        """
        pde = self.pde_nodes(term_dict, arg_dict)
        pde_dag = pde.gen_dag(self.config)
        if "t_coord" not in arg_dict or "x_coord" not in arg_dict:
            raise ValueError("Both `t_coord` and `x_coord` must be provided in "
                             "the arg_dict.")
        t_coord = arg_dict["t_coord"]
        x_coord = arg_dict["x_coord"]
        y_coord = arg_dict.get("y_coord", 0.)
        z_coord = arg_dict.get("z_coord", 0.)
        pred = inference_cartesian(
            self.model, pde_dag, t_coord, x_coord, y_coord, z_coord)
        return pred

    @classmethod
    @abstractmethod
    def pde_nodes(cls, term_dict: Dict, arg_dict: Dict) -> PDENodesCollector:
        r"""
        Construct the PDE nodes from the given terms and arguments.
        """

    @classmethod
    @abstractmethod
    def pde_latex(cls,
                  term_dict: Dict,
                  arg_dict: Dict,
                  args: Optional[Union[Namespace, DictConfig]] = None,
                  ) -> Tuple:
        r"""
        Generate the mathematical representation as well as a dictionary of
        coefficient values for the current PDE from the given terms and
        arguments.

        Returns:
            pde_latex (str): LaTeX expression of the PDE.
            coef_dict (OrderedDict): Dictionary of the coefficient values.
        """


def _term_cls_to_obj(term_cls_dict: Dict[str, type],
                     term_dict: Dict,
                     arg_dict: Dict) -> Dict:
    keep_all_coef = arg_dict.get("keep_all_coef", False)
    return {key: term_cls(term_dict[key], keep_all_coef)
            for key, term_cls in term_cls_dict.items()}


@record_pdeformer_solver("diffConvecReac", "dcr")  # pylint: disable=missing-docstring
class PDEformerSolverDCR(PDEformerSolverBase):
    __doc__ = r"""Solve diffusion-convection-reaction (DCR) equations using
    pre-trained PDEformer models.""" + "\n" + pde_types.DiffConvecReac2DInfo.__doc__

    IS_WAVE: bool = False
    LG_KAPPA: bool = False
    N_BC_PER_LINE = 2

    @classmethod
    def pde_nodes(cls, term_dict: Dict, arg_dict: Dict) -> PDEAsDAG:
        # coef values always given on regular grid points
        x_ext = np.linspace(0, 1, 129)[:-1].reshape(-1, 1)
        y_ext = np.linspace(0, 1, 129)[:-1].reshape(1, -1)
        coord_dict = {"x": x_ext, "y": y_ext}
        pde = PDENodesCollector(dim=2)
        term_obj_dict = cls._get_term_dict(term_dict, arg_dict)

        # Domain and unknown variable
        if "domain" in term_dict:
            raise NotImplementedError("Only the unit square domain is supported.")
        periodic = arg_dict.get("periodic", True)
        if np.all(periodic):
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
        pde.set_ic(u_node, term_dict["u_ic"], x=x_ext, y=y_ext)
        if cls.IS_WAVE:
            pde.set_ic(u_node.dt, term_dict["ut_ic"], x=x_ext, y=y_ext)

        # PDE terms
        sum_list = term_obj_dict["f0"].nodes(pde, u_node)
        sum_list.append(pde.dx(pde.sum(term_obj_dict["f1"].nodes(pde, u_node))))
        sum_list.append(pde.dy(pde.sum(term_obj_dict["f2"].nodes(pde, u_node))))
        sum_list.append(term_obj_dict["s"].nodes(pde, coord_dict))

        if arg_dict.get("inhom_diff_u", False):
            sum_list.append(term_obj_dict["Lu"].nodes(pde, u_node, coord_dict))
        else:
            sum_list.append(term_obj_dict["Lu"].nodes(pde, u_node))

        if cls.IS_WAVE:
            sum_list.append(term_obj_dict["mu"].nodes(pde, coord_dict) * u_node.dt)
            sum_list.append(u_node.dt.dt)
        else:
            sum_list.append(u_node.dt)

        pde.sum_eq0(sum_list)

        # Boundary condition
        if "bc" in term_dict:
            term_obj_dict["bc"].nodes(pde, u_node, domain, coord_dict)
        return pde

    @classmethod
    def _get_term_dict(cls,
                       term_dict: Dict,
                       arg_dict: Dict) -> Dict[str, terms.PDETermBase]:
        r"""Get the term objects from the term dictionary."""
        term_cls_dict = {"f0": terms.PolySinusNonlinTerm,
                         "f1": terms.PolySinusNonlinTerm,
                         "f2": terms.PolySinusNonlinTerm,
                         "s": terms.ConstOrField}
        if cls.IS_WAVE:
            term_cls_dict["mu"] = terms.ConstOrField

        # spatial 2nd-order term
        if arg_dict.get("inhom_diff_u", False):
            if cls.LG_KAPPA:
                term_cls_dict["Lu"] = terms.LgCoefInhomSpatialOrder2Term
            else:
                term_cls_dict["Lu"] = terms.InhomSpatialOrder2Term
        elif cls.LG_KAPPA:
            term_cls_dict["Lu"] = terms.LgCoefHomSpatialOrder2Term
        else:
            term_cls_dict["Lu"] = terms.HomSpatialOrder2Term

        # boundary conditions
        if "bc" not in term_dict:
            pass
        elif DAG_BC_VERSION == 1:
            raise NotImplementedError("Only version 2 of boundary conditions "
                                      "is supported.")
        elif DAG_BC_VERSION == 2:
            if cls.IS_WAVE:
                term_cls_dict["bc"] = boundary_v2.FullBCsWithMur
            else:
                term_cls_dict["bc"] = boundary_v2.FullBoundaryConditions
        else:
            raise NotImplementedError(f"Unsupported version {DAG_BC_VERSION}.")

        if "domain" in term_dict:
            raise NotImplementedError("Only the unit square domain is supported.")

        return _term_cls_to_obj(term_cls_dict, term_dict, arg_dict)

    @classmethod
    def pde_latex(cls,
                  term_dict: Dict,
                  arg_dict: Dict,
                  args: Optional[Union[Namespace, DictConfig]] = None,
                  ) -> Tuple:
        coef_dict = OrderedDict()
        keep_all_coef = arg_dict.get("keep_all_coef", False)
        latex_sum = terms.LaTeXSum(keep_all_coef, coef_dict)
        term_obj_dict = cls._get_term_dict(term_dict, arg_dict)

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


@record_pdeformer_solver("dcrLgK")
class PDEformerSolverDCRLgKappa(PDEformerSolverDCR):
    __doc__ = r"""Solve diffusion-convection-reaction (DCR) equations using
    pre-trained PDEformer models.""" + "\n" + pde_types.DCRLgKappa2DInfo.__doc__
    LG_KAPPA: bool = True


@record_pdeformer_solver("wave")
class PDEformerSolverWave(PDEformerSolverDCR):
    __doc__ = r"""Solve wave equations using pre-trained PDEformer models.""" + \
        "\n" + pde_types.Wave2DInfo.__doc__
    IS_WAVE: bool = True
