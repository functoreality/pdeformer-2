r"""
Loading datasets containing one specific PDE (single_pde) with solution
provided on cartesian grid points.
"""
import os
from typing import Tuple, Dict, Any, Optional, Union
from abc import abstractmethod

import h5py
import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig

from ..pde_dag import PDENodesCollector
from .basics import register_pde_type, CartesianGridInputFileDataset, EMPTY_SCALAR, float_dtype


@register_pde_type("thewell_acoustic")
class TheWellAcousticInputDataset(CartesianGridInputFileDataset):
    r"""
    Load 2D Acoustic Scattering dataset provided by TheWell.
    """
    n_vars: int = 3
    var_latex = "puv"
    dataset_size = 100  # per data file
    pde_latex = (r"$p_t+ku_x+kv_y=0$" + "\n"
                 r"$u_t+a(r)p_x=0$" + "\n"
                 r"$v_t+a(r)p_y=0$" + "\n"
                 r"$(un_x+vn_y)|_{LD}=0$" + "\n"
                 # r"$\partial_np|_{LD}=0$" + "\n"
                 r"$(u_x+v_y+(-\sqrt{a(r)/k})\partial_np)|_{RU}=0$")
    # NODE_FUNC_IDX = (1, 4, 7)  # IC of p; a(r); c(r)

    def __init__(self, config: DictConfig, pde_param: str) -> None:
        super().__init__(config, pde_param)

        # main data file
        filepath = f"acoustic_scattering_{pde_param}.hdf5"
        filepath = os.path.join(config.data.path, filepath)
        self.h5_file_u = h5py.File(filepath, "r")

        # spatio-temporal coordinates
        t_coord = self.h5_file_u["dimensions/time"][()]
        x_coord = self.h5_file_u["dimensions/x"][::2]
        y_coord = self.h5_file_u["dimensions/y"][::2]

        # temporal rescaling
        t_max = t_coord[-1]
        t_coord /= t_max
        k_value = 4 * t_max
        self.coef_dict = {"k": k_value, "T": t_max}

        # pde_dag
        pde = self._gen_pde_nodes(x_coord, y_coord, k_value)
        self.pde_dag = pde.gen_dag(config)
        self.txyz_coord = self._gen_coords(t_coord, x_coord, y_coord)

    def __getitem__(self, idx_pde: int) -> Tuple[NDArray[float]]:
        # solution
        p_label = self.h5_file_u["t0_fields/pressure"][idx_pde, :, ::2, ::2]
        uv_label = self.h5_file_u["t1_fields/velocity"][idx_pde, :, ::2, ::2]
        p_ic = p_label[0]
        # [n_t, n_x, n_y] -> [n_t, n_x, n_y, n_vars=1]
        p_label = np.expand_dims(p_label, axis=-1)
        sol_label = np.concatenate([p_label, uv_label], axis=-1)
        # [n_t, n_x, n_y, n_vars=3] -> [n_t, n_x, n_y, n_z=1, n_vars]
        sol_label = np.expand_dims(sol_label, axis=-2)

        # input_field
        rho = self.h5_file_u["t0_fields/density"][idx_pde, ::2, ::2]
        a_val = self.coef_dict["T"] / rho  # a(r) = T / rho(r)
        # b(r) = -\sqrt{a(r)/k}
        b_val = -np.sqrt(a_val / self.coef_dict["k"])
        # Shape is [n_x, n_y, n_fields=3].
        input_field = np.stack([p_ic, a_val, b_val], axis=-1)

        return input_field, EMPTY_SCALAR, self.txyz_coord, sol_label

    @staticmethod
    def _gen_pde_nodes(x_coord: NDArray[float],
                       y_coord: NDArray[float],
                       k_value: float) -> PDENodesCollector:
        r"""Generate nodes of a PDE for DAG construction"""
        pde = PDENodesCollector(dim=2)
        x_ext, y_ext = np.meshgrid(x_coord, y_coord, indexing="ij")

        # domain and variables
        sdf = np.stack([-x_ext, x_ext - 1, -y_ext, y_ext - 1]).max(axis=0)
        domain = pde.new_domain(sdf, x=x_ext, y=y_ext)  # field 0
        p_ = pde.new_uf(domain)
        u_ = pde.new_uf(domain)
        v_ = pde.new_uf(domain)

        # initial condition and coef field
        pde.set_ic(p_, np.nan, x=x_ext, y=y_ext)  # field 1
        pde.set_ic(u_, 0, x=x_ext, y=y_ext)  # field 2
        pde.set_ic(v_, 0, x=x_ext, y=y_ext)  # field 3
        a_ = pde.new_coef_field(np.nan, x=x_ext, y=y_ext)  # field 4

        # main PDE
        pde.sum_eq0(p_.dt, k_value * u_.dx, k_value * v_.dy)
        pde.sum_eq0(u_.dt, a_ * p_.dx)
        pde.sum_eq0(v_.dt, a_ * p_.dy)

        # solid wall Neumann BC
        sdf = np.maximum(x_ext, y_ext)  # LD boundary
        boundary = pde.new_domain(sdf, x=x_ext, y=y_ext)  # field 5
        pde.bc_sum_eq0(boundary, u_ * domain.dx, v_ * domain.dy)

        # open (absorbing Mur) BC
        sdf = np.maximum(1 - x_ext, 1 - y_ext)  # RU boundary
        boundary = pde.new_domain(sdf, x=x_ext, y=y_ext)  # field 6
        b_ = pde.new_coef_field(np.nan, x=x_ext, y=y_ext)  # field 7
        bc_sum_list = [u_.dx, v_.dy] + pde.dn_sum_list(p_, domain, coef=b_)
        pde.bc_sum_eq0(boundary, bc_sum_list)
        return pde


@register_pde_type("fake")
class Fake2DInputDataset(CartesianGridInputFileDataset):
    r"""Fake dataset to test training performance."""
    n_vars: int = 1
    var_latex = "u"
    dataset_size: int = 10**6
    pde_latex = r"$u_t-\Delta u+u=0$"
    coef_dict = {}
    # NODE_FUNC_IDX = (0,)  # IC of u

    def __init__(self, config: DictConfig, pde_param: Any) -> None:
        super().__init__(config, pde_param)
        n_t = pde_param

        # spatio-temporal coordinates
        t_coord = np.linspace(0, 1, n_t + 1)[1:]
        x_coord = np.linspace(0, 1, 129)[:-1]
        y_coord = np.linspace(0, 1, 129)[:-1]

        # pde_dag
        pde = self._gen_pde_nodes(x_coord, y_coord)
        self.pde_dag = pde.gen_dag(config)
        self.txyz_coord = self._gen_coords(t_coord, x_coord, y_coord)

        # fake dataset
        self.u_label = np.random.randn(n_t, 128, 128, 1, 1)

    def __getitem__(self, idx_pde: int) -> Tuple[NDArray[float]]:
        input_field = self.u_label[0, ..., 0]  # [n_x, n_y, 1]
        return input_field, EMPTY_SCALAR, self.txyz_coord, self.u_label

    @staticmethod
    def _gen_pde_nodes(x_coord: NDArray[float],
                       y_coord: NDArray[float]) -> PDENodesCollector:
        r"""Generate nodes of a PDE for DAG construction"""
        pde = PDENodesCollector()
        x_ext, y_ext = np.meshgrid(x_coord, y_coord, indexing="ij")
        u_ = pde.new_uf()
        pde.set_ic(u_, np.nan, x=x_ext, y=y_ext)

        pde.sum_eq0(u_.dt, -(u_.dx.dx + u_.dy.dy), u_)
        return pde


@register_pde_type("mv_fake")
class MultiVarFake2DInputDataset(CartesianGridInputFileDataset):
    r"""Fake dataset to validate grammar."""
    var_latex = "uvwabcdef"
    dataset_size: int = 10**6
    pde_latex = ""
    coef_dict = {}

    def __init__(self, config: DictConfig, pde_param: Any) -> None:
        super().__init__(config, pde_param)
        self.n_vars = pde_param

        # spatio-temporal coordinates
        t_coord = 0.
        x_coord = np.linspace(0, 1, 129)[:-1]
        y_coord = np.linspace(0, 1, 129)[:-1]

        # pde_dag
        pde = self._gen_pde_nodes(x_coord, y_coord, self.n_vars)
        self.pde_dag = pde.gen_dag(config)
        self.txyz_coord = self._gen_coords(t_coord, x_coord, y_coord)

        # fake dataset
        self.u_label = np.random.randn(1, 128, 128, 1, self.n_vars)

    def __getitem__(self, idx_pde: int) -> Tuple[NDArray[float]]:
        input_field = self.u_label[0, :, :, 0]  # [n_x, n_y, n_vars]
        return input_field, EMPTY_SCALAR, self.txyz_coord, self.u_label

    @staticmethod
    def _gen_pde_nodes(x_coord: NDArray[float],
                       y_coord: NDArray[float],
                       n_vars: int) -> PDENodesCollector:
        r"""Generate nodes of a PDE for DAG construction"""
        pde = PDENodesCollector()
        x_ext, y_ext = np.meshgrid(x_coord, y_coord, indexing="ij")
        for _ in range(n_vars):
            u_ = pde.new_uf()
            pde.set_ic(u_, np.nan, x=x_ext, y=y_ext)
        return pde


@register_pde_type("eit_polar_seed")
class ElectricalImpedanceTomographyPolarCoord2D(CartesianGridInputFileDataset):
    r"""
    Load solution data for electrical impedance tomography (EIT). The solution
    is deformed, mapping the polar coordinates to the rectangular ones.
    """
    n_vars: int = 1
    var_latex = "u"
    dataset_size: int = 1000
    pde_latex = (r"$\nabla\cdot(\sigma(r)\nabla u)=0$" + "\n"
                 r"$a(r)\partial_n u+b(r)=0$")
    coef_dict = {}
    # NODE_FUNC_IDX = (1, 2)

    def __init__(self, config: DictConfig, pde_param: int) -> None:
        super().__init__(config, pde_param)

        # main data file
        filepath = f"{pde_param}.hdf5"
        filepath = os.path.join(config.data.path, filepath)
        self.h5_file_u = h5py.File(filepath, mode="r")

        # spatio-temporal coordinates
        x_coord = np.linspace(0, 1, 128)  # actually normalized theta
        y_coord = np.linspace(0, 1, 128)  # actually r

        # pde_dag
        pde = self._gen_pde_nodes(x_coord, y_coord)
        self.pde_dag = pde.gen_dag(config)
        self.txyz_coord = self._gen_coords(0., x_coord, y_coord)

    def __getitem__(self, idx_pde: Union[int, Tuple[int]]) -> Tuple[NDArray[float]]:
        u_label = self.h5_file_u["sol/u"][idx_pde][:1]
        # [n_t=1, n_x, n_y] -> [n_t, n_x, n_y, n_z=1, n_vars=1]
        u_label = u_label[:, :, :, np.newaxis, np.newaxis]

        sigma = self.h5_file_u["coef/Lu/field"][idx_pde]
        beta = self.h5_file_u["coef/bc/outer/beta/field"][idx_pde]/sigma
        # Shape is [n_x, n_y, n_fields=2].
        input_field = np.stack((sigma, beta), axis=-1)
        return input_field, EMPTY_SCALAR, self.txyz_coord, u_label

    @staticmethod
    def _gen_pde_nodes(x_coord: NDArray[float],
                       y_coord: NDArray[float]) -> PDENodesCollector:
        pde = PDENodesCollector(dim=2)
        x_ext, y_ext = np.meshgrid(x_coord, y_coord, indexing="ij")

        # domain
        center = [0.5, 0.5]
        radius = 0.5
        sdf = np.sqrt((x_ext - center[0])**2 + (y_ext - center[1])**2) - radius
        domain = pde.new_domain(sdf, x=x_ext, y=y_ext)  # field 0

        u_ = pde.new_uf(domain)
        sigma = pde.new_coef_field(np.nan, x=x_ext, y=y_ext)  # field 1
        pde.sum_eq0(pde.dx(sigma * u_.dx), pde.dy(sigma * u_.dy))

        # Neumann BC
        b_ = pde.new_coef_field(np.nan, x=x_ext, y=y_ext)  # field 2
        sum_list = [b_] + pde.dn_sum_list(u_, domain)
        boundary = pde.new_domain(np.abs(sdf), x=x_ext, y=y_ext)  # field 3
        pde.bc_sum_eq0(boundary, sum_list)
        return pde


@register_pde_type("eit_seed")
class ElectricalImpedanceTomography2D(ElectricalImpedanceTomographyPolarCoord2D):
    r"""Load solution data for electrical impedance tomography (EIT)."""

    def __init__(self, config: DictConfig, pde_param: str) -> None:
        super().__init__(config, pde_param)
        # Both has shape [1, n_ax1, n_ax2].
        x_coord = self.h5_file_u["coord/x"]
        y_coord = self.h5_file_u["coord/y"]
        txyz_coord = np.stack(np.broadcast_arrays(
            0., x_coord, y_coord, 0.), axis=-1)  # [1, n_ax1, n_ax2, 4]
        # [nt=1, n_ax1, n_ax2, 4] -> [nt, n_ax1, n_ax2, nz=1, 4]
        txyz_coord = np.expand_dims(txyz_coord, axis=0)
        txyz_coord = np.expand_dims(txyz_coord, axis=3)
        self.txyz_coord = txyz_coord.astype(float_dtype)


class VarNuCustomInputDataset(CartesianGridInputFileDataset):
    r"""Load custom 2D Equation data from a specific data file."""
    dataset_size = 1000
    coef_dict = {"nu": np.nan}

    def __init__(self, config: DictConfig, pde_param: str) -> None:
        super().__init__(config, pde_param)

        # main data file
        filepath = pde_param + ".hdf5"
        filepath = os.path.join(config.data.path, filepath)
        self.h5_file_u = h5py.File(filepath, "r")

        # load spatio-temporal coordinates
        t_coord = self.h5_file_u["coord/t"][1:]  # truncated n_t
        x_coord = self.h5_file_u["coord/x"][()]
        y_coord = self.h5_file_u["coord/y"][()]

        # pde_dag
        # For custom dataset, the grid points for the coefficient fields and
        # the solutions may differ. The former always use uniform grids.
        ax_coord = np.linspace(0, 1, 129)[:-1]
        pde = self._gen_pde_nodes(ax_coord, ax_coord, self.h5_file_u["args"])
        self.pde_dag = pde.gen_dag(config)
        # nu_idx is an array with one or two entries, depending on the PDE.
        self.nu_idx, _ = np.nonzero(np.isnan(self.pde_dag.node_scalar))
        self.txyz_coord = self._gen_coords(t_coord, x_coord, y_coord)

    @staticmethod
    @abstractmethod
    def _gen_pde_nodes(x_coord: NDArray[float],
                       y_coord: NDArray[float],
                       h5_args: h5py.Group) -> PDENodesCollector:
        r"""Generate nodes of a PDE for DAG construction"""


@register_pde_type("sinegordon_fname")
class DedalusSineGordonInputDataset(VarNuCustomInputDataset):
    r"""
    Load custom 2D Sine-Gordon Equation data (generated by Dedalus) from a
    specific data file.
    """
    n_vars: int = 1
    var_latex = "u"
    pde_latex = r"$u_{tt}-\nu\Delta u+(-1)sin(u)=0$"
    # NODE_FUNC_IDX = (0, 1)  # IC of u, u_t

    def __getitem__(self, idx_pde: int) -> Tuple[NDArray[float]]:
        # Shape [n_t, n_x, n_y]. n_t truncated since FNO3D input
        # requires n_t to be even.
        u_label = self.h5_file_u["sol/u"][idx_pde, 1:]
        # [n_t, n_x, n_y] -> [n_t, n_x, n_y, 1, n_vars=1]
        u_label = np.expand_dims(u_label, axis=-1)
        u_label = np.expand_dims(u_label, axis=-1)
        nu_value = self.h5_file_u["coef/Lu/value"][idx_pde]  # shape []
        nu_value = np.full(shape=1, fill_value=nu_value)  # [1]
        input_field = np.stack([self.h5_file_u["coef/u_ic"][idx_pde],
                                self.h5_file_u["coef/ut_ic"][idx_pde]],
                               axis=-1)  # [n_x, n_y, n_field=2]
        return input_field, nu_value, self.txyz_coord, u_label

    def get_pde_info(self,
                     idx_pde: int,
                     idx_var: Optional[int] = None) -> Dict[str, Any]:
        data_info = super().get_pde_info(idx_pde, idx_var)
        nu_val = self.h5_file_u["coef/Lu/value"][idx_pde]  # shape []
        data_info["coef_dict"] = {"nu": nu_val}
        return data_info

    @staticmethod
    def _gen_pde_nodes(x_coord: NDArray[float],
                       y_coord: NDArray[float],
                       h5_args: h5py.Group) -> PDENodesCollector:
        r"""Generate nodes of a PDE for DAG construction"""
        pde = PDENodesCollector(dim=2)
        x_ext, y_ext = np.meshgrid(x_coord, y_coord, indexing="ij")

        u_ = pde.new_uf()
        pde.set_ic(u_, np.nan, x=x_ext, y=y_ext)
        pde.set_ic(u_.dt, np.nan, x=x_ext, y=y_ext)
        nu_u = pde.new_coef(np.nan)
        neg_one = pde.new_coef(-1)

        pde.sum_eq0(u_.dt.dt,
                    -(nu_u * (u_.dx.dx + u_.dy.dy)),
                    neg_one * pde.sin(u_))
        return pde


@register_pde_type('sinegordon_t_last')
class DedalusSineGordonTLastInputDataset(DedalusSineGordonInputDataset):
    r"""
    Load custom 2D Sine-Gordon Equation data (generated by Dedalus) from a
    specific data file for last timestep inference.
    """

    def __init__(self, config: DictConfig, pde_param: str) -> None:
        super().__init__(config, pde_param)
        t_coord = self.h5_file_u["coord/t"][-1]  # last timestep
        x_coord = self.h5_file_u["coord/x"][()]
        y_coord = self.h5_file_u["coord/y"][()]
        self.txyz_coord = self._gen_coords(t_coord, x_coord, y_coord)

    def __getitem__(self, idx_pde: int) -> Tuple[NDArray[float]]:
        # Shape [n_t, n_x, n_y]. n_t truncated since FNO3D input
        # requires n_t to be even.
        u_label = self.h5_file_u["sol/u"][idx_pde, -1]
        u_label = np.expand_dims(u_label, 0)
        # [n_t, n_x, n_y] -> [n_t, n_x, n_y, 1, n_vars=1]
        u_label = np.expand_dims(u_label, axis=-1)
        u_label = np.expand_dims(u_label, axis=-1)
        nu_value = self.h5_file_u["coef/Lu/value"][idx_pde]  # shape []
        nu_value = np.full(shape=1, fill_value=nu_value)  # [1]
        input_field = np.stack([self.h5_file_u["coef/u_ic"][idx_pde],
                                self.h5_file_u["coef/ut_ic"][idx_pde]],
                               axis=-1)  # [n_x, n_y, n_field=2]
        return input_field, nu_value, self.txyz_coord, u_label


@register_pde_type("ins_fname")
class DedalusIncompressibleNSInputDataset(VarNuCustomInputDataset):
    r"""
    Load custom 2D Incompressible NS Equation data (generated by Dedalus) from
    a specific data file.
    """
    n_vars: int = 2
    var_latex = "uvp"
    pde_latex = (r"$u_t-\nu(u_{xx}+u_{yy})+s^u(r)+(u^2)_x+(uv)_y+p_x=0$"
                 "\n"
                 r"$v_t-\nu(v_{xx}+v_{yy})+s^v(r)+(uv)_x+(v^2)_y+p_y=0$"
                 "\n"
                 r"$u_x+v_y=0$")
    # NODE_FUNC_IDX = (1, 2, 3, 4)  # IC of u, v; src of u, v

    def __init__(self, config: DictConfig, pde_param: str) -> None:
        super().__init__(config, pde_param)
        if "y_wall" in self.h5_file_u["args"] and self.h5_file_u["args/y_wall"][()]:
            self.pde_latex = self.pde_latex + "\n$u|_{UD}=0,\\ v|_{UD}=0$"

    def __getitem__(self, idx_pde: int) -> Tuple[NDArray[float]]:
        # Shape is [n_t, n_x, n_y, n_vars=2]. Reasons for truncated n_t:
        # (1) continuity of solution when IC not divergence-free,
        # (2) FNO3D input requires n_t to be even.
        u_label = np.stack([self.h5_file_u["sol/u"][idx_pde, 1:],
                            self.h5_file_u["sol/v"][idx_pde, 1:]], axis=-1)
        # [n_t, n_x, n_y, n_vars] -> [n_t, n_x, n_y, 1, n_vars]
        u_label = np.expand_dims(u_label, axis=-2)
        nu_ext = self.h5_file_u["coef/visc/value"][idx_pde]  # shape []
        nu_ext = np.full(shape=2, fill_value=nu_ext)  # [2]
        input_field = np.stack([self.h5_file_u["coef/u_ic"][idx_pde],
                                self.h5_file_u["coef/v_ic"][idx_pde],
                                self.h5_file_u["coef/s_u"][idx_pde],
                                self.h5_file_u["coef/s_v"][idx_pde]],
                               axis=-1)  # [n_x, n_y, n_field=4]
        return input_field, nu_ext, self.txyz_coord, u_label

    def get_pde_info(self,
                     idx_pde: int,
                     idx_var: Optional[int] = None) -> Dict[str, Any]:
        data_info = super().get_pde_info(idx_pde, idx_var)
        nu_val = self.h5_file_u["coef/visc/value"][idx_pde]  # shape []
        data_info["coef_dict"] = {"nu": nu_val}
        return data_info

    @staticmethod
    def _gen_pde_nodes(x_coord: NDArray[float],
                       y_coord: NDArray[float],
                       h5_args: h5py.Group) -> PDENodesCollector:
        pde = PDENodesCollector(dim=2)
        x_ext, y_ext = np.meshgrid(x_coord, y_coord, indexing="ij")

        # check if y_wall is enabled
        if "y_wall" in h5_args:
            y_wall = h5_args["y_wall"][()]
        else:
            y_wall = False

        # domain and variables
        if y_wall:
            sdf = np.maximum(-y_ext, y_ext - 1)
            domain = pde.new_domain(sdf, x=x_ext, y=y_ext)  # field 0
        else:
            _ = pde.new_domain(0, x=x_ext, y=y_ext)  # placeholder for field 0
            domain = None
        u_ = pde.new_uf(domain)
        v_ = pde.new_uf(domain)
        p_ = pde.new_uf(domain)

        # IC and coefs
        pde.set_ic(u_, np.nan, x=x_ext, y=y_ext)  # field 1
        pde.set_ic(v_, np.nan, x=x_ext, y=y_ext)  # field 2
        s_u = pde.new_coef_field(np.nan, x=x_ext, y=y_ext)  # field 3
        s_v = pde.new_coef_field(np.nan, x=x_ext, y=y_ext)  # field 4
        nu_u = pde.new_coef(np.nan)
        nu_v = pde.new_coef(np.nan)

        # main PDE
        pde.sum_eq0(u_.dt,
                    -(nu_u * (u_.dx.dx + u_.dy.dy)),
                    s_u,
                    (u_ * u_).dx,
                    (u_ * v_).dy,
                    p_.dx)
        pde.sum_eq0(v_.dt,
                    -(nu_v * (v_.dx.dx + v_.dy.dy)),
                    s_v,
                    (u_ * v_).dx,
                    (v_ * v_).dy,
                    p_.dy)
        pde.sum_eq0(u_.dx, v_.dy)

        # boundary conditions
        if y_wall:
            boundary = pde.new_domain(np.abs(sdf), x=x_ext, y=y_ext)  # field 5
            pde.bc_sum_eq0(boundary, u_)
            pde.bc_sum_eq0(boundary, v_)
        return pde


@register_pde_type("ins_t_last")
class DedalusIncompressibleNStLastInputDataset(DedalusIncompressibleNSInputDataset):
    r"""
    Load custom 2D Incompressible NS Equation data (generated by Dedalus) from
    a specific data file, ONLY predicting the last timestep.
    """

    def __init__(self, config: DictConfig, pde_param: str) -> None:
        super().__init__(config, pde_param)
        t_coord = self.h5_file_u["coord/t"][-1]  # truncated n_t
        x_coord = self.h5_file_u["coord/x"][()]
        y_coord = self.h5_file_u["coord/y"][()]
        self.txyz_coord = self._gen_coords(t_coord, x_coord, y_coord)

    def __getitem__(self, idx_pde) -> Tuple[NDArray[float]]:
        u_label = np.stack([self.h5_file_u["sol/u"][idx_pde, 1:],
                            self.h5_file_u["sol/v"][idx_pde, 1:]], axis=-1)
        # [n_t, n_x, n_y, n_vars] -> [n_t, n_x, n_y, 1, n_vars]
        u_label = np.expand_dims(u_label, axis=-2)
        u_label = u_label[-1, :, : ,:, :]
        u_label = np.expand_dims(u_label, axis = 0)
        nu_ext = self.h5_file_u["coef/visc/value"][idx_pde]  # shape []
        nu_ext = np.full(shape=2, fill_value=nu_ext)  # [2]
        input_field = np.stack([self.h5_file_u["coef/u_ic"][idx_pde],
                                self.h5_file_u["coef/v_ic"][idx_pde],
                                self.h5_file_u["coef/s_u"][idx_pde],
                                self.h5_file_u["coef/s_v"][idx_pde]],
                               axis=-1)  # [n_x, n_y, n_field=4]
        return input_field, nu_ext, self.txyz_coord, u_label


@register_pde_type("ins_coupled_fname")
class DedalusINSCoupledVarInputDataset(CartesianGridInputFileDataset):
    r"""
    Load custom 2D Incompressible Navier-Stokes Equation with a coupled
    variable (tracer or buoyancy) data (generated by Dedalus) from a specific
    data file.
    """
    dataset_size = 1000
    n_vars: int = 3
    var_latex = "uvbp"
    pde_latex = (r"$u_t-\nu(u_{xx}+u_{yy})+(u^2)_x+(uv)_y+p_x=0$" + "\n"
                 r"$v_t-\nu(v_{xx}+v_{yy})+(uv)_x+(v^2)_y+p_y+(-1)b=0$" + "\n"
                 r"$b_t-D(b_{xx}+b_{yy})+(ub)_x+(vb)_y=0$" + "\n"
                 r"$u_x+v_y=0$")
    # NODE_FUNC_IDX = (1, 2, 3)  # IC for u, v, b

    def __init__(self, config: DictConfig, pde_param: str) -> None:
        super().__init__(config, pde_param)

        # main data file
        filepath = pde_param + ".hdf5"
        filepath = os.path.join(config.data.path, filepath)
        self.h5_file_u = h5py.File(filepath, "r")

        # load spatio-temporal coordinates
        t_coord = self.h5_file_u["coord/t"][1:]  # truncated n_t
        x_coord = self.h5_file_u["coord/x"][()]
        y_coord = self.h5_file_u["coord/y"][()]

        # read args
        self.coef_dict = {r"\nu": self.h5_file_u["args/viscosity"][()],
                          r"D": self.h5_file_u["args/diffusivity"][()],
                          r"\delta b_0": self.h5_file_u["args/delta_b"][()]}
        if not self.h5_file_u["args/buoyancy"][()]:
            self.pde_latex = self.pde_latex.replace("+(-1)b", "")
        if self.h5_file_u["args/y_wall"][()]:
            self.pde_latex = self.pde_latex + "\n" + (
                r"$u|_{UD}=0,\ v|_{UD}=0,\ b|_U=0,\ b|_D=\delta b_0$")

        # pde_dag
        pde = self._gen_pde_nodes(self.coef_dict, self.h5_file_u["args"])
        self.pde_dag = pde.gen_dag(config)
        self.txyz_coord = self._gen_coords(t_coord, x_coord, y_coord)

    def __getitem__(self, idx_pde: int) -> Tuple[NDArray[float]]:
        # Shape is [n_t, n_x, n_y, n_vars=3]. Reasons for truncated n_t:
        # (1) continuity of solution when IC not divergence-free,
        # (2) FNO3D input requires n_t to be even.
        u_label = np.stack([self.h5_file_u["sol/u"][idx_pde, 1:],
                            self.h5_file_u["sol/v"][idx_pde, 1:],
                            self.h5_file_u["sol/b"][idx_pde, 1:]], axis=-1)
        # [n_t, n_x, n_y, n_vars] -> [n_t, n_x, n_y, 1, n_vars]
        u_label = np.expand_dims(u_label, axis=-2)
        # Shape is [n_x, n_y, n_fields=3].
        input_field = np.stack([self.h5_file_u["coef/u_ic"][idx_pde],
                                self.h5_file_u["coef/v_ic"][idx_pde],
                                self.h5_file_u["coef/b_ic"][idx_pde]], axis=-1)
        return input_field, EMPTY_SCALAR, self.txyz_coord, u_label

    @staticmethod
    def _gen_pde_nodes(coef_dict: Dict[str, float],
                       h5_args: h5py.Group) -> PDENodesCollector:
        pde = PDENodesCollector(dim=2)
        # For custom dataset, the grid points for the coefficient fields and
        # the solutions may differ. The former always use uniform grids.
        ax_coord = np.linspace(0, 1, 129)[:-1]
        x_ext, y_ext = np.meshgrid(ax_coord, ax_coord, indexing="ij")

        # domain and variables
        if h5_args["y_wall"][()]:
            sdf = np.maximum(-y_ext, y_ext - 1)
            domain = pde.new_domain(sdf, x=x_ext, y=y_ext)  # field 0
        else:
            _ = pde.new_domain(0, x=x_ext, y=y_ext)  # placeholder for field 0
            domain = None
        u_ = pde.new_uf(domain)
        v_ = pde.new_uf(domain)
        b_ = pde.new_uf(domain)
        p_ = pde.new_uf(domain)

        # IC and coefs
        pde.set_ic(u_, np.nan, x=x_ext, y=y_ext)  # field 1
        pde.set_ic(v_, np.nan, x=x_ext, y=y_ext)  # field 2
        pde.set_ic(b_, np.nan, x=x_ext, y=y_ext)  # field 3

        # main PDE
        pde.sum_eq0(u_.dt,
                    -(coef_dict[r"\nu"] * (u_.dx.dx + u_.dy.dy)),
                    (u_ * u_).dx,
                    (u_ * v_).dy,
                    p_.dx)
        if h5_args["buoyancy"][()]:
            buoyancy = pde.new_coef(-1) * b_
        else:
            buoyancy = 0
        pde.sum_eq0(v_.dt,
                    -(coef_dict[r"\nu"] * (v_.dx.dx + v_.dy.dy)),
                    (u_ * v_).dx,
                    (v_ * v_).dy,
                    p_.dy,
                    buoyancy)
        pde.sum_eq0(b_.dt,
                    -(coef_dict[r"D"] * (b_.dx.dx + b_.dy.dy)),
                    (u_ * b_).dx,
                    (v_ * b_).dy)
        pde.sum_eq0(u_.dx, v_.dy)

        # boundary conditions
        if h5_args["y_wall"][()]:
            bottom = pde.new_domain(y_ext, x=x_ext, y=y_ext)  # field 4
            pde.bc_sum_eq0(bottom, u_)
            pde.bc_sum_eq0(bottom, v_)
            pde.bc_sum_eq0(bottom, b_, -coef_dict[r"\delta b_0"])
            top = pde.new_domain(1 - y_ext, x=x_ext, y=y_ext)  # field 5
            pde.bc_sum_eq0(top, u_)
            pde.bc_sum_eq0(top, v_)
            pde.bc_sum_eq0(top, b_)
        return pde


@register_pde_type("ins_c_convec")
class DedalusINSCoupledConvectionDataset(DedalusINSCoupledVarInputDataset):
    __doc__ = DedalusINSCoupledVarInputDataset.__doc__ + r"""
        The PDE is specified in convection form, and the diffusive term
        uses negative coefficients instead of negation nodes, which are unseen
        during pre-training.
        """

    @staticmethod
    def _gen_pde_nodes(coef_dict: Dict[str, float],
                       h5_args: h5py.Group) -> PDENodesCollector:
        pde = PDENodesCollector(dim=2)
        # For custom dataset, the grid points for the coefficient fields and
        # the solutions may differ. The former always use uniform grids.
        ax_coord = np.linspace(0, 1, 129)[:-1]
        x_ext, y_ext = np.meshgrid(ax_coord, ax_coord, indexing="ij")

        # domain and variables
        if h5_args["y_wall"][()]:
            sdf = np.maximum(-y_ext, y_ext - 1)
            domain = pde.new_domain(sdf, x=x_ext, y=y_ext)  # field 0
        else:
            _ = pde.new_domain(0, x=x_ext, y=y_ext)  # placeholder for field 0
            domain = None
        u_ = pde.new_uf(domain)
        v_ = pde.new_uf(domain)
        b_ = pde.new_uf(domain)
        p_ = pde.new_uf(domain)

        # IC and coefs
        pde.set_ic(u_, np.nan, x=x_ext, y=y_ext)  # field 1
        pde.set_ic(v_, np.nan, x=x_ext, y=y_ext)  # field 2
        pde.set_ic(b_, np.nan, x=x_ext, y=y_ext)  # field 3

        # main PDE
        pde.sum_eq0(u_.dt,
                    (-coef_dict[r"\nu"]) * (u_.dx.dx + u_.dy.dy),  # <0 coef
                    u_ * u_.dx,  # convection form
                    v_ * u_.dy,  # convection form
                    p_.dx)
        if h5_args["buoyancy"][()]:
            buoyancy = pde.new_coef(-1) * b_
        else:
            buoyancy = 0
        pde.sum_eq0(v_.dt,
                    (-coef_dict[r"\nu"]) * (v_.dx.dx + v_.dy.dy),  # <0 coef
                    u_ * v_.dx,  # convection form
                    v_ * v_.dy,  # convection form
                    p_.dy,
                    buoyancy)
        pde.sum_eq0(b_.dt,
                    (-coef_dict[r"D"]) * (b_.dx.dx + b_.dy.dy),  # <0 coef
                    u_ * b_.dx,  # convection form
                    v_ * b_.dy)  # convection form
        pde.sum_eq0(u_.dx, v_.dy)

        # boundary conditions
        if h5_args["y_wall"][()]:
            bottom = pde.new_domain(y_ext, x=x_ext, y=y_ext)  # field 4
            pde.bc_sum_eq0(bottom, u_)
            pde.bc_sum_eq0(bottom, v_)
            pde.bc_sum_eq0(bottom, b_, -coef_dict[r"\delta b_0"])
            top = pde.new_domain(1 - y_ext, x=x_ext, y=y_ext)  # field 5
            pde.bc_sum_eq0(top, u_)
            pde.bc_sum_eq0(top, v_)
            pde.bc_sum_eq0(top, b_)
        return pde


@register_pde_type("ins_c_incomplete")
class DedalusINSCoupledIncompletePDEDataset(DedalusINSCoupledVarInputDataset):
    __doc__ = DedalusINSCoupledVarInputDataset.__doc__ + r"""
        The PDE specified is incomplete, without:
            - the pressure variable $p$,
            - the incompressibility constraint,
            - the diffusive term,
            - the buoyancy effect (if exists),
            - the boundary condition (if exists).
        """

    @staticmethod
    def _gen_pde_nodes(coef_dict: Dict[str, float],
                       h5_args: h5py.Group) -> PDENodesCollector:
        pde = PDENodesCollector()
        ax_coord = np.linspace(0, 1, 129)[:-1]
        x_ext, y_ext = np.meshgrid(ax_coord, ax_coord, indexing="ij")

        # domain and variables
        _ = pde.new_domain(0, x=x_ext, y=y_ext)  # placeholder for field 0
        u_ = pde.new_uf()
        v_ = pde.new_uf()
        b_ = pde.new_uf()

        # IC and coefs
        pde.set_ic(u_, np.nan, x=x_ext, y=y_ext)  # field 1
        pde.set_ic(v_, np.nan, x=x_ext, y=y_ext)  # field 2
        pde.set_ic(b_, np.nan, x=x_ext, y=y_ext)  # field 3

        # main PDE
        pde.sum_eq0(u_.dt, (u_ * u_).dx, (u_ * v_).dy)
        pde.sum_eq0(v_.dt, (u_ * v_).dx, (v_ * v_).dy)
        pde.sum_eq0(b_.dt, (u_ * b_).dx, (v_ * b_).dy)
        return pde


@register_pde_type("ins_c_unknown")
class DedalusINSCoupledUnknownPDEDataset(DedalusINSCoupledVarInputDataset):
    __doc__ = DedalusINSCoupledVarInputDataset.__doc__ + r"""
        The PDE is assumed to be unknown to the model.
        """

    @staticmethod
    def _gen_pde_nodes(coef_dict: Dict[str, float],
                       h5_args: h5py.Group) -> PDENodesCollector:
        pde = PDENodesCollector()
        ax_coord = np.linspace(0, 1, 129)[:-1]
        x_ext, y_ext = np.meshgrid(ax_coord, ax_coord, indexing="ij")

        # domain and variables
        _ = pde.new_domain(0, x=x_ext, y=y_ext)  # placeholder for field 0
        u_ = pde.new_uf()
        v_ = pde.new_uf()
        b_ = pde.new_uf()

        # IC and coefs
        pde.set_ic(u_, np.nan, x=x_ext, y=y_ext)  # field 1
        pde.set_ic(v_, np.nan, x=x_ext, y=y_ext)  # field 2
        pde.set_ic(b_, np.nan, x=x_ext, y=y_ext)  # field 3

        # main PDE
        f_u, f_v, f_b = pde.unknown_func(u_, v_, b_)
        pde.sum_eq0(u_.dt, f_u)
        pde.sum_eq0(v_.dt, f_v)
        pde.sum_eq0(b_.dt, f_b)
        return pde


@register_pde_type("ins_c_wrong")
class DedalusINSCoupledWrongPDEDataset(DedalusINSCoupledVarInputDataset):
    __doc__ = DedalusINSCoupledVarInputDataset.__doc__ + r"""
        The PDE specified is wrong (the Lorenz ODE system).
        """

    @staticmethod
    def _gen_pde_nodes(coef_dict: Dict[str, float],
                       h5_args: h5py.Group) -> PDENodesCollector:
        pde = PDENodesCollector()
        ax_coord = np.linspace(0, 1, 129)[:-1]
        x_ext, y_ext = np.meshgrid(ax_coord, ax_coord, indexing="ij")

        # domain and variables
        _ = pde.new_domain(0, x=x_ext, y=y_ext)  # placeholder for field 0
        u_ = pde.new_uf()
        v_ = pde.new_uf()
        b_ = pde.new_uf()

        # IC and coefs
        pde.set_ic(u_, np.nan, x=x_ext, y=y_ext)  # field 1
        pde.set_ic(v_, np.nan, x=x_ext, y=y_ext)  # field 2
        pde.set_ic(b_, np.nan, x=x_ext, y=y_ext)  # field 3

        # main PDE
        sigma, rho, beta = 10, 8 / 3, 0.8
        pde.sum_eq0(u_.dt, sigma * v_, (-sigma) * u_)
        pde.sum_eq0(v_.dt, rho * u_, (-1) * v_, pde.prod(-1, u_, b_))
        pde.sum_eq0(b_.dt, u_ * v_, (-beta) * b_)
        return pde
