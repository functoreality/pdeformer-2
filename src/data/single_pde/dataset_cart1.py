r"""
Loading datasets containing one specific PDE (single_pde) with solution
provided on cartesian grid points.
"""
import os
from typing import Tuple, Dict, Any

import h5py
import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig

from ..pde_dag import PDENodesCollector
from .basics import register_pde_type, CartesianGridInputFileDataset, EMPTY_SCALAR


def _debug_print_arr(data: NDArray[float], name: str = ""):
    print(f"{name}: shape {data.shape}, "
          f"range [{data.min():.4g}, {data.max():.4g}]")


@register_pde_type("reacdiff2d_nt0", "fn_nt0", "fitzhugh_nagumo_nt0")
class FN2DInputDataset(CartesianGridInputFileDataset):
    r"""Load 2D reaction-diffusion (Fitzhugh-Nagumo) PDE data from file."""
    n_vars: int = 2
    var_latex = "uv"
    dataset_size: int = 1000
    pde_latex = ("$u_t-D_u(u_{xx}+u_{yy})+T_-u+T_+u^3+k+T_+v=0$\n"
                 "$v_t-D_v(v_{xx}+v_{yy})+T_-u+T_+v=0$\n"
                 r"$\partial_nu|_{\partial\Omega}=0,\ "
                 r"\partial_nv|_{\partial\Omega}=0$")
    # NODE_FUNC_IDX = (1, 2)  # 0 for domain SDF; 1,2 for IC of u,v, resp.

    def __init__(self, config: DictConfig, pde_param: int) -> None:
        super().__init__(config, pde_param)

        # main data file
        filepath = os.path.join(config.data.path, "2D_diff-react_NA_NA.h5")
        self.h5_file_u = h5py.File(filepath, "r")

        # load spatio-temporal coordinates
        self.t_start = pde_param
        # ignore the first 't_start' time-steps
        t_coord = self.h5_file_u["0000/grid/t"][self.t_start:]
        x_coord = self.h5_file_u["0000/grid/x"][:]
        y_coord = self.h5_file_u["0000/grid/y"][:]

        # spatio-temporal rescaling: t [T0,T=5] -> [0,1], x,y [-1,1] -> [0,1]
        t_coord -= t_coord.min()
        t_max = t_coord.max()
        t_coord /= t_max
        x_coord = (1 + x_coord) / 2
        y_coord = (1 + y_coord) / 2
        self.coef_dict = {"D_u": 1e-3 * t_max / 4,
                          "D_v": 5e-3 * t_max / 4,
                          "k": 5e-3 * t_max,
                          "T_+": t_max,
                          "T_-": -t_max}

        # pde_dag
        pde = self._gen_pde_nodes(x_coord, y_coord, self.coef_dict)
        self.pde_dag = pde.gen_dag(config)
        self.txyz_coord = self._gen_coords(t_coord, x_coord, y_coord)

    def __getitem__(self, idx_pde: int) -> Tuple[NDArray[float]]:
        u_label = self.h5_file_u[f"{idx_pde:04d}/data"][self.t_start:]
        input_field = u_label[0]  # initial condition, [n_x, n_y, n_fields=2]
        # [n_t, n_x, n_y, n_vars] -> [n_t, n_x, n_y, 1, n_vars]
        u_label = np.expand_dims(u_label, axis=-2)
        return input_field, EMPTY_SCALAR, self.txyz_coord, u_label

    @staticmethod
    def _gen_pde_nodes(x_coord: NDArray[float],
                       y_coord: NDArray[float],
                       coef_dict: Dict[str, float]) -> PDENodesCollector:
        r"""Generate nodes of a PDE for DAG construction"""
        pde = PDENodesCollector(dim=2)
        x_ext, y_ext = np.meshgrid(x_coord, y_coord, indexing="ij")

        # domain
        sdf = np.stack([-x_ext, x_ext - 1, -y_ext, y_ext - 1]).max(axis=0)
        domain = pde.new_domain(sdf, x=x_ext, y=y_ext)

        # variables
        u_ = pde.new_uf(domain)
        v_ = pde.new_uf(domain)
        pde.set_ic(u_, np.nan, x=x_ext, y=y_ext)
        pde.set_ic(v_, np.nan, x=x_ext, y=y_ext)

        # main PDE
        t_max = pde.new_coef(coef_dict["T_+"])  # create node only once
        t_v = t_max * v_
        neg_t_u = coef_dict["T_-"] * u_
        pde.sum_eq0(u_.dt,
                    -(coef_dict["D_u"] * (u_.dx.dx + u_.dy.dy)),
                    neg_t_u,
                    t_max * u_.cubic,
                    coef_dict["k"],
                    t_v)
        pde.sum_eq0(v_.dt,
                    -(coef_dict["D_v"] * (v_.dx.dx + v_.dy.dy)),
                    neg_t_u,
                    t_v)

        # no-flow Neumann BC
        boundary = pde.new_domain(np.abs(sdf), x=x_ext, y=y_ext)
        pde.bc_sum_eq0(boundary, pde.dn_sum_list(u_, domain))
        pde.bc_sum_eq0(boundary, pde.dn_sum_list(v_, domain))
        return pde


@register_pde_type("shallow_water", "swe", "rdb", "swe_rdb")
class SWE2DInputDataset(CartesianGridInputFileDataset):
    r"""Load 2D shallow-water PDE solution data from file."""
    n_vars: int = 1
    var_latex = "h"
    dataset_size: int = 1000
    pde_latex = ("$h_t+(hu)_x+(hv)_y=0$\n"
                 "$u_t+uu_x+vu_y+g_1h_x=0$\n"
                 "$v_t+uv_x+vv_y+g_2h_y=0$")
    # NODE_FUNC_IDX = (0,)  # IC of h

    def __init__(self, config: DictConfig, pde_param: Any) -> None:
        # pde_param is not used
        super().__init__(config, pde_param)

        # main data file
        filepath = os.path.join(config.data.path, "2D_rdb_NA_NA.h5")
        self.h5_file_u = h5py.File(filepath, "r")

        # load spatio-temporal coordinates
        t_coord = self.h5_file_u["0000/grid/t"][1:]
        x_coord = self.h5_file_u["0000/grid/x"][()]
        y_coord = self.h5_file_u["0000/grid/y"][()]

        # spatio-temporal rescaling, xy [-2.5,2.5] -> [0,1]
        x_coord = 0.5 + x_coord / 5
        y_coord = 0.5 + y_coord / 5
        # We shall also rescale u,v as u' = u / 5, so that the resulting PDE
        # preserves the convective form without additional coefficients.
        grav = 1.0 / 5**2
        self.coef_dict = {"g_1": grav, "g_2": grav}
        # Note: In dedalus_v5.1 data, SWE equation has $g_1,g_2\in[0.1,1]$, and
        # the current case $g_1=g_2=0.04$ is out-of-distribution (OoD).

        # pde_dag
        pde = self._gen_pde_nodes(x_coord, y_coord, self.coef_dict)
        self.pde_dag = pde.gen_dag(config)
        self.txyz_coord = self._gen_coords(t_coord, x_coord, y_coord)

    def __getitem__(self, idx_pde: int) -> Tuple[NDArray[float]]:
        u_label = self.h5_file_u[f"{idx_pde:04d}/data"][()]  # h component
        input_field = u_label[0]  # initial condition, [n_x, n_y, n_fields=1]
        # [n_t, n_x, n_y, n_vars] -> [n_t - 1, n_x, n_y, 1, n_vars]
        u_label = np.expand_dims(u_label[1:], axis=-2)
        return input_field, EMPTY_SCALAR, self.txyz_coord, u_label

    @staticmethod
    def _gen_pde_nodes(x_coord: NDArray[float],
                       y_coord: NDArray[float],
                       coef_dict: Dict[str, float]) -> PDENodesCollector:
        r"""Generate nodes of a PDE for DAG construction"""
        pde = PDENodesCollector(dim=2)
        x_ext, y_ext = np.meshgrid(x_coord, y_coord, indexing="ij")
        h_ = pde.new_uf()
        u_ = pde.new_uf()
        v_ = pde.new_uf()
        pde.set_ic(h_, np.nan, x=x_ext, y=y_ext)
        pde.set_ic(u_, 0, x=x_ext, y=y_ext)
        pde.set_ic(v_, 0, x=x_ext, y=y_ext)

        pde.sum_eq0(h_.dt, (h_ * u_).dx, (h_ * v_).dy)
        pde.sum_eq0(u_.dt, u_ * u_.dx, v_ * u_.dy, coef_dict["g_1"] * h_.dx)
        pde.sum_eq0(v_.dt, u_ * v_.dx, v_ * v_.dy, coef_dict["g_2"] * h_.dy)
        # about BC: According to the data generation code of PDEBench,
        # homogeneous Neumann boundary conditions are applied to $h$ and $hu$
        # (not to $hv$). However, due to the symmetry of the PDE as well as the
        # initial condition (IC), using periodic boundaries along both axes
        # would produce the same result, and hence we may omit the BC in the
        # DAG construction.
        return pde


@register_pde_type("darcy_beta", "darcyflow_beta")
class DarcyFlow2DInputDataset(CartesianGridInputFileDataset):
    r"""Load 2D Darcy-flow (time-independent) PDE solution data from file."""
    n_vars: int = 1
    var_latex = "u"
    dataset_size: int = 10000
    pde_latex = ("$(a(x)u_x)_x+(a(x)u_y)_y+\\beta=0$\n"
                 r"u|_{\partial\Omega}=0")
    # NODE_FUNC_IDX = (1,)  # 0 for domain SDF, 1 for a, 2 for boundary SDF

    def __init__(self, config: DictConfig, pde_param: float) -> None:
        super().__init__(config, pde_param)

        # main data file
        beta = float(pde_param)
        filepath = f"2D_DarcyFlow_beta{beta}_Train.hdf5"
        filepath = os.path.join(config.data.path, filepath)
        self.h5_file_u = h5py.File(filepath, "r")
        self.coef_dict = {r"\beta": beta}

        # spatio-temporal coordinates
        x_coord = self.h5_file_u["x-coordinate"][:]
        y_coord = self.h5_file_u["y-coordinate"][:]

        # pde_dag
        pde = self._gen_pde_nodes(beta, x_coord, y_coord)
        self.pde_dag = pde.gen_dag(config)
        self.txyz_coord = self._gen_coords(0., x_coord, y_coord)

    def __getitem__(self, idx_pde: int) -> Tuple[NDArray[float]]:
        u_label = self.h5_file_u["tensor"][idx_pde]
        # [1, 128, 128] -> [1, 128, 128, 1, 1]
        u_label = u_label[:, :, :, np.newaxis, np.newaxis]
        input_field = self.h5_file_u["nu"][idx_pde]
        # [128, 128] -> [128, 128, 1]
        input_field = input_field[:, :, np.newaxis]
        return input_field, EMPTY_SCALAR, self.txyz_coord, u_label

    @staticmethod
    def _gen_pde_nodes(beta: float,
                       x_coord: NDArray[float],
                       y_coord: NDArray[float]) -> PDENodesCollector:
        r"""Generate nodes of a PDE for DAG construction"""
        pde = PDENodesCollector()
        x_ext, y_ext = np.meshgrid(x_coord, y_coord, indexing="ij")

        # domain
        sdf = np.stack([-x_ext, x_ext - 1, -y_ext, y_ext - 1]).max(axis=0)
        domain = pde.new_domain(sdf, x=x_ext, y=y_ext)

        u_ = pde.new_uf(domain)
        a_ = pde.new_coef_field(np.nan, x=x_ext, y=y_ext)
        pde.sum_eq0(pde.dx(a_ * u_.dx), pde.dy(a_ * u_.dy), beta)

        # Dirichlet BC
        boundary = pde.new_domain(np.abs(sdf), x=x_ext, y=y_ext)
        pde.bc_sum_eq0(boundary, u_)
        return pde


@register_pde_type("piececonst", "fno_darcyflow")
class FNODarcyFlow2DInputDataset(CartesianGridInputFileDataset):
    r"""Load 2D Darcy-flow (time-independent) PDE solution data from file."""
    n_vars: int = 1
    var_latex = "u"
    dataset_size: int = 1024
    pde_latex = ("$(a(x)u_x)_x+(a(x)u_y)_y=f$\n"
                 r"u|_{\partial\Omega}=0")
    # NODE_FUNC_IDX = (1,)  # 0 for domain SDF, 1 for a, 2 for boundary SDF

    def __init__(self, config: DictConfig, pde_param: int) -> None:
        super().__init__(config, pde_param)

        # main data file
        filepath = f"piececonst_seed{pde_param}.h5"
        filepath = os.path.join(config.data.path, filepath)
        self.h5_file_u = h5py.File(filepath, mode='r')
        f = 1
        self.coef_dict = {"f": f}

        # spatio-temporal coordinates
        x_coord = np.linspace(0, 1, 129)[:-1]
        y_coord = np.linspace(0, 1, 129)[:-1]

        # pde_dag
        pde = self._gen_pde_nodes(f, x_coord, y_coord)
        self.pde_dag = pde.gen_dag(config)
        self.txyz_coord = self._gen_coords(0., x_coord, y_coord)

    def __getitem__(self, idx_pde: int) -> Tuple[NDArray[float]]:
        # because $a\in{3, 12}$ and $u\sim0.01$, rescale to make both around 0.1
        u_label = self.h5_file_u["u"][idx_pde] * 10  # rescale
        # [128, 128] -> [1, 128, 128, 1, 1]
        u_label = u_label.reshape((1, 128, 128, 1, 1))

        input_field = self.h5_file_u["a"][idx_pde] * 0.1  # rescale
        # [128, 128] -> [128, 128, 1]
        input_field = input_field.reshape((128, 128, 1))
        return input_field, EMPTY_SCALAR, self.txyz_coord, u_label

    @staticmethod
    def _gen_pde_nodes(f: float,
                       x_coord: NDArray[float],
                       y_coord: NDArray[float]) -> PDENodesCollector:
        r"""Generate nodes of a PDE for DAG construction"""
        pde = PDENodesCollector()
        x_ext, y_ext = np.meshgrid(x_coord, y_coord, indexing="ij")

        # domain
        sdf = np.stack([-x_ext, x_ext - 1, -y_ext, y_ext - 1]).max(axis=0)
        domain = pde.new_domain(sdf, x=x_ext, y=y_ext)

        u_ = pde.new_uf(domain)
        a_ = pde.new_coef_field(np.nan, x=x_ext, y=y_ext)
        pde.sum_eq0(pde.dx(a_ * u_.dx), pde.dy(a_ * u_.dy), -f)

        # Dirichlet BC
        boundary = pde.new_domain(np.abs(sdf), x=x_ext, y=y_ext)
        pde.bc_sum_eq0(boundary, u_)
        return pde


@register_pde_type("ns_v", "fno_ins_mlgV")
class FNOIncompressibleNSInputDataset(CartesianGridInputFileDataset):
    r"""Load 2D NS Equation (Vorticity Form) PDE solution data from file."""
    n_vars: int = 1
    var_latex = "w"
    dataset_size: int = 1000
    pde_latex = ("$w_t+uw_x+vw_y=\\nu(w_xx+w_yy)+f$\n"
                 "$u_x+v_y=0$\n"
                 "$w=v_x-u_y$")
    # NODE_FUNC_IDX = (0,)  # IC of w

    def __init__(self, config: DictConfig, pde_param: int) -> None:
        super().__init__(config, pde_param)

        # main data file
        if pde_param in np.arange(5):
            nv = 1e-3
            max_time = 50
            filepath = f"ns_V1e-3seed{pde_param}.h5"
        elif pde_param in np.arange(5, 15):
            nv = 1e-4
            max_time = 50
            filepath = f"ns_V1e-4seed{pde_param}.h5"
        else:
            nv = 1e-5
            max_time = 20
            filepath = f"ns_V1e-5seed{pde_param}.h5"
        filepath = os.path.join(config.data.path, filepath)
        self.h5_file_u = h5py.File(filepath, mode='r')

        # spatio-temporal coordinates
        t_coord = np.linspace(0, 1, 101)[1:]
        x_coord = np.linspace(0, 1, 129)[:-1]
        y_coord = np.linspace(0, 1, 129)[:-1]

        # rescale T to [0, 1]
        self.coef_dict = {"nv": nv, '1/T': 1/max_time}

        # pde_dag
        pde = self._gen_pde_nodes(x_coord, y_coord, self.coef_dict)
        self.pde_dag = pde.gen_dag(config)
        self.txyz_coord = self._gen_coords(t_coord, x_coord, y_coord)

    def __getitem__(self, idx_pde: int) -> Tuple[NDArray[float]]:
        # [n, n_t, n_x, n_y]
        u_label = self.h5_file_u["u"][idx_pde]
        # initial condition, [n_t, n_x, n_y] -> [n_x, n_y, n_fields=1]
        input_field = u_label[0, :, :, np.newaxis]
        # [n_t, n_x, n_y] -> [n_t - 1, n_x, n_y, 1, n_vars]
        u_label = u_label[1:, :, :, np.newaxis, np.newaxis]
        return input_field, EMPTY_SCALAR, self.txyz_coord, u_label

    @staticmethod
    def _gen_pde_nodes(x_coord: NDArray[float],
                       y_coord: NDArray[float],
                       coef_dict: Dict[str, float]) -> PDENodesCollector:
        r"""Generate nodes of a PDE for DAG construction"""
        pde = PDENodesCollector()
        x_ext, y_ext = np.meshgrid(x_coord, y_coord, indexing="ij")

        w_ = pde.new_uf()
        u_ = pde.new_uf()
        v_ = pde.new_uf()
        pde.set_ic(w_, np.nan, x=x_ext, y=y_ext)

        force = -0.1 * (np.sin(2*np.pi*(x_ext+y_ext)) + np.cos(2*np.pi*(x_ext+y_ext)))
        f_ = pde.new_coef_field(force, x=x_ext, y=y_ext)

        t_max = pde.new_coef(coef_dict["1/T"])
        nv = pde.new_coef(coef_dict["nv"])

        pde.sum_eq0(t_max * w_.dt,
                    u_ * w_.dx,
                    v_ * w_.dy,
                    -(nv * (w_.dx.dx + w_.dy.dy)),
                    f_)
        pde.sum_eq0(u_.dx, v_.dy)
        pde.sum_eq0(w_, u_.dy, -v_.dx)
        return pde


@register_pde_type("ns_pdegym", "pdegym_ins")
class PDEGymIncompNSInputDataset(CartesianGridInputFileDataset):
    r"""
    Load 2D Incompressible NS Equation (Velocity Form) dataset of PDEGym
    from NetCDF file. The order of dimensions in NetCDF file is
    [sample, time, channel, x, y].
    """
    n_vars: int = 2
    var_latex = "uv"
    pde_latex = (r"$u_t+(u^2)_x+(uv)_y+p_x-\nu(u_{xx}+u_{yy})=0$"
                 "\n"
                 r"$v_t+(uv)_x+(v^2)_y+p_y-\nu(v_{xx}+v_{yy})=0$"
                 "\n"
                 r"$u_x+v_y=0$")
    # NODE_FUNC_IDX = (0, 1)  # IC of u, v

    def __init__(self, config: DictConfig, pde_param: str) -> None:
        super().__init__(config, pde_param)

        self.ic_only = pde_param.endswith("-ic")
        if self.ic_only:
            pde_param = pde_param[:-3]

        if pde_param.lower() in ["gauss", "gaussian", "ns-gauss", "ns_gauss"]:
            filepath = os.path.join(config.data.path, "NS-Gauss.nc")
        elif pde_param.lower() in ["sin", "sine", "sines", "ns-sine", "ns_sines",
                                   "ns_sine", "ns_sines"]:
            filepath = os.path.join(config.data.path, "NS-Sines.nc")
        else:
            raise NotImplementedError(f"Unknown PDE parameter: {pde_param}")

        # main netCDF data file
        self.nc_file = h5py.File(filepath, "r")

        self.dataset_size = self.nc_file["sample"].size

        # spatio-temporal coordinates
        if self.ic_only:
            t_coord = 0.
        else:
            t_coord = np.linspace(0, 1, self.nc_file["time"].size)[1:]
        x_coord = np.linspace(0, 1, self.nc_file["x"].size + 1)[:-1]
        y_coord = np.linspace(0, 1, self.nc_file["y"].size + 1)[:-1]

        self.coef_dict = {"nu": 4e-4}  # viscosity, fixed

        # pde_dag
        pde = self._gen_pde_nodes(x_coord, y_coord, self.coef_dict)
        self.pde_dag = pde.gen_dag(config)
        self.txyz_coord = self._gen_coords(t_coord, x_coord, y_coord)

    def __getitem__(self, idx_pde: int) -> Tuple[NDArray[float]]:
        # Shape is [n_t, n_vars=2, n_x, n_y]
        u_label = self.nc_file["velocity"][idx_pde, :, 0:2, :, :]
        # [n_t, n_vars, n_x, n_y] -> [n_t, n_x, n_y, n_vars=2]
        u_label = np.transpose(u_label, (0, 2, 3, 1))
        input_field = u_label[0]  # initial condition, [n_x, n_y, n_fields=2]
        if self.ic_only:
            # [n_t, n_x, n_y, n_vars] -> [1, n_x, n_y, 1, n_vars]
            u_label = np.expand_dims(u_label[:1], axis=-2)
        else:
            # [n_t, n_x, n_y, n_vars] -> [n_t - 1, n_x, n_y, 1, n_vars]
            u_label = np.expand_dims(u_label[1:], axis=-2)
        return input_field, EMPTY_SCALAR, self.txyz_coord, u_label

    @staticmethod
    def _gen_pde_nodes(x_coord: NDArray[float],
                       y_coord: NDArray[float],
                       coef_dict: Dict[str, float]) -> PDENodesCollector:
        r"""Generate nodes of a PDE for DAG construction"""
        pde = PDENodesCollector(dim=2)
        x_ext, y_ext = np.meshgrid(x_coord, y_coord, indexing="ij")

        u_ = pde.new_uf()
        v_ = pde.new_uf()
        p_ = pde.new_uf()
        pde.set_ic(u_, np.nan, x=x_ext, y=y_ext)
        pde.set_ic(v_, np.nan, x=x_ext, y=y_ext)

        pde.sum_eq0(u_.dt,
                    (u_ * u_).dx,
                    (u_ * v_).dy,
                    p_.dx,
                    -(coef_dict["nu"] * (u_.dx.dx + u_.dy.dy)))
        pde.sum_eq0(v_.dt,
                    (u_ * v_).dx,
                    (v_ * v_).dy,
                    p_.dy,
                    -(coef_dict["nu"] * (v_.dx.dx + v_.dy.dy)))
        pde.sum_eq0(u_.dx, v_.dy)
        return pde


@register_pde_type("pdegym_ins_aug_t")
class PDEGymIncompNSAugT0InputDataset(PDEGymIncompNSInputDataset):
    r"""
    Load 2D Incompressible NS Equation (Velocity Form) dataset of PDEGym,
    in which the initial time step is randomly selected.
    NOTE: Only available for INR-based networks such as PDEformer/DeepONet!
    """

    def __getitem__(self, idx_pde: int) -> Tuple[NDArray[float]]:
        # Shape is [n_t, n_vars=2, n_x, n_y]
        u_label = self.nc_file["velocity"][idx_pde, :, 0:2, :, :]
        # [n_t, n_vars, n_x, n_y] -> [n_t, n_x, n_y, n_vars=2]
        u_label = np.transpose(u_label, (0, 2, 3, 1))
        txyz_coord = self.txyz_coord
        if not self.for_eval:  # randomly select init frame
            n_t = u_label.shape[0]
            delta_t = np.random.randint(n_t - 1)
            u_label = u_label[delta_t:]  # [n_t - delta_t, n_x, n_y, 2]
            # Shape is [n_t - 1 - delta_t, n_x, n_y, n_z=1, 4].
            txyz_coord = txyz_coord[delta_t:]
            # Reset time coordinate to make it start from zero.
            txyz_coord[..., 0] -= txyz_coord.flat[0]

        input_field = u_label[0]  # initial condition, [n_x, n_y, n_fields=2]
        # [n_t, n_x, n_y, n_vars] -> [n_t - 1, n_x, n_y, 1, n_vars]
        u_label = np.expand_dims(u_label[1:], axis=-2)
        return input_field, EMPTY_SCALAR, txyz_coord, u_label


@register_pde_type("pdegym_ins_ic")
class PDEGymIncompNSInitCondInputDataset(PDEGymIncompNSInputDataset):
    r"""
    Load 2D Incompressible NS Equation (Velocity Form) dataset of PDEGym,
    but only learning to fit its initial value.
    """

    def __init__(self, config: DictConfig, pde_param: str) -> None:
        if not pde_param.endswith("-ic"):
            pde_param += "-ic"
        super().__init__(config, pde_param)

    @staticmethod
    def _gen_pde_nodes(x_coord: NDArray[float],
                       y_coord: NDArray[float],
                       coef_dict: Dict[str, float]) -> PDENodesCollector:
        pde = PDENodesCollector(dim=2)
        x_ext, y_ext = np.meshgrid(x_coord, y_coord, indexing="ij")

        u_ = pde.new_uf()
        v_ = pde.new_uf()
        pde.set_ic(u_, np.nan, x=x_ext, y=y_ext)
        pde.set_ic(v_, np.nan, x=x_ext, y=y_ext)
        return pde


@register_pde_type("pdegym_ace")
class PDEGymAllenCahnInputDataset(CartesianGridInputFileDataset):
    r"""
    Load 2D Allen-Cahn Equation dataset of PDEgym from the assembled NetCDF
    data file `ACE.nc`.
    """
    n_vars: int = 1
    var_latex = "u"
    pde_latex = r"$u_t-a\Delta u+c_1u+c_2u^3=0$"
    REAC_COEF = 15100  # estimated from dataset; the value 220**2 in the paper seem incorrect
    coef_dict = {"a": 2e-4, "c_1": -2e-4 * REAC_COEF, "c_2": 2e-4 * REAC_COEF}
    # NODE_FUNC_IDX = (0,)  # IC of u

    def __init__(self, config: DictConfig, pde_param: Any) -> None:
        # pde_param is not used
        super().__init__(config, pde_param)

        # main netCDF data file
        filepath = os.path.join(config.data.path, "ACE.nc")
        self.nc_file = h5py.File(filepath, "r")

        self.dataset_size = self.nc_file["sample"].size

        # spatio-temporal coordinates
        t_coord = np.linspace(0, 1, self.nc_file["time"].size)
        x_coord = np.linspace(0, 1, self.nc_file["x"].size + 1)[:-1]
        y_coord = np.linspace(0, 1, self.nc_file["y"].size + 1)[:-1]

        # pde_dag
        pde = self._gen_pde_nodes(x_coord, y_coord, self.coef_dict)
        self.pde_dag = pde.gen_dag(config)
        self.txyz_coord = self._gen_coords(t_coord, x_coord, y_coord)

    def __getitem__(self, idx_pde: int) -> Tuple[NDArray[float]]:
        u_label = self.nc_file["solution"][idx_pde]
        # [n_t, n_x, n_y] -> [n_t, n_x, n_y, n_vars=1]
        u_label = np.expand_dims(u_label, axis=-1)
        input_field = u_label[0]  # initial condition, [n_x, n_y, n_fields=1]
        # [n_t, n_x, n_y, n_vars] -> [n_t, n_x, n_y, n_z=1, n_vars=1]
        u_label = np.expand_dims(u_label, axis=-2)
        return input_field, EMPTY_SCALAR, self.txyz_coord, u_label

    @staticmethod
    def _gen_pde_nodes(x_coord: NDArray[float],
                       y_coord: NDArray[float],
                       coef_dict: Dict[str, float]) -> PDENodesCollector:
        r"""Generate nodes of a PDE for DAG construction"""
        pde = PDENodesCollector(dim=2)
        x_ext, y_ext = np.meshgrid(x_coord, y_coord, indexing="ij")

        u_ = pde.new_uf()
        pde.set_ic(u_, np.nan, x=x_ext, y=y_ext)
        pde.sum_eq0(u_.dt,
                    -(coef_dict["a"] * (u_.dx.dx + u_.dy.dy)),
                    coef_dict["c_1"] * u_,
                    coef_dict["c_2"] * u_.cubic)
        return pde


@register_pde_type("pdegym_wave")
class PDEGymWaveInputDataset(CartesianGridInputFileDataset):
    r"""
    Load 2D Wave Equation dataset of PDEgym from the assembled NetCDF
    data file `Wave-Gauss.nc` or `Wave-Layer.nc`.
    """
    n_vars: int = 1
    var_latex = "u"
    pde_latex = (r"$u_{tt}-c(r)^2\Delta u=0$"
                 "\n"
                 r"$(u_t+c(r)\partial_nu)|_{\partial\Omega}=0$")
    coef_dict = {}
    # NODE_FUNC_IDX = (1, 3, 5)  # IC of u, c^2, c

    def __init__(self, config: DictConfig, pde_param: str) -> None:
        super().__init__(config, pde_param)

        # pde_param
        pde_param = pde_param.lower()
        if pde_param.startswith("layer"):
            filepath = os.path.join(config.data.path, "Wave-Layer.nc")
            # c_scaling is visually estimated from data id=5
            self.c_scaling = 5 * 20 / 128 / 3163
        elif pde_param.startswith("gauss"):
            filepath = os.path.join(config.data.path, "Wave-Gauss.nc")
            # c_scaling is visually estimated from data id=4
            self.c_scaling = 5 * 15 / 128 / 2040
        else:
            raise ValueError(f"Unknown 'pde_param' {pde_param} for 'pdegym_wave'.")
        if pde_param.endswith("-rescale"):
            self.u_scale_fn = lambda x: 6 * x
        elif pde_param.endswith("-norm"):
            self.u_scale_fn = lambda x: (x - 0.34) / 0.12
        elif "-" not in pde_param:
            self.u_scale_fn = lambda x: x
        else:
            raise ValueError(f"Unexpected 'pde_param' {pde_param} for 'pdegym_wave'.")

        # main netCDF data file
        self.nc_file = h5py.File(filepath, "r")
        self.dataset_size = self.nc_file["sample"].size

        # spatio-temporal coordinates
        t_coord = np.linspace(0, 1, self.nc_file["time"].size)
        x_coord = np.linspace(0, 1, self.nc_file["x"].size + 1)[:-1]
        y_coord = np.linspace(0, 1, self.nc_file["y"].size + 1)[:-1]

        # pde_dag
        pde = self._gen_pde_nodes(x_coord, y_coord)
        self.pde_dag = pde.gen_dag(config)
        self.txyz_coord = self._gen_coords(t_coord, x_coord, y_coord)

    def __getitem__(self, idx_pde: int) -> Tuple[NDArray[float]]:
        u_label = self.nc_file["solution"][idx_pde]
        u_label = self.u_scale_fn(u_label)
        c_val = self.c_scaling * self.nc_file["c"][idx_pde]
        # Shape is [n_x, n_y, n_fields=3].
        input_field = np.stack([u_label[0], c_val**2, c_val], axis=-1)
        # [n_t, n_x, n_y] -> [n_t, n_x, n_y, n_z=1, n_vars=1]
        u_label = u_label[:, :, :, np.newaxis, np.newaxis]
        return input_field, EMPTY_SCALAR, self.txyz_coord, u_label

    @staticmethod
    def _gen_pde_nodes(x_coord: NDArray[float],
                       y_coord: NDArray[float]) -> PDENodesCollector:
        r"""Generate nodes of a PDE for DAG construction"""
        pde = PDENodesCollector(dim=2)
        x_ext, y_ext = np.meshgrid(x_coord, y_coord, indexing="ij")

        # domain and variables
        sdf = np.stack([-x_ext, x_ext - 1, -y_ext, y_ext - 1]).max(axis=0)
        domain = pde.new_domain(sdf, x=x_ext, y=y_ext)  # field 0
        u_ = pde.new_uf(domain)

        # main PDE
        pde.set_ic(u_, np.nan, x=x_ext, y=y_ext)  # field 1
        pde.set_ic(u_.dt, 0, x=x_ext, y=y_ext)  # field 2
        c2_ = pde.new_coef_field(np.nan, x=x_ext, y=y_ext)  # field 3
        pde.sum_eq0(u_.dt.dt, -(c2_ * (u_.dx.dx + u_.dy.dy)))

        # absorbing Mur BC
        boundary = pde.new_domain(np.abs(sdf), x=x_ext, y=y_ext)  # field 4
        c_ = pde.new_coef_field(np.nan, x=x_ext, y=y_ext)  # field 5
        bc_sum_list = [u_.dt] + pde.dn_sum_list(u_, domain, coef=c_)
        pde.bc_sum_eq0(boundary, bc_sum_list)
        return pde


@register_pde_type("pdegym_ce_rm")
class PDEGymCERMInputDataset(CartesianGridInputFileDataset):
    r"""
    Load 2D Compressible Euler equation, Richtmeyer-Meshkov problem dataset of
    PDEgym from the assembled NetCDF data file `CE-RM.nc`.
    """
    n_vars: int = 4
    var_latex = [r"\rho", "u", "v", "p", "s"]
    pde_latex = (r"$\rho_t+(\rho u)_x+(\rho v)_y=0$"
                 "\n"
                 r"$(\rho u)_t+(\rho u^2+p)_x+(\rho uv)_y=0$"
                 "\n"
                 r"$(\rho v)_t+(\rho uv)_x+(\rho v^2+p)_y=0$"
                 "\n"
                 r"$E_t+((E+p)u)_x+((E+p)v)_y=0$"
                 "\n"
                 r"$E=0.5(\rho u^2+\rho v^2)+2.5p$")
    coef_dict = {}

    def __init__(self, config: DictConfig, pde_param: Any) -> None:
        super().__init__(config, pde_param)

        # main netCDF data file
        filepath = os.path.join(config.data.path, "CE-RM.nc")
        self.nc_file = h5py.File(filepath, "r")
        self.dataset_size = self.nc_file["sample"].size  # 1260

        # spatio-temporal coordinates
        t_coord = np.linspace(0, 1, self.nc_file["time"].size)[1:]  # [21 - 1]
        x_coord = np.linspace(0, 1, self.nc_file["x"].size + 1)[:-1]  # [128]
        y_coord = np.linspace(0, 1, self.nc_file["y"].size + 1)[:-1]  # [128]

        # pde_dag
        pde = self._gen_pde_nodes(x_coord, y_coord)
        self.pde_dag = pde.gen_dag(config)
        self.txyz_coord = self._gen_coords(t_coord, x_coord, y_coord)

    def __getitem__(self, idx_pde: int) -> Tuple[NDArray[float]]:
        # Shape is [n_t=21, n_vars=4, n_x=128, n_y=128].
        sol_label = self.nc_file["solution"][idx_pde, :, :-1]
        # [n_t, n_vars, n_x, n_y] -> [n_t, n_x, n_y, n_vars]
        sol_label = sol_label.transpose(0, 2, 3, 1)
        input_field = sol_label[0]  # initial condition, [n_x, n_y, n_fields=4]
        # [n_t, n_x, n_y, n_vars] -> [n_t-1, n_x, n_y, n_z=1, n_vars]
        # n_t truncated since FNO3D input requires n_t to be even.
        sol_label = np.expand_dims(sol_label[1:], axis=-2)
        return input_field, EMPTY_SCALAR, self.txyz_coord, sol_label

    @staticmethod
    def _gen_pde_nodes(x_coord: NDArray[float],
                       y_coord: NDArray[float]) -> PDENodesCollector:
        r"""Generate nodes of a PDE for DAG construction"""
        pde = PDENodesCollector()
        x_ext, y_ext = np.meshgrid(x_coord, y_coord, indexing="ij")

        # variables
        rho = pde.new_uf()
        u_ = pde.new_uf()
        v_ = pde.new_uf()
        p_ = pde.new_uf()

        # initial condition
        pde.set_ic(rho, np.nan, x=x_ext, y=y_ext)  # field 0
        pde.set_ic(u_, np.nan, x=x_ext, y=y_ext)  # field 1
        pde.set_ic(v_, np.nan, x=x_ext, y=y_ext)  # field 2
        pde.set_ic(p_, np.nan, x=x_ext, y=y_ext)  # field 3

        # intermediate variables that can be reused
        rho_u = rho * u_
        rho_v = rho * v_
        rho_u_v = pde.prod(rho, u_, v_)
        rho_u2 = rho * u_.square
        rho_v2 = rho * v_.square
        energy = 0.5 * (rho_u2 + rho_v2) + 2.5 * p_
        e_plus_p = energy + p_

        # main PDE
        pde.sum_eq0(rho.dt, rho_u.dx, rho_v.dy)
        pde.sum_eq0(rho_u.dt, (rho_u2 + p_).dx, rho_u_v.dy)
        pde.sum_eq0(rho_v.dt, (rho_v2 + p_).dy, rho_u_v.dx)
        pde.sum_eq0(energy.dt, (e_plus_p * u_).dx, (e_plus_p * v_).dy)
        return pde


@register_pde_type("Eikonal", "eikonal")
class EikonalInputDataset(CartesianGridInputFileDataset):
    r"""Load Eikonal PDE solution data from file."""
    n_vars: int = 1
    var_latex = "u"
    dataset_size: int = 1000
    pde_latex = r"$u_x^2 + u_y^2 + (-1)s^2(r) = 0$"
    coef_dict = {}

    def __init__(self, config: DictConfig, pde_param: Any) -> None:
        super().__init__(config, pde_param)

        filepath = os.path.join(config.data.path, pde_param + '.hdf5')
        self.h5_file_u = h5py.File(filepath, "r")
        self.dataset_size = self.h5_file_u["sol/u"].shape[0]

        # spatio-temporal coordinates
        t_coord = np.array([0.])
        x_coord = np.linspace(0, 1, 128)
        y_coord = np.linspace(0, 1, 128)

        # pde_dag
        pde = self._gen_pde_nodes(x_coord, y_coord)
        self.pde_dag = pde.gen_dag(config)
        self.txyz_coord = self._gen_coords(t_coord, x_coord, y_coord)

    def __getitem__(self, idx_pde: int) -> Tuple[NDArray[float]]:
        u_label = self.h5_file_u["sol/u"][idx_pde]
        source_position = self.h5_file_u["coef/src"][idx_pde]
        velocity = self.h5_file_u["coef/vel"][idx_pde]
        # stack to input field
        input_field = np.stack([source_position, velocity], axis=-1)
        # [n_x, n_y] -> [n_t, n_x, n_y, n_z=1, n_vars=1]
        u_label = u_label[np.newaxis, :, :, np.newaxis, np.newaxis]
        return input_field, EMPTY_SCALAR, self.txyz_coord, u_label

    @staticmethod
    def _gen_pde_nodes(x_coord: NDArray[float],
                       y_coord: NDArray[float]) -> PDENodesCollector:
        r"""Generate nodes of a PDE for DAG construction"""
        pde = PDENodesCollector(dim=2)
        x_ext, y_ext = np.meshgrid(x_coord, y_coord, indexing="ij")

        # domain and variables
        sdf = np.stack([-x_ext, x_ext - 1, -y_ext, y_ext - 1]).max(axis=0)
        domain = pde.new_domain(sdf, x=x_ext, y=y_ext)
        u_ = pde.new_uf(domain)
        s_ = pde.new_coef_field(np.nan, x=x_ext, y=y_ext)
        pde.sum_eq0(u_.dx.square, u_.dy.square, (-1) * s_.square)

        # value zero at source point
        boundary = pde.new_domain(np.nan, x=x_ext, y=y_ext)
        pde.bc_sum_eq0(boundary, u_)
        return pde
