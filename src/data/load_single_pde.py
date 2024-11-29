r"""Loading datasets containing one specific PDE (single_pde), mainly PDEBench datasets."""
import os
from typing import Tuple, Union, List, Dict, Any
from abc import abstractmethod

import h5py
import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig

from .env import float_dtype
from .pde_dag import PDENodesCollector, PDEAsDAG
from .utils_dataload import Dataset, datasets2loader, concat_datasets
from .multi_pde.dataloader import StaticDatasetFakeUpdater


_pde_type_class_dict = {}


def _record_pde_type(*pde_type_all):
    def add_class(cls):
        for pde_type in pde_type_all:
            _pde_type_class_dict[pde_type] = cls
        return cls
    return add_class


def _debug_print_arr(data: NDArray[float], name: str = ""):
    print(f"{name}: shape {data.shape}, "
          f"range [{data.min():.4g}, {data.max():.4g}]")


class PDEInputFileDataset(Dataset):
    r"""Base class for loading the PDE solution data from a file."""
    dataset_size: int
    n_vars: int
    var_latex: Union[str, List[str]]  # length should be equal to n_vars
    t_coord: Union[float, NDArray[float]] = 0
    x_coord: Union[float, NDArray[float]] = 0
    y_coord: Union[float, NDArray[float]] = 0
    z_coord: Union[float, NDArray[float]] = 0
    DATA_COLUMN_NAMES = ["input_field", "u_label"]

    @abstractmethod
    def __init__(self, config: DictConfig, pde_param: Any) -> None:
        super().__init__()

    def __len__(self) -> int:
        return self.dataset_size

    @abstractmethod
    def get_pde_dag_info(self,
                         idx_pde: int,
                         idx_var: int,
                         input_field: NDArray[float]) -> Tuple[NDArray]:
        r"""
        Get the NumPy arrays containing the information of directed acyclic
        graph (DAG) representation of the PDE, which corresponds to the data
        sample indexed `idx_pde`, and the variable (PDE component) indexed
        `idx_var`.

        Args:
            idx_pde (int): PDE data sample index.
            idx_var (int): PDE variable (component) index.
            input_field (NDArray[float]): Input field values for other neural
                operators. (This argument is introduced to avoid loading the
                same data again.)

        Returns:
            node_type (NDArray[int]): Shape [n_node, 1].
            node_scalar (NDArray[float]): Shape [n_scalar_node, 1].
            node_function (NDArray[float]): Shape [n_field_node,
                n_points_function, 1 + SPACE_DIM + 1].
            in_degree (NDArray[int]): Shape [n_node].
            out_degree (NDArray[int]): Shape [n_node].
            attn_bias (NDArray[float]): Shape [n_node, n_node].
            spatial_pos (NDArray[int]): Shape [n_node, n_node].
        """

    @abstractmethod
    def get_pde_latex(self, idx_pde: int) -> Tuple:
        r"""
        Get the LaTeX representation of the PDE as well as a dictionary of the
        scalar-valued PDE coefficients.
        """


class SinglePDEInputFileDataset(PDEInputFileDataset):
    r"""Base class for loading solution data of a single PDE from a file."""
    pde_latex: str
    coef_dict: Dict[str, float]
    pde_dag: PDEAsDAG
    NODE_FUNC_IDX: Union[Tuple[int], slice]

    def get_pde_dag_info(self,
                         idx_pde: int,
                         idx_var: int,
                         input_field: NDArray[float]) -> Tuple[NDArray]:
        pde_dag = self.pde_dag

        # node_function
        n_fields = input_field.shape[-1]
        # [..., n_fields] -> [n_points_function, n_fields]
        # -> [n_fields, n_points_function]
        input_field = input_field.reshape((-1, n_fields)).transpose()
        # [max_n_functions, n_points_function, 1 + SPACE_DIM + 1]
        node_function = np.copy(pde_dag.node_function)
        node_function[self.NODE_FUNC_IDX, :, -1] = input_field

        # specify the variable (with index idx_var) to be considered
        spatial_pos, attn_bias = pde_dag.get_spatial_pos_attn_bias(idx_var)

        return (pde_dag.node_type,  # [n_node, 1]
                pde_dag.node_scalar,  # [n_scalar_node, 1]
                node_function,  # [n_field_node, n_points_function, 5]
                pde_dag.in_degree,  # [n_node]
                pde_dag.out_degree,  # [n_node]
                attn_bias,  # [n_node, n_node]
                spatial_pos,  # [n_node, n_node]
                )

    def get_pde_latex(self, idx_pde: int) -> Tuple:
        return self.pde_latex, self.coef_dict

    @abstractmethod
    def __getitem__(self, idx_pde: int) -> None:
        pass


@_record_pde_type("reacdiff2d_nt0", "fn_nt0", "fitzhugh_nagumo_nt0")
class FN2DInputDataset(SinglePDEInputFileDataset):
    r"""Load 2D reaction-diffusion (Fitzhugh-Nagumo) PDE data from file."""
    n_vars: int = 2
    var_latex = "uv"
    dataset_size: int = 1000
    pde_latex = ("$u_t-D_u(u_{xx}+u_{yy})+T_-u+T_+u^3+k+T_+v=0$\n"
                 "$v_t-D_v(v_{xx}+v_{yy})+T_-u+T_+v=0$\n"
                 r"$\partial_nu|_{\partial\Omega}=0,\ "
                 r"\partial_nv|_{\partial\Omega}=0$")
    NODE_FUNC_IDX = (1, 2)  # 0 for domain SDF; 1,2 for IC of u,v, resp.

    def __init__(self, config: DictConfig, pde_param: int) -> None:
        super().__init__(config, pde_param)

        # main data file
        filepath = os.path.join(config.data.path, "2D_diff-react_NA_NA.h5")
        self.h5_file_u = h5py.File(filepath, "r")

        # load spatio-temporal coordinates
        self.t_start = pde_param
        # ignore the first 't_start' time-steps
        self.t_coord = self.h5_file_u["0000/grid/t"][self.t_start:]
        self.x_coord = self.h5_file_u["0000/grid/x"][:]
        self.y_coord = self.h5_file_u["0000/grid/y"][:]

        # spatio-temporal rescaling: t [T0,T=5] -> [0,1], x,y [-1,1] -> [0,1]
        self.t_coord -= self.t_coord.min()
        t_max = self.t_coord.max()
        self.t_coord /= t_max
        self.x_coord = (1 + self.x_coord) / 2
        self.y_coord = (1 + self.y_coord) / 2
        self.coef_dict = {"D_u": 1e-3 * t_max / 4,
                          "D_v": 5e-3 * t_max / 4,
                          "k": 5e-3 * t_max,
                          "T_+": t_max,
                          "T_-": -t_max}

        # pde_dag
        pde = self._gen_pde_nodes(self.x_coord, self.y_coord, self.coef_dict)
        self.pde_dag = pde.gen_dag(config)

    def __getitem__(self, idx_pde: int) -> Tuple[NDArray[float]]:
        u_label = self.h5_file_u[f"{idx_pde:04d}/data"][self.t_start:]
        input_field = u_label[0]  # initial condition, [n_x, n_y, n_vars=2]
        # [n_t, n_x, n_y, n_vars] -> [n_t, n_x, n_y, 1, n_vars]
        u_label = np.expand_dims(u_label, axis=-2)
        return input_field, u_label

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


@_record_pde_type("shallow_water", "swe", "rdb", "swe_rdb")
class SWE2DInputDataset(SinglePDEInputFileDataset):
    r"""Load 2D shallow-water PDE solution data from file."""
    n_vars: int = 1
    var_latex = "h"
    dataset_size: int = 1000
    pde_latex = ("$h_t+(hu)_x+(hv)_y=0$\n"
                 "$u_t+uu_x+vu_y+g_1h_x=0$\n"
                 "$v_t+uv_x+vv_y+g_2h_y=0$")
    NODE_FUNC_IDX = (0,)  # IC of h

    def __init__(self, config: DictConfig, pde_param: Any) -> None:
        # pde_param is not used
        super().__init__(config, pde_param)

        # main data file
        filepath = os.path.join(config.data.path, "2D_rdb_NA_NA.h5")
        self.h5_file_u = h5py.File(filepath, "r")

        # load spatio-temporal coordinates
        self.t_coord = self.h5_file_u["0000/grid/t"][:]
        self.x_coord = self.h5_file_u["0000/grid/x"][:]
        self.y_coord = self.h5_file_u["0000/grid/y"][:]

        # spatio-temporal rescaling, xy [-2.5,2.5] -> [0,1]
        self.x_coord = 0.5 + self.x_coord / 5
        self.y_coord = 0.5 + self.y_coord / 5
        # We shall also rescale u,v as u' = u / 5, so that the resulting PDE
        # preserves the convective form without additional coefficients.
        grav = 1.0 / 5**2
        self.coef_dict = {"g_1": grav, "g_2": grav}
        # Note: In dedalus_v5.1 data, SWE equation has $g_1,g_2\in[0.1,1]$, and
        # the current case $g_1=g_2=0.04$ is out-of-distribution (OoD).

        # pde_dag
        pde = self._gen_pde_nodes(self.x_coord, self.y_coord, self.coef_dict)
        self.pde_dag = pde.gen_dag(config)

    def __getitem__(self, idx_pde: int) -> Tuple[NDArray[float]]:
        u_label = self.h5_file_u[f"{idx_pde:04d}/data"][()]  # h component
        input_field = u_label[0]  # initial condition, [n_x, n_y, n_vars=1]
        # [n_t, n_x, n_y, n_vars] -> [n_t, n_x, n_y, 1, n_vars]
        u_label = np.expand_dims(u_label, axis=-2)
        return input_field, u_label

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


@_record_pde_type("darcy_beta", "darcyflow_beta")
class DarcyFlow2DInputDataset(SinglePDEInputFileDataset):
    r"""Load 2D Darcy-flow (time-independent) PDE solution data from file."""
    n_vars: int = 1
    var_latex = "u"
    dataset_size: int = 10000
    pde_latex = ("$(a(x)u_x)_x+(a(x)u_y)_y+\\beta=0$\n"
                 r"u|_{\partial\Omega}=0")
    NODE_FUNC_IDX = (1,)  # 0 for domain SDF, 1 for a, 2 for boundary SDF

    def __init__(self, config: DictConfig, pde_param: float) -> None:
        super().__init__(config, pde_param)

        # main data file
        beta = float(pde_param)
        filepath = f"2D_DarcyFlow_beta{beta}_Train.hdf5"
        filepath = os.path.join(config.data.path, filepath)
        self.h5_file_u = h5py.File(filepath, "r")
        self.coef_dict = {r"\beta": beta}

        # spatio-temporal coordinates
        self.x_coord = self.h5_file_u["x-coordinate"][:]
        self.y_coord = self.h5_file_u["y-coordinate"][:]

        # pde_dag
        pde = self._gen_pde_nodes(beta, self.x_coord, self.y_coord)
        self.pde_dag = pde.gen_dag(config)

    def __getitem__(self, idx_pde: int) -> Tuple[NDArray[float]]:
        u_label = self.h5_file_u["tensor"][idx_pde]
        # [1, 128, 128] -> [1, 128, 128, 1, 1]
        u_label = u_label[:, :, :, np.newaxis, np.newaxis]
        input_field = self.h5_file_u["nu"][idx_pde]
        # [128, 128] -> [128, 128, 1]
        input_field = input_field[:, :, np.newaxis]
        return input_field, u_label

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


@_record_pde_type("piececonst", "fno_darcyflow")
class FNODarcyFlow2DInputDataset(SinglePDEInputFileDataset):
    r"""Load 2D Darcy-flow (time-independent) PDE solution data from file."""
    n_vars: int = 1
    var_latex = "u"
    dataset_size: int = 1024
    pde_latex = ("$(a(x)u_x)_x+(a(x)u_y)_y=f$\n"
                 r"u|_{\partial\Omega}=0")
    NODE_FUNC_IDX = (1,)  # 0 for domain SDF, 1 for a, 2 for boundary SDF

    def __init__(self, config: DictConfig, pde_param: int) -> None:
        super().__init__(config, pde_param)

        # main data file
        filepath = f"piececonst_seed{pde_param}.h5"
        filepath = os.path.join(config.data.path, filepath)
        self.h5_file_u = h5py.File(filepath, mode='r')
        f = 1
        self.coef_dict = {"f": f}

        # spatio-temporal coordinates
        self.x_coord = np.linspace(0, 1, 129)[:-1]
        self.y_coord = np.linspace(0, 1, 129)[:-1]

        # pde_dag
        pde = self._gen_pde_nodes(f, self.x_coord, self.y_coord)
        self.pde_dag = pde.gen_dag(config)

    def __getitem__(self, idx_pde: int) -> Tuple[NDArray[float]]:
        # because $a\in{3, 12}$ and $u\sim0.01$, rescale to make both around 0.1
        u_label = self.h5_file_u["u"][idx_pde] * 10  # rescale
        # [128, 128] -> [1, 128, 128, 1, 1]
        u_label = u_label.reshape((1, 128, 128, 1, 1))

        input_field = self.h5_file_u["a"][idx_pde] * 0.1  # rescale
        # [128, 128] -> [128, 128, 1]
        input_field = input_field.reshape((128, 128, 1))
        return input_field, u_label

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


@_record_pde_type("ns_v", "fno_ins_mlgV")
class FNOIncompressibleNSInputDataset(SinglePDEInputFileDataset):
    r"""Load 2D NS Equation (Vorticity Form) PDE solution data from file."""
    n_vars: int = 1
    var_latex = "w"
    dataset_size: int = 1200
    pde_latex = ("$w_t+uw_x+vw_y=\\nu(w_xx+w_yy)+f$\n"
                 "$u_x+v_y=0$\n"
                 "$w=v_x-u_y$")
    NODE_FUNC_IDX = (0,)  # IC of w

    def __init__(self, config: DictConfig, pde_param: int) -> None:
        super().__init__(config, pde_param)

        # main data file
        nv = 10 ** (-pde_param)
        # possible values: (1e-3, 50), (1e-4, 50), (1e-5, 20)
        if nv == 1e-5:
            max_time = 20
        else:
            max_time = 50
        filepath = f"ns_V1e-0{pde_param}.h5"
        filepath = os.path.join(config.data.path, filepath)
        self.h5_file_u = h5py.File(filepath, mode='r')

        # spatio-temporal coordinates
        self.t_coord = np.linspace(0, 1, 101)[1:]
        self.x_coord = np.linspace(0, 1, 129)[:-1]
        self.y_coord = np.linspace(0, 1, 129)[:-1]

        # rescale T to [0, 1]
        self.coef_dict = {"nv": nv, '1/T': 1/max_time}

        # pde_dag
        pde = self._gen_pde_nodes(self.x_coord, self.y_coord, self.coef_dict)
        self.pde_dag = pde.gen_dag(config)

    def __getitem__(self, idx_pde: int) -> Tuple[NDArray[float]]:
        # [n, n_t, n_x, n_y]
        u_label = self.h5_file_u["u"][idx_pde]
        # initial condition, [n_t, n_x, n_y] -> [n_x, n_y, n_vars=1]
        input_field = u_label[0, :, :, np.newaxis]
        # [n_t, n_x, n_y] -> [n_t - 1, n_x, n_y, 1, n_vars]
        u_label = u_label[1:, :, :, np.newaxis, np.newaxis]
        return input_field, u_label

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
        f = pde.new_coef_field(force, x=x_ext, y=y_ext)

        t_max = pde.new_coef(coef_dict["1/T"])
        nv = pde.new_coef(coef_dict["nv"])

        pde.sum_eq0(t_max * w_.dt,
                    u_ * w_.dx,
                    v_ * w_.dy,
                    -(nv * (w_.dx.dx + w_.dy.dy)),
                    f)
        pde.sum_eq0(u_.dx, v_.dy)
        pde.sum_eq0(w_, u_.dy, -v_.dx)
        return pde


@_record_pde_type("fake")
class Fake2DInputDataset(SinglePDEInputFileDataset):
    r"""Fake dataset to test training performance."""
    n_vars: int = 1
    var_latex = "u"
    dataset_size: int = 10**6
    pde_latex = r"$u_t-\Delta u+u=0$"
    coef_dict = {}
    NODE_FUNC_IDX = (0,)  # IC of u

    def __init__(self, config: DictConfig, pde_param: Any) -> None:
        super().__init__(config, pde_param)

        # spatio-temporal coordinates
        self.t_coord = np.linspace(0, 1, 101)
        self.x_coord = np.linspace(0, 1, 129)[:-1]
        self.y_coord = np.linspace(0, 1, 129)[:-1]

        # pde_dag
        pde = self._gen_pde_nodes(self.x_coord, self.y_coord)
        self.pde_dag = pde.gen_dag(config)

        # fake dataset
        self.u_label = np.random.randn(101, 128, 128, 1, 1)

    def __getitem__(self, idx_pde: int) -> Tuple[NDArray[float]]:
        input_field = self.u_label[0, ..., 0]  # [n_x, n_y, 1]
        return input_field, self.u_label

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


class PDEOutputDataset(Dataset):
    r"""Base class for loading the PDE solution data for different models."""
    DATA_COLUMN_NAMES = ["input_field", "u_label"]

    def __init__(self,
                 config: DictConfig,
                 input_dataset: PDEInputFileDataset,
                 n_samples: int,
                 test: bool = False,
                 deterministic: bool = False) -> None:
        super().__init__()
        self.input_dataset = input_dataset
        self.n_samples = n_samples
        self.test = test
        # self.input_pde_param = config.data.single_pde.get("input_pde_param", False)
        if deterministic:
            # self.data_augment = False
            self.num_txyz_samp_pts = -1
        else:
            # self.data_augment = config.data.augment
            self.num_txyz_samp_pts = config.train.num_txyz_samp_pts

        # spatial-temporal coordinates
        self.txyz_coord = np.stack(np.meshgrid(
            input_dataset.t_coord,
            input_dataset.x_coord,
            input_dataset.y_coord,
            input_dataset.z_coord,
            copy=False,
            indexing="ij",
        ), axis=-1).astype(float_dtype)  # [n_t, n_x, n_y, n_z, 4]

    def __getitem__(self, idx_pde: int) -> Tuple[NDArray]:
        if self.test:
            idx_pde = len(self.input_dataset) - 1 - idx_pde
        input_field, u_label = self.input_dataset[idx_pde]
        return input_field.astype(float_dtype), u_label.astype(float_dtype)

    def __len__(self) -> int:
        return self.n_samples

    def get_pde_info(self, idx_data: int) -> Dict[str, Any]:
        r"""Get a dictionary containing the information of the current PDE."""
        idx_pde = idx_data
        if self.test:
            idx_pde = len(self.input_dataset) - 1 - idx_pde
        n_t, n_x, n_y, n_z, _ = self.txyz_coord.shape
        pde_latex, coef_dict = self.input_dataset.get_pde_latex(idx_pde)
        var_latex = ",".join(self.input_dataset.var_latex)  # eg. "u,v,p"
        data_info = {"pde_latex": pde_latex, "coef_dict": coef_dict,
                     "idx_pde": idx_pde, "idx_var": -1,
                     "var_latex": var_latex,
                     "n_t_grid": n_t, "n_x_grid": n_x,
                     "n_y_grid": n_y, "n_z_grid": n_z}
        return data_info


class FNO3DModelPDEDataset(PDEOutputDataset):
    r"""Base class for loading the PDE solution data for FNO 3D."""
    DATA_COLUMN_NAMES = ["grid_in", "coordinate", "u_label"]

    def __getitem__(self, idx_pde: int) -> Tuple[NDArray[float]]:
        input_field, u_label = super().__getitem__(idx_pde)

        # [n_x, n_y, n_fields] -> [1, n_x, n_y, n_fields]
        input_field = np.expand_dims(input_field, axis=0)
        # [n_t, n_x, n_y, n_z, n_vars] -> [n_t, n_x, n_y, n_vars]
        u_label = u_label[..., 0, :]
        n_t_grid = u_label.shape[0]
        # [1, n_x, n_y, n_fields] -> [n_t, n_x, n_y, n_fields]
        grid_in = np.repeat(input_field, n_t_grid, axis=0)
        coordinate = self.txyz_coord
        return grid_in, coordinate, u_label


class INRModelPDEDataset(PDEOutputDataset):
    r"""Base class for loading the PDE solution data for INRs."""
    DATA_COLUMN_NAMES = ["input_field", "coordinate", "u_label"]

    def __getitem__(self, idx_pde: int) -> Tuple[NDArray[float]]:
        input_field, u_label = super().__getitem__(idx_pde)

        # subsampled coordinate, u_label
        # shape is [n_t_grid, n_x_grid, n_y_grid, n_z_grid, 4 + n_vars]
        txyz_u = np.concatenate([self.txyz_coord, u_label], axis=-1)
        txyz_u = txyz_u.reshape((-1, txyz_u.shape[-1]))
        if self.num_txyz_samp_pts > 0:
            num_txyz_pts = txyz_u.shape[0]
            txyz_sample_idx = np.random.randint(
                0, num_txyz_pts, self.num_txyz_samp_pts)
            txyz_u = txyz_u[txyz_sample_idx, :]
        coordinate = txyz_u[:, :4]  # [n_txyz_pts, 4]
        u_label = txyz_u[:, 4:]  # [n_txyz_pts, n_vars]

        return input_field, coordinate, u_label


class DeepONetPDEDataset(INRModelPDEDataset):
    r"""Loading PDE dataset for DeepONet."""
    DATA_COLUMN_NAMES = ["branch_in", "trunk_in", "u_label"]

    def __getitem__(self, idx_pde: int) -> Tuple[NDArray[float]]:
        input_field, coordinate, u_label = super().__getitem__(idx_pde)
        trunk_in = coordinate
        branch_in = input_field.ravel()
        return branch_in, trunk_in, u_label


class PDEformerPDEDataset(INRModelPDEDataset):
    r"""Loading PDE dataset for PDEformer."""
    DATA_COLUMN_NAMES = ['node_type', 'node_scalar', 'node_function',
                         'in_degree', 'out_degree', 'attn_bias',
                         'spatial_pos', 'coordinate', 'u_label']

    def __getitem__(self, idx_data: int) -> Tuple[NDArray[float]]:
        idx_pde, idx_var = divmod(idx_data, self.input_dataset.n_vars)

        input_field, coordinate, u_label = super().__getitem__(idx_pde)
        dag_tuple = self.input_dataset.get_pde_dag_info(
            idx_pde, idx_var, input_field)
        ui_label = u_label[:, [idx_var]]  # [n_txyz_pts, 1]

        data_tuple = (*dag_tuple, coordinate, ui_label)
        return data_tuple

    def __len__(self):
        return self.n_samples * self.input_dataset.n_vars

    def get_pde_info(self, idx_data: int) -> Dict[str, Any]:
        idx_pde, idx_var = divmod(idx_data, self.input_dataset.n_vars)
        if self.test:
            idx_pde = len(self.input_dataset) - 1 - idx_pde
        n_t, n_x, n_y, n_z, _ = self.txyz_coord.shape
        pde_latex, coef_dict = self.input_dataset.get_pde_latex(idx_pde)
        var_latex = self.input_dataset.var_latex[idx_var]
        data_info = {"pde_latex": pde_latex, "coef_dict": coef_dict,
                     "idx_pde": idx_pde, "idx_var": idx_var,
                     "var_latex": var_latex,
                     "n_t_grid": n_t, "n_x_grid": n_x,
                     "n_y_grid": n_y, "n_z_grid": n_z}
        return data_info


def get_dataset(config: DictConfig,
                pde_type: str,
                pde_param: Union[float, List[float]],
                n_samples: int,
                test: bool,
                deterministic: bool) -> Dataset:
    r"""Obtain PDE solution dataset for the current network model."""
    # input dataset (file handling)
    input_dataset_cls = _pde_type_class_dict[pde_type]
    input_dataset = input_dataset_cls(config, pde_param)

    # output dataset (model-specific)
    model_type = config.model_type.lower()
    if model_type in ["pdeformer", "pdeformer-1", "pf"]:
        data_cls = PDEformerPDEDataset
    elif model_type == "deeponet":
        data_cls = DeepONetPDEDataset
    elif model_type == "fno3d":
        data_cls = FNO3DModelPDEDataset
    else:
        raise NotImplementedError(f"unknown model_type: {model_type}")

    return data_cls(config, input_dataset, n_samples, test, deterministic)


def gen_loader_dict(config: DictConfig,
                    n_samples: int,
                    pde_param_list: Union[List[float], List[List[float]]],
                    batch_size: int,
                    test: bool = False) -> Dict[str, Dict[str, Tuple]]:
    r"""
    Generate a dictionary containing the dataloaders (`BatchDataset` class
    objects in MindSpore) for the training or testing datasets.
    """
    deterministic = True
    shuffle = not deterministic
    pde_type = config.data.single_pde.param_name

    def dataloader_from_param(pde_param):
        dataset = get_dataset(config, pde_type, pde_param,
                              n_samples, test, deterministic)
        dataloader = datasets2loader(
            [dataset], batch_size, shuffle, config.data.num_workers,
            create_iter=True)
        return (dataloader, dataset)

    if config.eval.get("dataset_per_type", -1) >= 0:
        pde_param_list = pde_param_list[:config.eval.dataset_per_type]
    param_loader_dict = {pde_param: dataloader_from_param(pde_param)
                         for pde_param in pde_param_list}
    return {pde_type: param_loader_dict}


class RegularizedFineTuneDataset(Dataset):
    r"""
    To avoid overfitting when fine-tuning PDEformer on small datasets, we
    include the pre-training (multi_pde) dataset during the fine-tuning stage
    as a regularization.
    Each sample in `dataset`, with probability `regularize_ratio`, is replaced
    by a randomly selected sample from `regularize_dataset`.
    """
    DATA_COLUMN_NAMES = ['node_type', 'node_scalar', 'node_function',
                         'in_degree', 'out_degree', 'attn_bias',
                         'spatial_pos', 'coordinate', 'u_label']

    def __init__(self,
                 datasets: List[Dataset],
                 regularize_datasets: List[Dataset],
                 regularize_ratio: float) -> None:
        self.dataset = concat_datasets(datasets)
        self.regularize_dataset = concat_datasets(regularize_datasets)
        self.regularize_ratio = regularize_ratio
        self.num_regularize_data = len(self.regularize_dataset)

    def __getitem__(self, idx_pde: int) -> Tuple[NDArray[float]]:
        if np.random.rand() < self.regularize_ratio:
            idx_reg = np.random.randint(self.num_regularize_data)
            return self.regularize_dataset[idx_reg]
        return self.dataset[idx_pde]

    def __len__(self) -> int:
        return len(self.dataset)


def single_pde_dataset(config: DictConfig) -> Tuple:
    r"""
    Generate dataloaders for the single_pde datasets.

    Args:
        config (DictConfig): Training configurations.

    Returns:
        dataloader_train (BatchDataset): Data loader for the training dataset.
        data_updater (Callable): A function that does nothing.
        train_loader_dict (Dict[str, Dict[str, Tuple]]): A nested
            dictionary containing the data iterator instances of the training
            dataset for the evaluation, in which the random operations (data
            augmentation, spatio-temporal subsampling, etc) are disabled. Here,
            to be more specific, 'Tuple' actually refers to
            'Concatenate[TupleIterator, Dataset]'.
        test_loader_dict (Dict[str, Dict[str, Tuple]]): Similar to
            `train_loader_dict`, but for the testing dataset.
    """
    num_samples_train = config.data.num_samples_per_file.train
    num_samples_test = config.data.num_samples_per_file.test
    train_params = config.data.single_pde.train
    test_params = config.data.single_pde.get("test", train_params)

    train_datasets = [get_dataset(
        config, config.data.single_pde.param_name, pde_param,
        num_samples_train, test=False, deterministic=False,
    ) for pde_param in train_params]
    regularize_ratio = config.data.single_pde.get("regularize_ratio", 0.)
    if regularize_ratio > 0 and config.model_type == "pdeformer":
        # include regularization dataset (custom multi_pde utilized in
        # pre-training) during the fine-tuning stage
        if not config.train.num_txyz_samp_pts > 0:
            raise ValueError("When 'regularize_ratio' is positive, "
                             "'num_txyz_samp_pts' should be positive as well.")
        reg_dataset_dict = StaticDatasetFakeUpdater(
            config,
            config.data.multi_pde.train,
            config.data.num_samples_per_file.regularize
        ).train_dataset_dict

        # {pde_type: [dataset]} -> [dataset]
        regularize_datasets = []
        for dataset_list in reg_dataset_dict.values():
            regularize_datasets.extend(dataset_list)
        finetune_dataset = RegularizedFineTuneDataset(
            train_datasets, regularize_datasets, regularize_ratio)
        train_datasets = [finetune_dataset]
    dataloader_train = datasets2loader(
        train_datasets, config.train.total_batch_size, True,
        config.data.num_workers, create_iter=False)

    train_loader_dict = gen_loader_dict(
        config, num_samples_train, train_params,
        batch_size=config.eval.total_batch_size)
    test_loader_dict = gen_loader_dict(
        config, num_samples_test, test_params,
        batch_size=config.eval.total_batch_size, test=True)

    def data_updater(*_):
        return  # doing nothing
    out_tuple = (dataloader_train, data_updater, train_loader_dict, test_loader_dict)
    return out_tuple
