r"""
Loading datasets containing one specific PDE (single_pde) with solution
provided on scattered points.
"""
import os
import time
from typing import Tuple, Dict, Any, Callable

import h5py
import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig

from ..pde_dag import PDENodesCollector
from .basics import register_pde_type, ScatteredPointsInputFileDataset, EMPTY_SCALAR, float_dtype


@register_pde_type("rigno_wave")
class RIGNOWaveCInputDataset(ScatteredPointsInputFileDataset):
    r"""
    Load 2D Wave Equation dataset on a disk domain (provided by RIGNO) from the
    NetCDF data file `Wave-C-Sines.nc`.
    """
    n_vars: int = 1
    var_latex = "u"
    # Wave velocity c^2=4 reduced to c^2=0.01 after coordinate rescaling (xy/2, t*10).
    pde_latex = (r"$u_{tt}-a\Delta u=0$" + "\n"
                 r"$u|_{\partial\Omega}=0$")
    coef_dict = {"a": 0.01}

    def __init__(self, config: DictConfig, pde_param: float) -> None:
        super().__init__(config, pde_param)
        self.scaling = pde_param

        # main netCDF data file
        filepath = os.path.join(config.data.path, "Wave-C-Sines.nc")
        self.nc_file = h5py.File(filepath, "r")
        # Shape is [1500, 21, 16431, 1].
        self.dataset_size, self.n_t_grid, n_xy, _ = self.nc_file["u"].shape
        self.n_t_grid -= 1  # truncate first frame

        # spatio-temporal coordinates
        r_old = self.nc_file["x"][0, 0]  # [n_xy, 2]
        r_old = (r_old + 0.5) / 2  # rescale coordinates
        # [n_xy, 2] -> [n_t - 1, n_xy, 2]
        xy_ext = np.repeat(r_old[np.newaxis], self.n_t_grid, axis=0)
        t_ext = np.linspace(0, 1, self.n_t_grid + 1)[1:]  # truncate first frame
        t_ext = t_ext[:, np.newaxis, np.newaxis]  # [n_t - 1] -> [n_t - 1, 1, 1]
        t_ext = np.repeat(t_ext, n_xy, axis=1)  # [n_t - 1, n_xy, 1]
        self.txyz_coord = np.concatenate(
            [t_ext, xy_ext, np.zeros_like(t_ext)],
            axis=-1).astype(float_dtype)  # [n_t - 1, n_xy, 4]

        # pde_dag
        pde = self._gen_pde_nodes(self.coef_dict)
        self.pde_dag = pde.gen_dag(config)

        # load interpolated initial conditions to RAM
        interp_path = os.path.join(config.data.path, "preprocess",
                                   "Wave-C-Sines-interp.npy")
        if not os.path.exists(interp_path):
            raise FileNotFoundError(
                f"The file {interp_path} does not exist. Please run the"
                " following command before training (as shown in"
                " scripts/run_distributed_train.sh): \n\n\t"
                "python3 preprocess_data.py -c CONFIG_PATH\n")
        self.ic_interp_all = np.load(interp_path)

    def __getitem__(self, idx_pde: int) -> Tuple[NDArray[float]]:
        u_label = self.nc_file["u"][idx_pde, 1:]  # [n_t - 1, n_xy, n_vars=1]
        # Shape is [128, 128, n_fields=1].
        input_field = self.ic_interp_all[idx_pde]
        u_label = self.scaling * u_label
        input_field = self.scaling * input_field
        return input_field, EMPTY_SCALAR, self.txyz_coord, u_label

    @classmethod
    def preprocess_data(cls,
                        config: DictConfig,
                        pde_param: Any,
                        print_fn: Callable[[str], None] = print) -> None:
        interp_path = os.path.join(config.data.path, "preprocess")
        os.makedirs(interp_path, exist_ok=True)

        # check if preprocessing results already exist
        interp_path = os.path.join(interp_path, "Wave-C-Sines-interp.npy")
        if os.path.exists(interp_path):
            return  # no need to (re)generate preprocessing results

        # main netCDF data file
        filepath = os.path.join(config.data.path, "Wave-C-Sines.nc")
        nc_file = h5py.File(filepath, "r")
        r_old = nc_file["x"][0, 0]  # [n_xy, 2]
        r_old = (r_old + 0.5) / 2  # rescale coordinates

        # interpolate initial condition
        ic_interp_all = []
        dataset_size = nc_file["u"].shape[0]
        time_next = time.time()
        for i in range(dataset_size):
            if time.time() > time_next:
                print_fn(f"interpolating RIGNO wave IC: {i}/{dataset_size}")
                time_next = time.time() + 20
            ic_old = nc_file["u"][i, 0, :, 0]
            ic_interp = cls._interpolate2grid(ic_old, r_old)
            ic_interp_all.append(ic_interp)
        ic_interp_all = np.array(ic_interp_all, dtype=float_dtype)
        # [n_data, 128, 128] -> [n_data, 128, 128, 1]
        ic_interp_all = np.expand_dims(ic_interp_all, axis=-1)
        np.save(interp_path, ic_interp_all)
        print_fn(f"File saved: {interp_path}")

    @staticmethod
    def _gen_pde_nodes(coef_dict: Dict[str, float]) -> PDENodesCollector:
        r"""Generate nodes of a PDE for DAG construction"""
        pde = PDENodesCollector(dim=2)
        x_ext, y_ext = np.mgrid[0:1:128j, 0:1:128j]  # both [128, 128]

        # domain and variables
        sdf = np.sqrt((x_ext - 0.5)**2 + (y_ext - 0.5)**2) - 0.5
        domain = pde.new_domain(sdf, x=x_ext, y=y_ext)  # field 0
        u_ = pde.new_uf(domain)

        # main PDE
        pde.set_ic(u_, np.nan, x=x_ext, y=y_ext)  # field 1
        pde.set_ic(u_.dt, 0, x=x_ext, y=y_ext)  # field 2
        pde.sum_eq0(u_.dt.dt, -(coef_dict["a"] * (u_.dx.dx + u_.dy.dy)))

        # Dirichlet BC
        boundary = pde.new_domain(np.abs(sdf), x=x_ext, y=y_ext)  # field 3
        pde.bc_sum_eq0(boundary, u_)
        return pde
