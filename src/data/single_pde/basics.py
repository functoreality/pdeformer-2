r"""Loading datasets containing one specific PDE (single_pde), mainly PDEBench datasets."""
from typing import Tuple, Union, List, Dict, Any, Optional, Callable
from abc import abstractmethod

import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig
from scipy.interpolate import griddata

from ..env import float_dtype
from ..pde_dag import PDEAsDAG
from ..utils_dataload import Dataset


pde_type_class_dict = {}


def register_pde_type(*pde_type_all):
    r"""Register the name of the class for one type of PDE."""
    def add_class(cls):
        for pde_type in pde_type_all:
            if pde_type in pde_type_class_dict:
                raise KeyError(f"pde_type '{pde_type}' already registered!")
            pde_type_class_dict[pde_type] = cls
        return cls
    return add_class


EMPTY_SCALAR = np.empty(0, dtype=float_dtype)


class SinglePDEInputFileDataset(Dataset):
    r"""Base class for loading solution data of a single PDE from a file."""
    dataset_size: int
    n_vars: int
    var_latex: Union[str, List[str]]  # length should be equal to n_vars
    DATA_COLUMN_NAMES = ["input_field", "input_scalar", "coordinates", "u_label"]
    pde_latex: str
    coef_dict: Dict[str, float]
    pde_dag: PDEAsDAG
    node_func_idx: NDArray[int]
    node_scalar_idx: NDArray[int]
    for_eval: bool

    @abstractmethod
    def __init__(self, config: DictConfig, pde_param: Any) -> None:
        super().__init__()

    def __len__(self) -> int:
        return self.dataset_size

    @classmethod
    def preprocess_data(cls,
                        config: DictConfig,
                        pde_param: Any,
                        print_fn: Callable[[str], None] = print) -> None:
        r"""Preprocess data if needed."""
        # nothing to do by default

    def post_init(self, for_eval: bool) -> None:
        r"""
        Complete initialization of the class object.

        This method shall set 'node_func_idx' and 'node_scalar_idx': In the DAG
        representation of the PDE, some fields and scalars are specified at the
        beginning, while others depend on the specific data sample, and have to
        be determined on-the-fly according to 'input_field' and 'input_scalar'.
        We collect the indices of these fields and scalars, and store them in
        'node_func_idx' and 'node_scalar_idx', respectively.
        """
        self.for_eval = for_eval
        self.node_scalar_idx, = np.nonzero(np.isnan(
            self.pde_dag.node_scalar.flat))
        self.node_func_idx, = np.nonzero(np.isnan(
            self.pde_dag.node_function[:, 0, -1]))

    def get_pde_dag_info(self,
                         idx_pde: int,
                         idx_var: int,
                         input_field: NDArray[float],
                         input_scalar: NDArray[float]) -> Tuple[NDArray]:
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
            input_scalar (NDArray[float]): Input scalar values for other neural
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
        pde_dag = self.pde_dag

        # node_function
        n_fields = input_field.shape[-1]
        # [..., n_fields] -> [n_points_function, n_fields]
        # -> [n_fields, n_points_function]
        input_field = input_field.reshape((-1, n_fields)).transpose()
        # [max_n_functions, n_points_function, 1 + SPACE_DIM + 1]
        node_function = np.copy(pde_dag.node_function)
        node_function[self.node_func_idx, :, -1] = input_field

        # node_scalar
        node_scalar = np.copy(pde_dag.node_scalar)  # [n_scalar_node, 1]
        node_scalar[self.node_scalar_idx, 0] = input_scalar

        # specify the variable (with index idx_var) to be considered
        spatial_pos, attn_bias = pde_dag.get_spatial_pos_attn_bias(idx_var)

        return (pde_dag.node_type,  # [n_node, 1]
                node_scalar,  # [n_scalar_node, 1]
                node_function,  # [n_field_node, n_points_function, 5]
                pde_dag.in_degree,  # [n_node]
                pde_dag.out_degree,  # [n_node]
                attn_bias,  # [n_node, n_node]
                spatial_pos,  # [n_node, n_node]
                )

    def get_pde_info(self,
                     idx_pde: int,
                     idx_var: Optional[int] = None) -> Dict[str, Any]:
        r"""Get a dictionary containing the information of the current PDE."""
        if idx_var is None:
            var_latex = ",".join(self.var_latex)  # eg. "u,v,p"
        else:
            var_latex = self.var_latex[idx_var]
        data_info = {"pde_latex": self.pde_latex, "coef_dict": self.coef_dict,
                     "idx_pde": idx_pde, "idx_var": idx_var,
                     "var_latex": var_latex}
        return data_info


class CartesianGridInputFileDataset(SinglePDEInputFileDataset):
    r"""
    Base class for loading solution data of a single PDE from a file. The
    solution label is recorded on cartesian grid points.
    """
    txyz_coord: NDArray[float] # shape [n_t, n_x, n_y, n_z, 4]

    def get_pde_info(self,
                     idx_pde: int,
                     idx_var: Optional[int] = None) -> Dict[str, Any]:
        data_info = super().get_pde_info(idx_pde, idx_var)
        n_t, n_x, n_y, n_z, _ = self.txyz_coord.shape
        data_info.update({"n_t_grid": n_t, "n_x_grid": n_x,
                          "n_y_grid": n_y, "n_z_grid": n_z})
        return data_info

    @staticmethod
    def _gen_coords(t_coord: Union[float, NDArray[float]],
                    x_coord: NDArray[float],
                    y_coord: NDArray[float],
                    z_coord: Union[float, NDArray[float]] = 0.) -> NDArray[float]:
        r"""Get the spatio-temporal coordinates on Cartesian grid points."""
        txyz_coord = np.stack(np.meshgrid(
            t_coord, x_coord, y_coord, z_coord,
            copy=False, indexing="ij",
        ), axis=-1).astype(float_dtype)  # [n_t, n_x, n_y, n_z, 4]
        return txyz_coord


class ScatteredPointsInputFileDataset(SinglePDEInputFileDataset):
    r"""
    Base class for loading solution data of a single PDE from a file. The
    solution label is recorded on irregular scattered points.
    """
    n_t_grid: int
    PLOT_SCATTER_SIZE: float = 1.

    def get_pde_info(self,
                     idx_pde: int,
                     idx_var: Optional[int] = None) -> Dict[str, Any]:
        data_info = super().get_pde_info(idx_pde, idx_var)
        data_info["n_t_grid"] = self.n_t_grid
        data_info["scatter_size"] = self.PLOT_SCATTER_SIZE
        return data_info

    @staticmethod
    def _interpolate2grid(f_values: NDArray[float],
                          xy_coord: NDArray[float],
                          method: str = "linear") -> NDArray[float]:
        r"""
        Interpolate the input function given on scatter points to 128x128
        uniform grids. Using cubic (by default) interpolation inside the convex
        hull of the input points, and extrapolates to the rest of the domain
        according to the nearest point values.

        Input:
            f_values: function values of shape [n_old].
            xy_coord: (x,y) coordinate values of shape [n_old, 2].
            method: Interpolation method, options {"linear", "cubic", "nearest"}.
                Default: "linear", which typically works well if there are
                sufficient scatter points.

        Output:
            f_interp: Interpolated function values of shape [128, 128].
        """
        xy_new = np.mgrid[0:1:128j, 0:1:128j]  # [2, 128, 128]
        # [2, 128, 128] -> [2, 128*128] -> [128*128, 2]
        xy_new = xy_new.reshape(2, -1).transpose()
        f_interp = griddata(xy_coord, f_values, xy_new, method=method)
        nan_flag = np.isnan(f_interp)
        if np.any(nan_flag):  # some points lie outside the convex hull of xy_coord
            f_nearest = griddata(xy_coord, f_values, xy_new, method="nearest")
            f_interp = np.where(nan_flag, f_nearest, f_interp)
        f_interp = f_interp.reshape(128, 128)  # [128*128] -> [128, 128]
        return f_interp
