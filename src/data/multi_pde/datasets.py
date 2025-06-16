r"""Defining custom dataset consisting of multiple PDEs (multi_pde)."""
import os
from typing import Tuple, Dict, Any, Union, Callable, List, Optional
from abc import abstractmethod

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import griddata
import h5py
from omegaconf import DictConfig

from ..env import float_dtype, int_dtype
from ..pde_dag import PDEAsDAG, ModNodeSwapper
from ..utils_dataload import Dataset
from . import pde_types

pde_dataset_cls_dict = {}


def record_pde_dataset(*pde_type_all):
    r"""Register the current class with a specific name."""
    def add_class(cls):
        for pde_type in pde_type_all:
            pde_dataset_cls_dict[pde_type] = cls
        return cls
    return add_class


def get_pde_dataset_cls(pde_type: str) -> type:
    r"""Get the dataset class for a specific PDE type."""
    pde_type = pde_type.split("_")[0]
    return pde_dataset_cls_dict[pde_type]


def _cartesian_txyz_coord(
        *,
        t: Union[float, NDArray[float]] = 0.,
        x: Union[float, NDArray[float]] = 0.,
        y: Union[float, NDArray[float]] = 0.,
        z: Union[float, NDArray[float]] = 0.) -> NDArray[float]:
    r"""Get the coordinate grid in Cartesian form."""
    coord_list = np.meshgrid(t, x, y, z, copy=False, indexing="ij")
    # Shape is [n_t, n_x, n_y, n_z, 4].
    return np.stack(coord_list, axis=-1).astype(float_dtype)


def _sample_coord_tr(txyz_coord: NDArray[float],
                     ui_label: NDArray[float],
                     num_txyz_samp_pts: int) -> Tuple[NDArray[float]]:
    r"""
    Subsampled spatial-temporal coordinate, along with the corresponding
    u_label on data loading.
    """
    # flatten and concat
    ui_label = ui_label.reshape((-1, 1)).astype(float_dtype)
    txyz_coord = txyz_coord.reshape((-1, 4))
    txyz_u = np.concatenate([txyz_coord, ui_label], axis=-1)

    # subsample
    if num_txyz_samp_pts > 0:
        num_txyz_pts = txyz_u.shape[0]
        txyz_sample_idx = np.random.randint(
            0, num_txyz_pts, num_txyz_samp_pts)
        txyz_u = txyz_u[txyz_sample_idx, :]

    # separate subsampled coordinate and ui_label
    coordinate = txyz_u[:, :4]  # [n_txyz_pts, 4]
    ui_label = txyz_u[:, 4:]  # [n_txyz_pts, 1]

    return coordinate, ui_label


def _sample_coord_r(txyz_coord: NDArray[float],
                    ui_label: NDArray[float],
                    num_txyz_samp_pts: int) -> Tuple[NDArray[float]]:
    r"""
    Subsampled coordinate, u_label on data loading for <spatial-INR>.
    Deprecated.
    """
    # [nt, nx, ny] -> [nt, nx * ny] -> [nx * ny, nt]
    ui_label = ui_label.reshape((101, -1)).astype(float_dtype).T
    # [nt, nx, ny, nz, 4] -> [nx, ny, nz, 3] -> [nx * ny * nz, 3]
    xyz_coord = txyz_coord[0, ..., 1:].reshape((-1, 3))

    # subsample
    if num_txyz_samp_pts > 0:
        num_xyz_pts = ui_label.shape[0]
        xyz_sample_idx = np.random.randint(
            0, num_xyz_pts, num_txyz_samp_pts)
        ui_label = ui_label[xyz_sample_idx]  # [n_xyz, nt]
        xyz_coord = xyz_coord[xyz_sample_idx]  # [n_xyz, 3]

    return xyz_coord, ui_label


def _sample_coord_t(txyz_coord: NDArray[float],
                    ui_label: NDArray[float],
                    num_txyz_samp_pts: int) -> Tuple[NDArray[float]]:
    r"""
    Subsampled coordinate, u_label on data loading for <temporal-Dec>.
    Deprecated.
    """
    # [nt, nx, ny] -> [nt, nx, ny, 1]
    ui_label = np.expand_dims(ui_label, axis=-1)
    # [nt, nx, ny, nz, 4] -> [nt, nx, ny, 4]
    txyz_coord = txyz_coord.take(0, axis=-2)

    # subsample
    if num_txyz_samp_pts > 0:
        if num_txyz_samp_pts > 101:
            raise ValueError(
                f"'num_txyz_samp_pts' ({num_txyz_samp_pts}) "
                "too big for temporal-decoders!")
        num_t_pts = ui_label.shape[0]
        t_sample_idx = np.random.randint(0, num_t_pts, num_txyz_samp_pts)
        ui_label = ui_label[t_sample_idx]  # [t_sample, nx, ny, 1]
        txyz_coord = txyz_coord[t_sample_idx]  # [t_sample, nx, ny, 4]

    return txyz_coord, ui_label


def _compute_full_shape(shape_tuple: Tuple[int], total_size: int) -> Tuple[int]:
    r"""Compute the full shape from the shape tuple and the total size."""
    if shape_tuple.count(-1) > 1:
        raise ValueError("Shape tuple can contain at most one -1.")
    if total_size < 0:
        return shape_tuple

    known_size = np.prod([dim for dim in shape_tuple if dim != -1])

    if total_size % known_size != 0:
        raise ValueError("Total size is not divisible by the known size.")

    full_shape = tuple((total_size // known_size if dim == -1 else dim)
                       for dim in shape_tuple)

    return full_shape


def _downsample_array(array: NDArray[float],
                      target_shape: Tuple[int],
                      random_axes: Optional[List[int]] = None) -> NDArray[float]:
    r"""Downsamples the input array."""
    input_shape = array.shape

    # Check if the input shape and target shape are compatible
    if len(input_shape) != len(target_shape):
        raise ValueError("The number of dimensions in the input array does not"
                         " match the target shape.")

    for i, (input_dim, target_dim) in enumerate(zip(input_shape, target_shape)):
        if target_dim > input_dim:
            raise ValueError(f"Target size for axis {i} cannot be greater than"
                             " the input array size.")

    # Handle random sampling for the specified axes
    if random_axes is None:
        random_axes = []

    indices = []
    for i, (input_dim, target_dim) in enumerate(zip(input_shape, target_shape)):
        if target_dim < 0:
            target_dim = input_dim
        if i in random_axes:
            # Random sampling without replacement
            indices.append((np.random.random(target_dim) * input_dim).astype(int))
        else:
            # Evenly spaced sampling with a random starting point
            step = input_dim // target_dim
            max_start = input_dim - step * target_dim
            start = np.random.randint(0, max_start + 1)
            indices.append(np.arange(start, start + step * target_dim, step))

    # Use np.ix_ to create index arrays and perform the downsampling
    indexed_array = array[np.ix_(*indices)]

    return indexed_array


class MultiPDEDatasetBase(Dataset):
    r"""Multi-PDE Dataset to be fed into PDEformer."""
    DATA_COLUMN_NAMES = ['node_type', 'node_scalar', 'node_function',
                         'in_degree', 'out_degree', 'attn_bias',
                         'spatial_pos', 'coordinate', 'u_label']
    n_samples: int
    n_vars: int
    sample_coord: Callable[[NDArray, NDArray, int], Tuple[NDArray]]

    def __init__(self,
                 config: DictConfig,
                 filename: str,  # pylint: disable=unused-argument
                 n_samples: int,
                 test: bool = False,
                 for_eval: bool = False) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.test = test
        self.for_eval = for_eval
        self.disconn_attn_bias = float(config.data.pde_dag.disconn_attn_bias)

        if config.model.get("variant", "inr") == "patched":
            self.sample_coord = _sample_coord_t
            self.num_txyz_samp_pts = 1  # regardless of config and for_eval
        else:  # default case for INR variant
            self.sample_coord = self._sample_coord_default_inr
            if for_eval:
                sub_config = config.eval
                self.num_txyz_samp_pts = sub_config.get('num_txyz_samp_pts', -1)
            else:
                sub_config = config.train
                self.num_txyz_samp_pts = sub_config.get('num_txyz_samp_pts', 8192)
            self.samp_axes = sub_config.get('samp_axes', 'txyz').lower()
            if self.samp_axes == 't':
                shape = tuple(sub_config.get('samp_shape', [-1, 128, 128]))
                if len(shape) not in [3, 4]:
                    raise ValueError(
                        f"Invalid samp_shape {shape}. The shape should be a list"
                        " of 3 integers [n_t, n_x, n_y] or 4 integers"
                        " [n_t, n_x, n_y, n_z].")
                if len(shape) == 3:
                    shape = shape + (1,)  # (n_t, n_x, n_y) -> (n_t, n_x, n_y, 1)
                try:
                    self.samp_shape = _compute_full_shape(
                        shape, self.num_txyz_samp_pts)
                except:
                    raise ValueError(
                        f"The number of sampled points {self.num_txyz_samp_pts}"
                        f" and the samp_shape {shape} are incompatible.")
            elif self.samp_axes == 'xyz':
                shape = tuple(sub_config.get('samp_shape', [101, -1]))
                if len(shape) != 2:
                    raise ValueError(
                        f"Invalid samp_shape {shape}. The shape should be a list"
                        " of 2 integers [n_t, n_xyz].")
                try:
                    self.samp_shape = _compute_full_shape(
                        shape, self.num_txyz_samp_pts)
                except:
                    raise ValueError(
                        f"The number of sampled points {self.num_txyz_samp_pts}"
                        f" and the samp_shape {shape} are incompatible.")
            elif self.samp_axes == 'txyz':
                self.samp_shape = (self.num_txyz_samp_pts,)

    def __getitem__(self, idx_data: int) -> Tuple[NDArray]:
        idx_pde, idx_var = divmod(idx_data, self.n_vars)
        if self.test:
            idx_pde = -idx_pde - 1

        dag_info_tuple = self._get_dag_info(idx_pde, idx_var)
        coordinate = self._get_txyz_coord(idx_pde, idx_var)
        ui_label = self._get_ui_label(idx_pde, idx_var)
        coordinate, ui_label = self.sample_coord(
            coordinate, ui_label, self.num_txyz_samp_pts)
        data_tuple = (*dag_info_tuple, coordinate, ui_label)
        return data_tuple

    def __len__(self) -> int:
        if self.n_samples < 0:
            return self.total_samples * self.n_vars
        if self.n_samples > self.total_samples:
            raise ValueError(
                f"The number of samples {self.n_samples} is greater than the"
                f" total number of samples {self.total_samples}.")
        return self.n_samples * self.n_vars

    def use_datafile(self, filename: str) -> None:
        r"""
        Specify (or reset) the data file used for training. May be executed
        during the training process, i.e. dynamically switch to different data
        files.
        """
        # Nothing to do by default. Please override this method in subclasses.

    def get_pde_info(self, idx_data: int) -> Dict[str, Any]:
        r"""
        Get a dictionary containing the information of the current PDE indexed
        by `idx_data`.
        """
        idx_pde, idx_var = divmod(idx_data, self.n_vars)
        if self.test:
            idx_pde = -idx_pde - 1

        pde_latex, var_latex, coef_dict = self._get_pde_latex(idx_pde, idx_var)
        txyz_coord = self._get_txyz_coord(idx_pde, idx_var)
        n_t, n_x, n_y, n_z, _ = txyz_coord.shape
        data_info = {"pde_latex": pde_latex, "coef_dict": coef_dict,
                     "idx_pde": idx_pde, "idx_var": idx_var,
                     "var_latex": var_latex,
                     "n_t_grid": n_t, "n_x_grid": n_x,
                     "n_y_grid": n_y, "n_z_grid": n_z}
        return data_info

    @property
    def total_samples(self) -> int:
        r"""Total number of samples in the dataset."""
        return self._get_total_samples()

    def _sample_coord_default_inr(self,
                                  txyz_coord: NDArray[float],
                                  ui_label: NDArray[float],
                                  num_txyz_samp_pts: int,
                                  ) -> Tuple[NDArray[float]]:
        r"""Subsampled coordinate, u_label on data loading. Shape of txyz_coord
        is [n_t, ..., 4], and shape of ui_label is [n_t, ...]."""
        if self.samp_axes == "txyz":
            # flatten and concat
            ui_label = ui_label.reshape((-1, 1)).astype(float_dtype)
            txyz_coord = txyz_coord.reshape((-1, 4))
            # shape [n_t * n_xyz, 5]
            txyz_u = np.concatenate([txyz_coord, ui_label], axis=-1)

            # subsample
            if num_txyz_samp_pts > 0:
                shape = self.samp_shape + (5,)
                # shape [n_txyz_pts, 5]
                txyz_u = _downsample_array(txyz_u, shape, random_axes=[0])
        elif self.samp_axes == "t":
            if len(txyz_coord.shape) == 4:
                txyz_coord = np.expand_dims(txyz_coord, axis=-1)
            ui_label = ui_label.reshape((txyz_coord.shape[:-1]) + (1,))
            # shape [n_t, n_x, n_y, n_z, 5]
            txyz_u = np.concatenate([txyz_coord, ui_label], axis=-1)
            # subsample
            if num_txyz_samp_pts > 0:
                shape = self.samp_shape + (5,)
                txyz_u = _downsample_array(txyz_u, shape, random_axes=[0])
            # shape [n_txyz_pts, 5]
            txyz_u = txyz_u.reshape((-1, 5))
        elif self.samp_axes == "xyz":
            n_t = ui_label.shape[0]
            txyz_coord = txyz_coord.reshape((n_t, -1, 4))
            ui_label = ui_label.reshape((n_t, -1, 1))
            # shape [n_t, n_xyz, 5]
            txyz_u = np.concatenate([txyz_coord, ui_label], axis=-1)
            # subsample
            if num_txyz_samp_pts > 0:
                txyz_u = _downsample_array(
                    txyz_u, self.samp_shape + (5,), random_axes=[1])
            # shape [n_txyz_pts, 5]
            txyz_u = txyz_u.reshape((-1, 5))
        else:
            raise ValueError(f"Unknown 'samp_axes': {self.samp_axes}.")

        # separate subsampled coordinate and ui_label
        coordinate = txyz_u[:, :4]  # [n_txyz_pts, 4]
        ui_label = txyz_u[:, 4:]  # [n_txyz_pts, 1]

        return coordinate, ui_label

    def _sample_coord_spatial_inr(
            self,
            txyz_coord: NDArray[float],
            ui_label: NDArray[float]) -> Tuple[NDArray[float]]:
        r"""
        Subsampled coordinate, u_label on data loading for <spatial-INR>.
        Deprecated.
        """
        # [nt, nx, ny] -> [nt, nx * ny] -> [nx * ny, nt]
        ui_label = ui_label.reshape((101, -1)).astype(float_dtype).T
        # [nt, nx, ny, nz, 4] -> [nx, ny, nz, 3] -> [nx * ny * nz, 3]
        xyz_coord = txyz_coord[0, ..., 1:].reshape((-1, 3))

        # subsample
        if self.num_txyz_samp_pts > 0:
            num_xyz_pts = ui_label.shape[0]
            xyz_sample_idx = np.random.randint(
                0, num_xyz_pts, self.num_txyz_samp_pts)
            ui_label = ui_label[xyz_sample_idx]  # [n_xyz, nt]
            xyz_coord = xyz_coord[xyz_sample_idx]  # [n_xyz, 3]

        return xyz_coord, ui_label

    @abstractmethod
    def _get_dag_info(self, idx_pde: int, idx_var: int) -> Tuple[NDArray]:
        r"""Load the PDE DAG info on data loading."""

    @abstractmethod
    def _get_txyz_coord(self, idx_pde: int, idx_var: int) -> NDArray[float]:
        r"""
        Load the coordinate points corresponding to the solution instance with
        index `idx_pde` and component (i.e. variable) index `idx_var`.
        """

    @abstractmethod
    def _get_ui_label(self, idx_pde: int, idx_var: int) -> NDArray[float]:
        r"""Load the solution component with index `idx_var` on data loading."""

    @abstractmethod
    def _get_pde_latex(self, idx_pde: int, idx_var: int) -> Tuple:
        r"""
        Get the LaTeX representation of the PDE and the current variable, as
        well as a dictionary of the scalar-valued PDE coefficients.
        """

    @abstractmethod
    def _get_total_samples(self) -> int:
        r"""Total number of samples in the dataset."""


class Dedalus2DDatasetBase(MultiPDEDatasetBase):
    r"""Handing 2D PDE datasets generated by Dedalus."""
    n_vars: int = 1
    pde_info_cls: type
    h5_file_u: h5py.File
    dag_info: h5py.File
    node_func_xyz: NDArray[float]
    txyz_coord: NDArray[float]

    def __init__(self,
                 config: DictConfig,
                 filename: str,
                 n_samples: int,
                 test: bool = False,
                 for_eval: bool = False) -> None:
        super().__init__(config, filename, n_samples, test, for_eval)
        self.conf_info = {
            "dag_file_suffix": self.pde_info_cls.dag_file_suffix(config),
            # "n_func_nodes": config.data.pde_dag.max_n_function_nodes,
            "data_path": config.data.path}
        self.use_datafile(filename)

        # spatial coordinates in node_function
        node_func_xyz = np.stack(np.meshgrid(
            np.linspace(0, 1, 128 + 1)[:-1],  # x
            np.linspace(0, 1, 128 + 1)[:-1],  # y
            -1,  # z
            indexing="ij",
        ), axis=-1).astype(float_dtype)
        # [n_x, n_y, n_z, 3] -> [1, n_x * n_y * n_z, 3]
        node_func_xyz = node_func_xyz.reshape((1, -1, 3))
        n_function_nodes = config.data.pde_dag.max_n_function_nodes
        # [1, n_x * n_y * n_z, 3] -> [n_function_node, n_x * n_y * n_z, 3]
        self.node_func_xyz = np.repeat(node_func_xyz, n_function_nodes, axis=0)

        if self.n_vars > 1:
            if config.model.multi_inr.enable and config.model.multi_inr.separate_latent:
                uf_num_mod = config.model.inr.num_layers + config.model.inr2.num_layers - 2
            else:
                uf_num_mod = config.model.inr.num_layers - 1
            self.mod_node_swapper = ModNodeSwapper(uf_num_mod, self.n_vars)

    def use_datafile(self, filename: str) -> None:
        # main data file
        u_filepath = os.path.join(self.conf_info["data_path"], filename)
        self.h5_file_u = h5py.File(u_filepath + ".hdf5", "r")

        if "x" in self.h5_file_u["coord"]:  # shared coordinate case
            # spatial-temporal coordinates, shape [n_t, n_x, n_y, n_z, 4].
            self.txyz_coord = _cartesian_txyz_coord(
                t=self.h5_file_u.get("coord/t", np.array([0]))[:].flatten(),
                x=self.h5_file_u["coord/x"][:].flatten(),
                y=self.h5_file_u["coord/y"][:].flatten())

        # auxiliary data file containing dag_info
        dag_filepath = os.path.join(
            self.conf_info["data_path"], pde_types.DAG_INFO_DIR,
            filename + self.conf_info["dag_file_suffix"])
        if not os.path.exists(dag_filepath):
            raise FileNotFoundError(
                f"The file {dag_filepath} does not exist. Consider running the"
                " following command before training (as shown in"
                " scripts/run_distributed_train.sh): \n\n\t"
                "python3 preprocess_data.py -c CONFIG_PATH\n")
        self.dag_info = h5py.File(dag_filepath, "r")

    def _get_dag_info(self, idx_pde: int, idx_var: int) -> Tuple[NDArray]:
        idx_pde = self._get_file_idx(idx_pde)
        dag_info = self.dag_info

        # generate node_function
        # Shape is [n_function_nodes, 1, 1].
        node_func_t = dag_info["node_func_t"][idx_pde]
        _, n_xyz_pts, _ = self.node_func_xyz.shape
        # Shape is [n_function_nodes, n_xyz_pts, 1].
        node_func_t = np.repeat(node_func_t, n_xyz_pts, axis=1)
        node_function = np.concatenate([
            node_func_t,
            self.node_func_xyz,
            dag_info["node_func_f"][idx_pde],
        ], axis=-1)  # [n_function_nodes, n_xyz_pts, 1 + 3 + 1]

        # generate spatial_pos
        spatial_pos = dag_info["spatial_pos"][idx_pde].astype(int_dtype)
        if idx_var > 0:
            self.mod_node_swapper.apply_(spatial_pos, idx_var)

        # generate attn_bias
        node_type = dag_info["node_type"][idx_pde]
        attn_bias = PDEAsDAG.get_attn_bias(
            node_type, spatial_pos, self.disconn_attn_bias)

        dag_tuple = (node_type,  # [n_node, 1]
                     dag_info["node_scalar"][idx_pde],  # [n_scalar_node, 1]
                     node_function,  # [n_function_node, n_pts, 5]
                     dag_info["in_degree"][idx_pde],  # [n_node]
                     dag_info["out_degree"][idx_pde],  # [n_node]
                     attn_bias,  # [n_node, n_node]
                     spatial_pos,  # [n_node, n_node]
                     )
        return dag_tuple

    def _get_txyz_coord(self, idx_pde: int, idx_var: int) -> NDArray[float]:
        idx_pde = self._get_file_idx(idx_pde)
        if "x" in self.h5_file_u["coord"]:  # shared coordinate case
            return self.txyz_coord
        if "coords" not in self.h5_file_u["sol"]:
            raise RuntimeError("Cannot find coordinates in the data file.")
        t_coord = self.h5_file_u.get("coord/t", np.array([0]))[()]
        # [nt] -> [nt, 1, 1]
        t_coord = t_coord.reshape((-1, 1, 1)).astype(float_dtype)
        # Both has shape [1, n_ax1, n_ax2].
        x_coord = self.h5_file_u["sol/coords/x"][[idx_pde]].astype(float_dtype)
        y_coord = self.h5_file_u["sol/coords/y"][[idx_pde]].astype(float_dtype)
        txyz_coord = np.stack(np.broadcast_arrays(
            t_coord, x_coord, y_coord, float_dtype(0)), axis=-1)
        # [nt, n_ax1, n_ax2, 4] -> [nt, n_ax1, n_ax2, nz=1, 4]
        txyz_coord = np.expand_dims(txyz_coord, axis=3)
        return txyz_coord.astype(float_dtype)

    def _get_ui_label(self, idx_pde: int, idx_var: int) -> NDArray[float]:
        idx_pde = self._get_file_idx(idx_pde)
        if idx_var != 0:
            raise ValueError("When 'n_vars' is 1, 'idx_var' should be 0.")
        return self.h5_file_u["sol/u"][idx_pde]

    def _get_pde_latex(self, idx_pde: int, idx_var: int) -> Tuple:
        idx_pde = self._get_file_idx(idx_pde)
        pde_latex, coef_dict = self.pde_info_cls.pde_latex(
            self.h5_file_u, idx_pde, keep_all_coef=False)
        var_latex = self.pde_info_cls.var_latex(idx_var)
        return pde_latex, var_latex, coef_dict

    def _get_total_samples(self) -> int:
        return self.h5_file_u["args/num_pde"][()]

    def _get_file_idx(self, idx_pde: int) -> int:
        r"""Get the index of data in the h5file with logical index `idx_pde` in
        the dataset."""
        return idx_pde


@record_pde_dataset("diffConvecReac", "dcr")
class DiffConvecReac2DDataset(Dedalus2DDatasetBase):
    pde_info_cls: type = pde_types.DiffConvecReac2DInfo
    __doc__ = pde_info_cls.__doc__


@record_pde_dataset("dcrLgK")
class DCRLgKappa2DDataset(Dedalus2DDatasetBase):
    pde_info_cls: type = pde_types.DCRLgKappa2DInfo
    __doc__ = pde_info_cls.__doc__


@record_pde_dataset("wave")
class Wave2DDataset(Dedalus2DDatasetBase):
    pde_info_cls: type = pde_types.Wave2DInfo
    __doc__ = pde_info_cls.__doc__


@record_pde_dataset("mcompn", "multiComponent", "mcdcr", "mvdcr")  # pylint: disable=missing-docstring
class MultiComponent2DDataset(Dedalus2DDatasetBase):
    pde_info_cls: type = pde_types.MultiComponent2DInfo
    __doc__ = pde_info_cls.__doc__
    n_vars: int = None  # need to be initialized

    def use_datafile(self, filename: str) -> None:
        super().use_datafile(filename)
        n_vars = self.h5_file_u["args/n_vars"][()]
        if self.n_vars is None:  # not initialized
            self.n_vars = n_vars
        elif self.n_vars != n_vars:
            raise RuntimeError(
                f"The number of variables departures from {self.n_vars} to "
                f"{n_vars}. For dynamic multi_pde dataset, please do not mix "
                "multi-variable data files with different number of variables "
                "for the same virtual dataset object.")

    def _get_ui_label(self, idx_pde: int, idx_var: int) -> NDArray[float]:
        idx_pde = self._get_file_idx(idx_pde)
        return self.h5_file_u[f"sol/u{idx_var}"][idx_pde]


@record_pde_dataset("mcLgK", "mcdcrLgK", "mvdcrLgK")
class MCompnLgKappa2DDataset(MultiComponent2DDataset):
    pde_info_cls: type = pde_types.MCompnLgKappa2DInfo
    __doc__ = pde_info_cls.__doc__


@record_pde_dataset("mcwave", "mcWave", "MCWave", "mvwave")
class MCWave2DDataset(MultiComponent2DDataset):
    pde_info_cls: type = pde_types.MCWave2DInfo
    __doc__ = pde_info_cls.__doc__


@record_pde_dataset("divConstrDCR", "dcdcr")  # pylint: disable=missing-docstring
class DivConstraintDCR2DDataset(Dedalus2DDatasetBase):
    pde_info_cls: type = pde_types.DivConstraintDCR2DInfo
    __doc__ = pde_info_cls.__doc__
    n_vars: int = None  # need to be initialized
    valid_ic: bool  # need to be initialized

    def use_datafile(self, filename: str) -> None:
        super().use_datafile(filename)
        self.valid_ic = self.h5_file_u["args/valid_ic"][()]
        n_vars = 3 if "p" in self.h5_file_u["sol"] else 2
        if self.n_vars is None:  # not initialized
            self.n_vars = n_vars
        elif self.n_vars != n_vars:
            raise RuntimeError(
                f"The number of variables departures from {self.n_vars} to "
                f"{n_vars}. Some divergence-constrained PDE datasets contains "
                "labels of the pressure field, while others do not. For "
                "dynamic multi_pde dataset, please do not mix them up in the "
                "same virtual dataset object.")

    def _get_ui_label(self, idx_pde: int, idx_var: int) -> NDArray[float]:
        idx_pde = self._get_file_idx(idx_pde)
        var_name = "uvp"[idx_var]
        var_label = self.h5_file_u["sol/" + var_name][idx_pde]
        if idx_var == 2 or not self.valid_ic:
            var_label[0] = var_label[1]  # remove discontinuity
        return var_label


@record_pde_dataset("divConstrDCRLgK", "dcdcrLgK")
class DivConstraintDCRLgKappa2DDataset(DivConstraintDCR2DDataset):
    pde_info_cls: type = pde_types.DivConstraintDCRLgKappa2DInfo
    __doc__ = pde_info_cls.__doc__


@record_pde_dataset("divConstrWave", "dcwave")
class DivConstraintWave2DDataset(DivConstraintDCR2DDataset):
    pde_info_cls: type = pde_types.DivConstraintWave2DInfo
    __doc__ = pde_info_cls.__doc__


@record_pde_dataset("swe", "shallowWater", "gswe")
class ShallowWater2DDataset(Dedalus2DDatasetBase):
    pde_info_cls: type = pde_types.ShallowWater2DInfo
    __doc__ = pde_info_cls.__doc__
    n_vars: int = 3

    def _get_ui_label(self, idx_pde: int, idx_var: int) -> NDArray[float]:
        idx_pde = self._get_file_idx(idx_pde)
        var_name = "huv"[idx_var]
        return self.h5_file_u["sol/" + var_name][idx_pde]


@record_pde_dataset("sweLgK", "shallowWaterLgK", "gsweLgK")
class ShallowWaterLgKappa2DDataset(ShallowWater2DDataset):
    pde_info_cls: type = pde_types.ShallowWaterLgKappa2DInfo
    __doc__ = pde_info_cls.__doc__


@record_pde_dataset("elasticwave", "elasticWave", "ElasticWave")  # pylint: disable=missing-docstring
class ElasticWave2DDataset(Dedalus2DDatasetBase):
    pde_info_cls: type = pde_types.ElasticWave2DInfo
    __doc__ = pde_info_cls.__doc__
    n_vars: int = 2  # two variables
    save_scat_sol: bool  # need to be initialized
    valid_idx: NDArray[int]  # need to be initialized
    n_valid: int  # need to be initialized
    scat_txyz_coord: NDArray[float]  # need to be initialized

    def use_datafile(self, filename: str) -> None:
        super().use_datafile(filename)
        self.save_scat_sol = self.h5_file_u["args/save_scatter"][()]
        self.valid_idx = self.dag_info["valid_idx"][:]
        self.n_valid = self.dag_info["num_pde_valid"][()]
        if len(self.valid_idx) != self.n_valid:
            raise ValueError("The length of 'valid_idx' should be equal to "
                             f"'num_pde_valid'. Got {len(self.valid_idx)} and "
                             f"{self.n_valid} respectively.")
        if self.is_scat():
            self.scat_txyz_coord = self._get_scatter_txyz_coord()

    def _get_txyz_coord(self, idx_pde: int, idx_var: int) -> NDArray[float]:
        if self.is_scat():
            return self.scat_txyz_coord  # [n_t, n_xyz, 4]
        return self.txyz_coord  # [n_t, n_x, n_y, n_z, 4]

    def _get_ui_label(self, idx_pde: int, idx_var: int) -> NDArray[float]:
        idx_pde = self.valid_idx[idx_pde]
        # size is [n_x, n_y, n_t] for non-scatter solution
        # size is [n_xy, n_t] for scatter solution
        ui_label = self.h5_file_u["sol/solution"][idx_pde, ..., idx_var]
        if self.is_scat():
            # shape is [n_xyz, n_t] -> [n_t, n_xyz]
            ui_label = ui_label.T
        elif self.save_scat_sol:
            # interpolate to [n_x, n_y, n_t]
            points = self.h5_file_u["sol/mesh_points"][idx_pde]
            x_coord = self.h5_file_u["coord/x"][:].flatten()
            y_coord = self.h5_file_u["coord/y"][:].flatten()
            t_coord = self.h5_file_u.get("coord/t", np.array([0]))[:].flatten()
            interp_values = np.zeros(
                (len(x_coord), len(y_coord), len(t_coord)), dtype=float_dtype)
            x_grid, y_grid = np.meshgrid(x_coord, y_coord, indexing="ij")
            for t in range(len(t_coord)):
                interp_values[:, :, t] = griddata(
                    points, ui_label[:, t], (x_grid, y_grid), method="cubic")
            # shape is [n_x, n_y, n_t] -> [n_t, n_x, n_y]
            ui_label = interp_values.transpose(2, 0, 1)
        else:
            # shape is [n_x, n_y, n_t] -> [n_t, n_x, n_y]
            ui_label = ui_label.transpose(2, 0, 1)
        return ui_label

    def _get_dag_info(self, idx_pde: int, idx_var: int) -> Tuple[NDArray]:
        idx_pde = self.valid_idx[idx_pde]
        return super()._get_dag_info(idx_pde, idx_var)

    def _get_pde_latex(self, idx_pde: int, idx_var: int) -> Tuple:
        idx_pde = self.valid_idx[idx_pde]
        return super()._get_pde_latex(idx_pde, idx_var)

    def _get_scatter_txyz_coord(self) -> NDArray[float]:
        r"""Get the scatter coordinate grid of the current dataset."""
        if not self.save_scat_sol:
            raise ValueError("Scatter solution is not saved.")
        # shape is [n_xy, 2]
        xy_coord = self.h5_file_u["sol/mesh_points"][0]
        z_coord = np.zeros((xy_coord.shape[0], 1), dtype=float_dtype)
        # shape is [n_xyz, 3]  (in this case, n_xyz = n_xy)
        xyz_coord = np.concatenate([xy_coord, z_coord], axis=-1)
        # shape is [n_t]
        t_coord = self.h5_file_u.get("coord/t", np.array([0]))[:].flatten()
        # shape is [1, n_xyz, 3] -> [n_t, n_xyz, 3]
        xyz_coord = xyz_coord[np.newaxis, ...]
        xyz_coord = np.repeat(xyz_coord, len(t_coord), axis=0)
        # shape is [n_t, n_xyz, 4]
        t_coord_exp = t_coord[:, np.newaxis, np.newaxis]
        t_coord_exp = np.repeat(t_coord_exp, xy_coord.shape[0], axis=1)
        scat_txyz_coord = np.concatenate([t_coord_exp, xyz_coord], axis=-1)
        return scat_txyz_coord.astype(float_dtype)

    def _get_total_samples(self) -> int:
        return self.n_valid

    def is_scat(self) -> bool:
        r"""If __getitem__ method returns scatter solutions."""
        return self.save_scat_sol and not self.for_eval and self.samp_axes != 't'


@record_pde_dataset("elasticsteady", "elasticSteady", "ElasticSteady",  # pylint: disable=missing-docstring
                    "ElasticStatic", "ElasticSteadyState")
class ElasticSteady2DDataset(ElasticWave2DDataset):
    pde_info_cls: type = pde_types.ElasticSteady2DInfo
    __doc__ = pde_info_cls.__doc__

    def _get_ui_label(self, idx_pde: int, idx_var: int) -> NDArray[float]:
        idx_pde = self.valid_idx[idx_pde]
        # size is [n_x, n_y] for non-scatter solution
        # size is [n_xy] for scatter solution
        ui_label = self.h5_file_u["sol/solution"][idx_pde, ..., idx_var]
        if self.is_scat():
            # shape is [n_xyz] -> [n_xyz, 1]
            ui_label = ui_label[:, np.newaxis]
        elif self.save_scat_sol:
            # interpolate to [n_x, n_y]
            points = self.h5_file_u["sol/mesh_points"][idx_pde]
            x_coord = self.h5_file_u["coord/x"][:].flatten()
            y_coord = self.h5_file_u["coord/y"][:].flatten()
            interp_values = griddata(
                points, ui_label, (x_coord, y_coord), method="cubic")
            # shape is [n_x, n_y] -> [n_x, n_y, 1]
            ui_label = interp_values[:, :, np.newaxis]
        else:
            # shape is [n_x, n_y] -> [n_x, n_y, 1]
            ui_label = ui_label[:, :, np.newaxis]
        return ui_label

    def _get_scatter_txyz_coord(self) -> NDArray[float]:
        r"""Get the scatter coordinate grid of the current dataset."""
        if not self.save_scat_sol:
            raise ValueError("Scatter solution is not saved.")
        # shape is [n_xy, 2]
        xy_coord = self.h5_file_u["sol/mesh_points"][0]
        z_coord = np.zeros((xy_coord.shape[0], 1), dtype=float_dtype)
        # shape is [1, n_xy, 3]
        xyz_coord = np.concatenate([xy_coord, z_coord], axis=-1)[np.newaxis, ...]
        t_coord = np.zeros_like(xyz_coord[..., 0:1])
        # shape is [1, n_xy, 3] -> [1, n_xy, 4]
        scat_txyz_coord = np.concatenate([t_coord, xyz_coord], axis=-1)
        return scat_txyz_coord.astype(float_dtype)
