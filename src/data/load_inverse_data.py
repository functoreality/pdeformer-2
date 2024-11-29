r"""Loading datasets for inverse problems."""
import os
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
import h5py
from omegaconf import DictConfig
from mindspore import Tensor

from .env import float_dtype
from .pde_dag import DAGInfoTuple, PfDataTuple
from .multi_pde.pde_types import pde_info_cls_dict


def get_inverse_data(config: DictConfig, idx_pde: int) -> Tuple:
    r"""
    Generate the data tuple of one PDE for the inverse problems (recovery of
    scalar coefficients or coefficient fields), including multiple initial
    conditions at a time.
    """
    num_samples = config.inverse.num_samples_per_pde  # int
    keep_all_coef = config.inverse.system_identification  # bool
    if config.inverse.pde_type not in ["dcr", "wave"]:
        raise NotImplementedError
    pde_info_cls = pde_info_cls_dict[config.inverse.pde_type]  # class type
    dag_tuple = DAGInfoTuple([], [], [], [], [], [], [])

    u_filepath = config.inverse.data_file  # str
    u_filepath = os.path.join(config.data.path, u_filepath + ".hdf5")
    with h5py.File(u_filepath, "r") as h5_file_u:
        txyz_coord = np.stack(np.meshgrid(
            np.ravel(h5_file_u.get("coord/t", 0.)),
            np.ravel(h5_file_u.get("coord/x", 0.)),
            np.ravel(h5_file_u.get("coord/y", 0.)),
            np.ravel(h5_file_u.get("coord/z", 0.)),
            indexing="ij"), axis=-1)
        u_label = h5_file_u["sol/u"][idx_pde, :num_samples]
        pde_latex, coef_dict = pde_info_cls.pde_latex(
            h5_file_u, (idx_pde, 0), keep_all_coef=keep_all_coef)

        for i_sample in range(num_samples):
            pde = pde_info_cls.pde_nodes(
                h5_file_u, (idx_pde, i_sample), keep_all_coef)
            pde_dag = pde.gen_dag(config)
            dag_tuple.node_type.append(pde_dag.node_type)
            dag_tuple.node_scalar.append(pde_dag.node_scalar)
            dag_tuple.node_function.append(pde_dag.node_function)
            dag_tuple.in_degree.append(pde_dag.in_degree)
            dag_tuple.out_degree.append(pde_dag.out_degree)
            dag_tuple.attn_bias.append(pde_dag.attn_bias)
            dag_tuple.spatial_pos.append(pde_dag.spatial_pos)

    n_t, n_x, n_y, n_z, _ = txyz_coord.shape
    # [n_t, n_x, n_y, n_z, 4] -> [1, n_txyz, 4] -> [num_samples, n_txyz, 4]
    txyz_coord = txyz_coord.astype(float_dtype).reshape((1, -1, 4))
    txyz_coord = np.repeat(txyz_coord, num_samples, axis=0)
    # [num_samples, n_t, n_x, n_y] -> [num_samples, n_txyz, 1]
    u_label = u_label.reshape((num_samples, -1, 1)).astype(float_dtype)
    # Tuple[List[NDArray]] -> Tuple[NDArray]
    dag_tuple = DAGInfoTuple(*(np.array(arr_list) for arr_list in dag_tuple))

    # Add noise to initial condition.
    # node_function.shape is [num_samples, n_function_nodes, n_xyz, 5].
    ic_all = dag_tuple.node_function[:, 0, :, -1]  # [num_samples, n_xyz]
    ic_all = add_noise(
        ic_all,
        config.inverse.observation.ic_noise.type.lower(),
        config.inverse.observation.ic_noise.level)
    dag_tuple.node_function[:, 0, :, -1] = ic_all
    if config.inverse.pde_type == "wave":
        # add noise to initial condition for u_t
        ic_all = dag_tuple.node_function[:, 1, :, -1]
        ic_all = add_noise(
            ic_all,
            config.inverse.observation.ic_noise.type.lower(),
            config.inverse.observation.ic_noise.level)
        dag_tuple.node_function[:, 1, :, -1] = ic_all

    # data_tuple
    data_tuple = PfDataTuple(*dag_tuple, txyz_coord, u_label)
    data_tuple = PfDataTuple(*(Tensor(array) for array in data_tuple))

    # data_info
    data_info = {"pde_latex": pde_latex, "coef_dict": coef_dict,
                 "idx_pde": idx_pde,  # "n_xyz": n_x * n_y * n_z,
                 "n_t_grid": n_t, "n_x_grid": n_x,
                 "n_y_grid": n_y, "n_z_grid": n_z,}
    return data_tuple, data_info


def add_noise(u_label: NDArray[float],
              noise_type: str,
              noise_level: float) -> NDArray[float]:
    r"""Add noise to observations for inverse problems."""
    batch_size = u_label.shape[0]
    # [batch_size, ...] -> [batch_size, *]
    u_reshape = u_label.reshape((batch_size, -1))
    u_max = np.abs(u_reshape).max(axis=1, keepdims=True)  # [batch_size, 1]
    if noise_type == 'none':
        noise = 0
    elif noise_type == 'uniform':
        noise = np.random.uniform(low=-u_max, high=u_max, size=u_reshape.shape)
    elif noise_type == 'normal':
        noise = u_max * np.random.normal(size=u_reshape.shape)
    else:
        raise NotImplementedError(
            "The noise_type must be in ['none', 'uniform', 'normal'], but got"
            f"' {noise_type}'")
    u_noisy = u_reshape + noise_level * noise
    # [batch_size, *] -> [batch_size, ...]
    u_noisy = u_noisy.reshape(u_label.shape)
    return u_noisy


def get_observed_indices(batch_size: int,
                         n_t_grid: int,
                         n_xyz: int,
                         xyz_obs_type: str = "all",
                         n_xyz_obs_pts: int = 10,
                         t_obs_type: str = "all",
                         n_t_obs_pts: int = 10) -> NDArray[int]:
    r"""
    Generate the indices of the spatial-temporal observation points for inverse
    problems.
    """
    obs_inds = np.arange(batch_size * n_t_grid * n_xyz).reshape(
        (batch_size, n_t_grid, n_xyz))

    # x locations
    # [bsz, n_t_grid, n_xyz] -> [bsz, n_t_grid, n_xyz_obs_pts]
    if xyz_obs_type == "all":
        n_xyz_obs_pts = n_xyz
    elif xyz_obs_type == "equispaced":
        # should be equispaced along each axis rather than the concatenated axis
        stride = n_xyz // n_xyz_obs_pts
        obs_inds = obs_inds[:, :, ::stride]
        obs_inds = obs_inds[:, :, :n_xyz_obs_pts]
        raise NotImplementedError
    elif xyz_obs_type == "last":
        obs_inds = obs_inds[:, :, -n_xyz_obs_pts:]
        raise NotImplementedError
    elif xyz_obs_type == "random":
        x_inds = np.random.choice(n_xyz, size=n_xyz_obs_pts, replace=False)
        obs_inds = obs_inds[:, :, x_inds]
    else:
        raise NotImplementedError(f"Unknown xyz_obs_type {xyz_obs_type}.")

    # t locations
    # [bsz, n_t_grid, n_xyz_obs_pts] -> [bsz, n_t_obs_pts, n_xyz_obs_pts]
    if t_obs_type == "all":
        n_t_obs_pts = n_t_grid
    elif t_obs_type == "equispaced":
        stride = n_t_grid // n_t_obs_pts
        obs_inds = obs_inds[:, ::stride, :]
        obs_inds = obs_inds[:, -n_t_obs_pts:, :]
    elif t_obs_type == "last":
        obs_inds = obs_inds[:, -n_t_obs_pts:, :]
    elif t_obs_type == "t_random":
        t_inds = np.random.choice(n_t_grid, size=n_t_obs_pts, replace=False)
        obs_inds = obs_inds[:, t_inds, :]
    elif t_obs_type == "all_random":
        obs_inds_old = obs_inds.reshape((batch_size, n_t_grid * n_xyz_obs_pts))
        n_tx_obs_pts = n_t_obs_pts * n_xyz_obs_pts
        obs_inds = []
        for i in range(batch_size):
            inds_i = np.random.choice(
                obs_inds_old.shape[1], size=n_tx_obs_pts, replace=False)
            obs_inds.append(obs_inds_old[i, inds_i])  # [n_tx_obs_pts]
        obs_inds = np.array(obs_inds)  # [bsz, n_tx_obs_pts]
    else:
        raise NotImplementedError(f"Unknown t_obs_type {t_obs_type}.")

    return obs_inds.flat  # [bsz * n_t_obs_pts * n_xyz_obs_pts]


def inverse_observation(observe_config: DictConfig,
                        u_label: NDArray[float],
                        coord_gt: NDArray[float]) -> Tuple[NDArray[float]]:
    r"""
    Apply additive noise and restriction of spatial-temporal observation points
    for inverse problems.
    """
    u_noisy = add_noise(u_label, observe_config.noise.type.lower(),
                        observe_config.noise.level)

    # u_label.shape is: [batch_size, n_t, n_x, n_y, n_z, 1]
    batch_size, n_t_grid = u_label.shape[:2]
    n_xyz = np.size(u_label[0, 0])

    # obs_inds
    obs_inds = get_observed_indices(
        batch_size, n_t_grid, n_xyz,
        xyz_obs_type=observe_config.xyz_location.type.lower(),
        n_xyz_obs_pts=observe_config.xyz_location.num_pts,
        t_obs_type=observe_config.t_location.type.lower(),
        n_t_obs_pts=observe_config.t_location.num_pts)

    # u_obs_plot
    mask_obs = np.ones(u_label.shape, dtype=bool).flatten()
    mask_obs[obs_inds] = False
    mask_obs = mask_obs.reshape(u_label.shape)
    u_obs_plot = np.ma.masked_array(u_noisy, mask=mask_obs)

    # u_obs
    u_obs = u_noisy.reshape((-1, 1))  # [bsz * n_t_grid * n_xyz, 1]
    u_obs = u_obs[obs_inds, :]  # [bsz * n_t_obs_pts * n_xyz_obs_pts, 1]
    # Shape is [bsz, n_t_obs_pts * n_xyz_obs_pts, 1].
    u_obs = u_obs.reshape((batch_size, -1, 1))

    # coord_obs
    coord_obs = coord_gt.reshape((-1, 4))  # [bsz * n_t_grid * n_xyz, 4]
    coord_obs = coord_obs[obs_inds, :]  # [bsz * n_t_obs_pts * n_xyz_obs_pts, 4]
    # Shape is [bsz, n_t_obs_pts * n_xyz_obs_pts, 4].
    coord_obs = coord_obs.reshape((batch_size, -1, 4))

    return u_noisy, u_obs_plot, u_obs, coord_obs
