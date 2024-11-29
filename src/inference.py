r"""PDEformer inference on a given PDE."""
from typing import Union, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from mindspore import Tensor

from src.cell import PDEformer
from src.data.pde_dag import PDEAsDAG
from src.utils.visual import plot_2d_snapshots


def inference_pde(model: PDEformer,
                  pde_dag: PDEAsDAG,
                  txyz_coord: NDArray[float]) -> NDArray[float]:
    r"""
    Use the PDEformer model to predict the solution of the PDE specified in
    `pde_dag`.
    Input txyz_coord has shape [..., 4], output has shape [..., n_vars].
    """
    coordinate = txyz_coord.astype(np.float32).reshape((-1, 4))

    def as_tensor(array):
        return Tensor(array).expand_dims(0)  # [*] -> [1, *]

    # inference the first PDE component
    pred = model(as_tensor(pde_dag.node_type), as_tensor(pde_dag.node_scalar),
                 as_tensor(pde_dag.node_function), as_tensor(pde_dag.in_degree),
                 as_tensor(pde_dag.out_degree), as_tensor(pde_dag.attn_bias),
                 as_tensor(pde_dag.spatial_pos), as_tensor(coordinate))
    pred = pred.asnumpy().astype(np.float32)  # [1, n_pts, 1]

    # multi-component case, inference the rest components
    if pde_dag.n_vars > 1:
        pred_all = [pred]
        # iterate over all remaining components
        for idx_var in range(1, pde_dag.n_vars):
            spatial_pos, attn_bias = pde_dag.get_spatial_pos_attn_bias(idx_var)
            pred = model(
                as_tensor(pde_dag.node_type), as_tensor(pde_dag.node_scalar),
                as_tensor(pde_dag.node_function), as_tensor(pde_dag.in_degree),
                as_tensor(pde_dag.out_degree), as_tensor(attn_bias),
                as_tensor(spatial_pos), as_tensor(coordinate))
            pred = pred.asnumpy().astype(np.float32)  # [1, n_pts, 1]
            pred_all.append(pred)
        pred = np.concatenate(pred_all, axis=-1)  # [1, n_pts, n_vars]

    pred = pred.reshape(txyz_coord.shape[:-1] + (pde_dag.n_vars,))
    return pred


def inference_cartesian(model: PDEformer,
                        pde_dag: PDEAsDAG,
                        t_coord: NDArray[float],
                        x_coord: NDArray[float],
                        y_coord: Union[float, NDArray[float]] = 0.,
                        z_coord: Union[float, NDArray[float]] = 0.,
                        ) -> NDArray[float]:
    r"""
    Use the PDEformer model to predict the solution of the PDE as specified in
    `pde_dag`.
    """
    # coordinate
    txyz_coord = np.stack(np.meshgrid(
        t_coord, x_coord, y_coord, z_coord, indexing="ij"), axis=-1)
    pred = inference_pde(model, pde_dag, txyz_coord)
    return pred  # [n_t, n_x, n_y, n_z, n_vars]


def infer_plot_2d(model: PDEformer,
                  pde_dag: PDEAsDAG,
                  x_ext: NDArray[float],
                  y_ext: NDArray[float],
                  snap_t: Union[Tuple[float], NDArray[float]] = (0, 0.25, 0.5, 0.75, 1),
                  ) -> NDArray[float]:
    r"""
    Plot the predicted snapshots of a given (time-dependent) PDE, and return
    the snapshot array with shape [n_t, dim1, dim2, n_vars].
    """
    coord = np.stack(np.broadcast_arrays(
        np.reshape(snap_t, (-1, 1, 1)),  # [n_t] -> [n_t, 1, 1]
        np.expand_dims(x_ext, axis=0),  # [dim1, dim2] -> [1, dim1, dim2]
        np.expand_dims(y_ext, axis=0),
        0), axis=-1)  # [n_t, dim1, dim2, 4], (t,x,y,z) along the last axis
    # [n_t, dim1, dim2, n_vars]
    snapshots = inference_pde(model, pde_dag, coord)
    plot_2d_snapshots(snapshots, snap_t, x_ext, y_ext)
    plt.show()
    return snapshots


# fixed coordinates for CNN function encoder
x_fenc, y_fenc = np.meshgrid(np.linspace(0, 1, 129)[:-1],
                             np.linspace(0, 1, 129)[:-1],
                             indexing="ij")


def interp_fenc(field: NDArray[float], squeeze: bool = True) -> NDArray[float]:
    r"""
    During rollout of the 2D model, the final state generated on a coarser
    uniform grid is interpolated to the finer 128x128 uniform grid, so that it
    can be used as the initial condition of the next inference.

    Argument: field (NDArray[float]), predicted final state with shape
        [n_x, n_y, n_vars].
    Outputs: interp_field (NDArray[float]), interpolated field with shape
        [128, 128, n_vars], or [128, 128] if `n_vars` equals one and `squeeze`
        is True.
    """
    n_x, n_y, _ = field.shape
    x_old = np.linspace(0, 1, n_x)
    y_old = np.linspace(0, 1, n_y)
    interp = RegularGridInterpolator(
        (x_old, y_old), field, bounds_error=False, fill_value=None)
    interp_field = interp((x_fenc, y_fenc))
    if squeeze:
        interp_field = interp_field.squeeze()
    return interp_field
