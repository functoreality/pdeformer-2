r"""Some tool functions."""
import math
import random

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

import mindspore as ms
from mindspore import nn, ops, Tensor


def set_seed(seed: int) -> None:
    r"""Set random seed"""
    np.random.seed(seed)
    random.seed(seed)
    ms.set_seed(seed)


def calculate_num_params(model: nn.Cell) -> str:
    r"""Calculate the number of parameters."""
    num_params = 0
    for param in model.trainable_params():
        num_params += np.prod(param.shape)

    if num_params < 1000:
        num_str = str(num_params)
    elif num_params < 1000 * 1000:
        num_str = f"{(num_params / 1000):.2f}" + "K"
    elif num_params < 1000 * 1000 * 1000:
        num_str = f"{(num_params / (1000*1000)):.2f}" + "M"
    else:
        num_str = f"{(num_params / (1000*1000*1000)):.2f}" + "G"

    return num_str


class AllGather(nn.Cell):
    r"""Use nn.Cell to encapsulate ops.AllGather()."""

    def __init__(self) -> None:
        super().__init__()
        self.allgather = ops.AllGather()

    def construct(self, x: Tensor) -> Tensor:
        r"""construct"""
        return self.allgather(x)


def sample_grf(batch_size: int = 1,
               imshow: bool = False,
               alpha: float = 3.,
               tau: float = 5.,
               sigma: float = 50.) -> NDArray[float]:
    r"""
    Sample from a Gaussian Random Field (GRF) given as
    $X ~ N(\mu, \sigma^2(-\Delta + \tau^2 I)^{-\alpha})$
    with the mean value $\mu$ drawn from a one-dimensional normal distribution.
    """
    ndim = 2
    nxys = [128, 128]

    # compute sqrt_eig
    k2_sum = 0.
    nx_mul = 1.
    for dim_idx in range(ndim):
        nx = nxys[dim_idx]
        nx_mul *= nx
        kmax = nx // 2
        kx = np.concatenate(
            (np.arange(0, kmax), np.arange(-kmax, 0)), axis=0)
        kx = kx[(None,) * dim_idx + (...,) + (None,) * (ndim - 1 - dim_idx)]
        k2_sum = k2_sum + kx**2
    sqrt_eig = nx_mul * math.sqrt(2.) * sigma * (4 * math.pi**2 * k2_sum + tau**2)**(-alpha / 2.)

    # generate GRF
    noise_h = np.random.standard_normal((2, batch_size, *nxys))
    noise_h = noise_h[0] + 1.j * noise_h[1]  # [batch_size, *nxys]
    grf_h = sqrt_eig[np.newaxis, ...] * noise_h
    grf = np.fft.ifftn(grf_h, axes=tuple(range(1, ndim + 1)))
    grf = grf.real

    # post-processing
    if imshow:
        _, axes = plt.subplots(1, batch_size, squeeze=False)
        for i in range(batch_size):
            img = axes[0, i].imshow(grf[i].T, origin="lower", cmap="turbo")
            plt.colorbar(img, ax=axes[0, i])
        plt.suptitle("generated GRF")
    if batch_size == 1:  # squeeze
        grf, = grf  # [1, *nxys] -> [*nxys]
    return grf
