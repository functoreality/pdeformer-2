"""Loss functions."""
from typing import Optional, List, Union, Tuple
from abc import abstractmethod

import numpy as np
from omegaconf import OmegaConf, DictConfig
from mindspore import nn, ops, Tensor, Parameter, float32, context
from mindspore.ops import functional as F
from mindspore.ops import operations as P

from ..cell.baseline.dft import dft2


class PerSampleLossBase(nn.Cell):
    r"""
    Base class for per-sample losses.

    This class serves as a base for losses that are calculated on a per-sample
    basis. It provides methods to handle the reduction of loss tensors over the
    specified axes, starting from axis 1 to avoid reducing over the sample axis
    (axis 0).

    Inputs:
        - **logits** (Tensor) - Tensor of shape :math:`(M, *)` where :math:`*`
          means, any number of additional dimensions.
        - **labels** (Tensor) - Tensor of shape :math:`(N, *)`, same shape as
          the `logits` in common cases.  However, it supports the shape of
          `logits` is different from the shape of `labels` and they should be
          broadcasted to each other.

    Outputs:
        Tensor, weighted loss float tensor, the shape is :math:`(N)`.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction == "mean":
            self.reduce = ops.mean
        elif reduction == "sum":
            self.reduce = ops.sum
        elif reduction == "nanmean":
            self.reduce = ops.nanmean
        elif reduction == "nansum":
            self.reduce = ops.nansum
        else:
            raise ValueError

        self.mul = P.Mul()
        self.cast = P.Cast()

    @abstractmethod
    def construct(self,
                  logits: Tensor,
                  labels: Tensor,
                  coordinate: Optional[Tensor]) -> Tensor:
        r"""construct"""

    def get_loss(self, x: Tensor, weights: Union[float, Tensor] = 1.0) -> Tensor:
        r"""
        Computes the weighted loss for each data sample.

        Args:
            x (Tensor): Tensor of shape :math:`(N, *)` where :math:`*` means,
                any number of additional dimensions.
            weights (Union[float, Tensor]): Optional `Tensor` whose rank is
                either 0, or the same rank as inputs, and must be broadcastable
                to inputs (i.e., all dimensions must be either `1`, or the same
                as the corresponding inputs dimension). Default: ``1.0`` .

        Returns:
            The weighted loss for each data sample.
        """
        input_dtype = x.dtype
        x = self.cast(x, float32)
        weights = self.cast(weights, float32)
        x = self.mul(weights, x)
        x = x.reshape(x.shape[0], -1)  # [B, ...] -> [B, *]
        x = self.reduce(x, axis=-1)  # [B, *] -> [B]
        x = self.cast(x, input_dtype)
        return x


class PerSampleMSELoss(PerSampleLossBase):
    r"""
    Calculates the mean squared error between the predicted value and the label
    value for each sample.

    Args:
        reduction (str, optional): Apply specific reduction method to the
            output for each data sample: ``'none'`` , ``'mean'`` , ``'sum'`` .
            Default: ``'mean'`` .

            - ``'none'``: no reduction will be applied.
            - ``'mean'``: compute and return the mean of elements in the output.
            - ``'sum'``: the output elements will be summed.

    Inputs:
        - **logits** (Tensor) - Tensor of shape :math:`(N, *)` where :math:`*`
          means, any number of additional dimensions.
        - **labels** (Tensor) - Tensor of shape :math:`(N, *)`, same shape as
          the `logits` in common cases. However, it supports the shape of
          `logits` is different from the shape of `labels` and they should be
          broadcasted to each other.

    Outputs:
        Tensor, weighted loss float tensor and its shape is :math:`(N)`.

    Raises:
        ValueError: If `reduction` is not one of ``'none'``, ``'mean'`` or ``'sum'``.
        ValueError: If `logits` and `labels` have different shapes and cannot be broadcasted.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>> loss = PerSampleMSELoss()
        >>> logits = Tensor(np.ones((4, 2, 3)), mindspore.float32)
        >>> labels = Tensor(np.ones((4, 1, 1)), mindspore.float32)
        >>> output = loss(logits, labels)
        >>> print(output.shape)
        (4,)
    """

    def construct(self,
                  logits: Tensor,
                  labels: Tensor,
                  coordinate: Optional[Tensor] = None) -> Tensor:
        x = F.square(logits - labels)
        return self.get_loss(x)


class PerSampleRMSELoss(PerSampleMSELoss):
    r"""
    PerSampleRMSELoss creates a criterion to measure the root mean square error
    between :math:`x` and :math:`y` for each data sample, where :math:`x` is
    the predicted value and :math:`y` is the label.

    Inputs:
        - **logits** (Tensor) - Tensor of shape :math:`(N, *)` where :math:`*`
          means, any number of additional dimensions.
        - **labels** (Tensor) - Tensor of shape :math:`(N, *)`, same shape as
          the `logits` in common cases. However, it supports the shape of
          `logits` is different from the shape of `labels` and they should be
          broadcasted to each other.

    Outputs:
        Tensor, weighted loss float tensor and its shape is :math:`(N)`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>> loss = PerSampleRMSELoss()
        >>> logits = Tensor(np.ones((4, 2, 3)), mindspore.float32)
        >>> labels = Tensor(np.ones((4, 1, 1)), mindspore.float32)
        >>> output = loss(logits, labels)
        >>> print(output.shape)
        (4,)
    """

    def construct(self,
                  logits: Tensor,
                  labels: Tensor,
                  coordinate: Optional[Tensor] = None) -> Tensor:
        mse_loss = super().construct(logits, labels)
        rmse_loss = F.sqrt(mse_loss)
        return rmse_loss


class PerSampleMAELoss(PerSampleLossBase):
    r"""
    PerSampleMAELoss creates a criterion to measure the average absolute error
    between :math:`x` and :math:`y` for each sample, where :math:`x` is the
    predicted value and :math:`y` is the label.

    Args:
        reduction (str, optional): Apply specific reduction method to the
            output for each data sample: ``'none'`` , ``'mean'`` , ``'sum'`` .
            Default: ``'mean'`` .

            - ``'none'``: no reduction will be applied.
            - ``'mean'``: compute and return the mean of elements in the output.
            - ``'sum'``: the output elements will be summed.

    Inputs:
        - **logits** (Tensor) - Tensor of shape :math:`(N, *)` where :math:`*`
          means, any number of additional dimensions.
        - **labels** (Tensor) - Tensor of shape :math:`(N, *)`, same shape as
          the `logits` in common cases. However, it supports the shape of
          `logits` is different from the shape of `labels` and they should be
          broadcasted to each other.

    Outputs:
        Tensor, weighted loss float tensor and its shape is :math:`(N)`.

    Raises:
        ValueError: If `reduction` is not one of ``'none'``, ``'mean'`` or ``'sum'``.
        ValueError: If `logits` and `labels` have different shapes and cannot be broadcasted.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>> loss = PerSampleMAELoss()
        >>> logits = Tensor(np.ones((4, 2, 3)), mindspore.float32)
        >>> labels = Tensor(np.ones((4, 1, 1)), mindspore.float32)
        >>> output = loss(logits, labels)
        >>> print(output.shape)
        (4,)
    """

    def construct(self,
                  logits: Tensor,
                  labels: Tensor,
                  coordinate: Optional[Tensor] = None) -> Tensor:
        x = F.abs(logits - labels)
        return self.get_loss(x)


class PerSampleMixedLoss(PerSampleLossBase):
    r"""
    PerSampleMixedLoss creates a criterion to measure a weighted sum of average
    absolute error and root mean square error between :math:`x` and :math:`y`
    for each sample, where :math:`x` is the predicted value and :math:`y` is
    the label.

    Args:
        weight (float, optional): The weight of MAE loss in mixed loss, the weight
            of MSE loss will be calculated as :math:`1 - weight`.
            Default: ``0.1`` .

    Inputs:
        - **logits** (Tensor) - Tensor of shape :math:`(N, *)` where :math:`*`
          means, any number of additional dimensions.
        - **labels** (Tensor) - Tensor of shape :math:`(N, *)`, same shape as
          the `logits` in common cases. However, it supports the shape of
          `logits` is different from the shape of `labels` and they should be
          broadcasted to each other.

    Outputs:
        Tensor, weighted loss float tensor and its shape is :math:`(N)`.

    Raises:
        ValueError: If `reduction` is not one of ``'none'``, ``'mean'`` or ``'sum'``.
        ValueError: If `logits` and `labels` have different shapes and cannot be broadcasted.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>> loss = PerSampleMAELoss()
        >>> logits = Tensor(np.ones((4, 2, 3)), mindspore.float32)
        >>> labels = Tensor(np.ones((4, 1, 1)), mindspore.float32)
        >>> output = loss(logits, labels)
        >>> print(output.shape)
        (4,)
    """

    def __init__(self, weight: float = 0.1, reduction: str = "mean") -> None:
        super().__init__(reduction)
        self.mae_weight = weight
        self.rmse_weight = 1. - weight
        self.mae = PerSampleMAELoss(reduction)
        self.rmse = PerSampleRMSELoss(reduction)

    def construct(self,
                  logits: Tensor,
                  labels: Tensor,
                  coordinate: Optional[Tensor] = None) -> Tensor:
        rmse = self.rmse(logits, labels)
        mae = self.mae(logits, labels)
        loss = rmse * self.rmse_weight + mae * self.mae_weight
        return loss


class PerSampleSpectralLoss(PerSampleLossBase):
    r"""
    Calculates the weighted mean squared error in Fourier space between the
    predicted value and the label value for each sample, with lower frequency
    modes weighted more heavily.

    Args:
        modes (Tuple[int]): The number of Fourier modes to keep in each dimension.
        sqrt (bool): Whether to take square-root of the loss (i.e. using RMSE).
        reduction (str, optional): Apply specific reduction method to the
            output: 'none', 'mean', 'sum'.  Default: 'mean'

    Inputs:
        - **logits** (Tensor) - Tensor of shape :math:`(N, *)` where :math:`*`
          means any number of additional dimensions.
        - **labels** (Tensor) - Tensor of shape :math:`(N, *)`, same shape as
          the `logits`.

    Outputs:
        Tensor, weighted spectral loss float tensor and its shape is :math:`(N)`.
    """

    def __init__(self,
                 modes: Union[int, Tuple[int]] = (12, 12),
                 sqrt: bool = False,
                 reduction: str = "mean",
                 compute_dtype=float32) -> None:
        super().__init__(reduction)
        self.sqrt = sqrt
        if isinstance(modes, int):
            modes = (modes, modes)
        self.dft_cell = dft2(shape=(128, 128), modes=modes,
                             compute_dtype=compute_dtype)

    def construct(self,
                  logits: Tensor,
                  labels: Tensor,
                  coordinate: Optional[Tensor] = None) -> Tensor:
        error = logits - labels
        bsz = error.shape[0]
        n_vars = error.shape[-1]
        # [bsz, ..., n_vars] -> [bsz, nx=128, ny=128, n_vars]
        error = error.reshape(bsz, 128, 128, n_vars)
        # [bsz, nx=128, ny=128, n_vars] -> [bsz, n_vars, 128, 128]
        error = error.transpose(0, 3, 1, 2)

        # Transform error to Fourier space
        err_ft_re, err_ft_im = self.dft_cell((error, ops.zeros_like(error)))
        err_ft_magnitude = err_ft_re**2 + err_ft_im**2
        loss = self.get_loss(err_ft_magnitude)
        if self.sqrt:
            loss = ops.sqrt(loss)
        return loss


class PerSampleH1Loss(PerSampleLossBase):
    r"""
    PerSampleH1Loss creates a criterion to measure the H1 loss by Fourier transform
    between :math:`x` and :math:`y` for each sample, where :math:`x` is the
    predicted value and :math:`y` is the label.

    Args:
        kmax (float, optional): The maximum wavenumber to consider in the Fourier transform.
            Default: ``10.0`` .
        knum (int, optional): Randomly sample 'knum' wavenumbers to estimate norm of gradient.
            Default: ``8192`` .

    Inputs:
        - **logits** (Tensor) - Tensor of shape :math:`(N, *)` where :math:`*`
          means, any number of additional dimensions.
        - **labels** (Tensor) - Tensor of shape :math:`(N, *)`, same shape as
          the `logits` in common cases. However, it supports the shape of
          `logits` is different from the shape of `labels` and they should be
          broadcasted to each other.

    Outputs:
        Tensor, weighted loss float tensor and its shape is :math:`(N)`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>> loss = PerSampleRMSELoss()
        >>> logits = Tensor(np.ones((4, 2, 3)), mindspore.float32)
        >>> labels = Tensor(np.ones((4, 1, 1)), mindspore.float32)
        >>> output = loss(logits, labels)
        >>> print(output.shape)
        (4,)
    """

    def __init__(self,
                 kmax: float = 10.0,
                 knum: int = 8192,
                 reduction: str = "mean") -> None:
        super().__init__(reduction)
        self.kmax = kmax
        self.knum = knum
        self.mse = PerSampleMSELoss(reduction)

    def construct(self,
                  logits: Tensor,
                  labels: Tensor,
                  coordinate: Tensor) -> Tensor:
        abs_u = F.abs(logits - labels)  # [bsz, n_pts, 1]
        # ignore the z-coordinate
        coordinate = coordinate[:, :, :3]  # txyz -> txy, shape [bsz, n_pts, 3]
        ks = Tensor(2 * np.pi * np.random.uniform(0, self.kmax, (3, self.knum)), dtype=abs_u.dtype)
        kdotx = ops.matmul(coordinate, ks)  # [bsz, n_pts, knum]
        real = ops.mean(abs_u * ops.cos(kdotx), -2, keep_dims=True)  # [bsz, 1, knum]
        imag = ops.mean(abs_u * ops.sin(kdotx), -2, keep_dims=True)  # [bsz, 1, knum]
        u_hat_square = real * real + imag * imag
        du_hat_square = u_hat_square * ks * ks  # [bsz, 3, knum]
        du_norm_square = ops.mean(du_hat_square, -1)  # [bsz, 3]
        loss = F.sqrt(self.mse(logits, labels) + ops.sum(du_norm_square, -1))  # [bsz]
        return loss


class PerSampleH1LossT(PerSampleLossBase):
    r"""
    PerSampleH1LossT creates a criterion to measure the H1 loss when the samp_axes
    is 't' between :math:`x` and :math:`y` for each sample, where :math:`x` is the
    predicted value and :math:`y` is the label.

    Args:
        samp_shape (List[int]): The shape of the downsampled mesh.

    Inputs:
        - **logits** (Tensor) - Tensor of shape :math:`(N, *)` where :math:`*`
          means, any number of additional dimensions.
        - **labels** (Tensor) - Tensor of shape :math:`(N, *)`, same shape as
          the `logits` in common cases. However, it supports the shape of
          `logits` is different from the shape of `labels` and they should be
          broadcasted to each other.

    Outputs:
        Tensor, weighted loss float tensor and its shape is :math:`(N)`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>> loss = PerSampleRMSELoss()
        >>> logits = Tensor(np.ones((4, 2, 3)), mindspore.float32)
        >>> labels = Tensor(np.ones((4, 1, 1)), mindspore.float32)
        >>> output = loss(logits, labels)
        >>> print(output.shape)
        (4,)
    """

    def __init__(self, samp_shape: List[int], reduction: str = "mean") -> None:
        super().__init__(reduction)
        self.mse = PerSampleMSELoss(reduction)
        self.samp_shape = tuple(samp_shape)

    def construct(self,
                  logits: Tensor,
                  labels: Tensor,
                  coordinate: Tensor) -> Tensor:
        residual = logits - labels  # [bsz, n_pts, 1]
        bsz = residual.shape[0]
        shape = (bsz,) + self.samp_shape
        residual = residual.reshape(shape)  # [bsz, n_t, n_x, n_y, n_z]
        coordinate = coordinate.reshape(shape + (4,))  # [bsz, n_t, n_x, n_y, n_z, 4]
        x_coord = coordinate[:, 0:1, :, 0:1, 0:1, 1]  # [bsz, 1, n_x, 1, 1]
        y_coord = coordinate[:, 0:1, 0:1, :, 0:1, 2]  # [bsz, 1, 1, n_y, 1]
        diff_x = ops.diff(x_coord, axis=2)  # [bsz, 1, n_x - 1, 1, 1]
        diff_y = ops.diff(y_coord, axis=3)  # [bsz, 1, 1, n_y - 1, 1]
        diff_x = ops.diff(residual, axis=2) / diff_x  # [bsz, n_t, n_x - 1, n_y, n_z]
        diff_y = ops.diff(residual, axis=3) / diff_y  # [bsz, n_t, n_x, n_y - 1, n_z]
        loss = F.sqrt(self.mse(logits, labels) + self.mse(diff_x, 0. * diff_x) +
                      self.mse(diff_y, 0. * diff_y))
        return loss


class PerSampleH1LossXYZ(PerSampleLossBase):
    r"""
    PerSampleH1LossXYZ creates a criterion to measure the H1 loss when the samp_axes
    is 'xyz' between :math:`x` and :math:`y` for each sample, where :math:`x` is the
    predicted value and :math:`y` is the label.

    Args:
        samp_shape (List[int]): The shape of the downsampled mesh.

    Inputs:
        - **logits** (Tensor) - Tensor of shape :math:`(N, *)` where :math:`*`
          means, any number of additional dimensions.
        - **labels** (Tensor) - Tensor of shape :math:`(N, *)`, same shape as
          the `logits` in common cases. However, it supports the shape of
          `logits` is different from the shape of `labels` and they should be
          broadcasted to each other.

    Outputs:
        Tensor, weighted loss float tensor and its shape is :math:`(N)`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>> loss = PerSampleRMSELoss()
        >>> logits = Tensor(np.ones((4, 2, 3)), mindspore.float32)
        >>> labels = Tensor(np.ones((4, 1, 1)), mindspore.float32)
        >>> output = loss(logits, labels)
        >>> print(output.shape)
        (4,)
    """

    def __init__(self, samp_shape: List[int], reduction: str = "mean") -> None:
        super().__init__(reduction)
        self.mse = PerSampleMSELoss(reduction)
        self.samp_shape = tuple(samp_shape)

    def construct(self,
                  logits: Tensor,
                  labels: Tensor,
                  coordinate: Tensor) -> Tensor:
        residual = logits - labels  # [bsz, n_pts, 1]
        bsz = residual.shape[0]
        shape = (bsz,) + self.samp_shape
        residual = residual.reshape(shape)  # [bsz, n_t, n_xyz]
        coordinate = coordinate.reshape(shape + (4,))  # [bsz, n_t, n_xyz, 4]
        t_coord = coordinate[:, :, 0:1, 0]  # [bsz, n_t, 1]
        delta_t = ops.diff(t_coord, axis=1)  # [bsz, n_t - 1, 1]
        diff_t = ops.diff(residual, axis=1) / delta_t  # [bsz, n_t - 1, n_xyz]
        loss = F.sqrt(self.mse(logits, labels) + self.mse(diff_t, 0. * diff_t))
        return loss


class LossFunction(nn.Cell):
    r"""
    Computes the loss based on the specified loss type and parameters.
    This class encapsulates different types of loss functions and allows for various configurations
    such as normalization, reduction to mean, and causality weighting.

    Args:
        - sub_config (DictConfig): Configurations. config.train, config.eval or
          config.inverse in the config file.

            - 'loss' (DictConfig): Configurations for loss functions.

                - 'loss_type' (str): The type of loss to compute. Supported
                  values are 'MSE', 'RMSE', 'MAE', 'MIXED', 'H1'. Default: 'MSE'.
                - 'normalize' (bool): Whether to normalize the loss. Default: True.
                - 'normalize_eps' (float): A small value to add to the
                  denominator to avoid division by zero during normalization.
                  Default: 1.e-6.
                - 'kmax' (float): The maximum wavenumber to consider in the
                  Fourier transform. Default: 20.0.
                - 'knum' (int): Randomly sample 'knum' wavenumbers to estimate
                  norm of gradient. Default: 8192.
            - 'normalize_loss' (DictConfig): Configurations for normalization loss
              functions. Default: same as 'loss'.
            - samp_axes (str): Random downsampled axes. 't', 'xyz' or 'txyz'.
              Default: 'txyz'.
            - samp_shape (List[int]): The shape of the downsampled mesh.
              Default: [-1, 128, 128] for 't' axes, [101, -1] for 'xyz' axes.
        - reduce_mean (bool): Whether to reduce the loss to mean over the batch.
          Default: True.
    """

    def __init__(self,
                 sub_config: DictConfig,
                 reduce_mean: bool = True,
                 compute_dtype=float32) -> None:
        super().__init__()
        config_loss_main = sub_config.get("loss", OmegaConf.create({}))
        config_normalize_loss = sub_config.get("normalize_loss", config_loss_main)
        self.normalize = config_loss_main.get("normalize", True)
        self.reduce_mean = reduce_mean
        self.normalize_eps = config_loss_main.get("normalize_eps", 1.e-6)
        if self.normalize_eps <= 0:
            raise ValueError(
                f"'normalize_eps' should be a positive float, but got '{self.normalize_eps}'.")

        def get_loss_fn(config_loss: DictConfig) -> PerSampleLossBase:
            loss_type = config_loss.get("type", "RMSE").upper()
            reduction = config_loss.get("sample_reduce", "nanmean").lower()
            if loss_type == "MSE":
                loss_fn = PerSampleMSELoss(reduction)
            elif loss_type == "RMSE":
                loss_fn = PerSampleRMSELoss(reduction)
            elif loss_type == "MAE":
                loss_fn = PerSampleMAELoss(reduction)
            elif loss_type == "MIXED":
                weight = config_loss.mixed_weight
                loss_fn = PerSampleMixedLoss(weight, reduction)
            elif loss_type in ["SPEC", "SPECTRAL"]:
                loss_fn = PerSampleSpectralLoss(
                    config_loss.spectral_modes,
                    config_loss.spectral_sqrt,
                    reduction=reduction,
                    compute_dtype=compute_dtype)
            elif loss_type == "H1":
                samp_axes = sub_config.get("samp_axes", "txyz").lower()
                if samp_axes == "txyz":
                    kmax = OmegaConf.select(config_loss, "h1.kmax", default=20.0)
                    knum = OmegaConf.select(config_loss, "h1.knum", default=8192)
                    loss_fn = PerSampleH1Loss(kmax, knum, reduction)
                elif samp_axes == "t":
                    samp_shape = OmegaConf.select(sub_config, "samp_shape", default=[-1, 128, 128])
                    if len(samp_shape) not in [3, 4]:
                        raise ValueError(
                            f"Invalid samp_shape {samp_shape}. The shape should be a list of 3"
                            f" integers [n_t, n_x, n_y] or 4 integers [n_t, n_x, n_y, n_z].")
                    if len(samp_shape) == 3:
                        samp_shape.append(1)
                    loss_fn = PerSampleH1LossT(samp_shape, reduction)
                elif samp_axes == "xyz":
                    samp_shape = OmegaConf.select(sub_config, "samp_shape", default=[101, -1])
                    if len(samp_shape) != 2:
                        raise ValueError(
                            f"Invalid samp_shape {samp_shape}. The shape should be a list of 2"
                            f" integers [n_t, n_xyz].")
                    loss_fn = PerSampleH1LossXYZ(samp_shape, reduction)
            else:
                raise ValueError(
                    "'loss_type' should be one of ['MSE', 'RMSE', 'MAE', 'MIXED', 'H1'], "
                    f"but got '{loss_type}'.")
            return loss_fn

        self.sample_loss_fn = get_loss_fn(config_loss_main)
        self.sample_normalize_fn = get_loss_fn(config_normalize_loss)

    def construct(self,
                  pred: Tensor,
                  label: Tensor,
                  coordinate: Optional[Tensor] = None) -> Tensor:
        r"""construct"""
        loss = self.sample_loss_fn(pred, label, coordinate)  # [bsz]
        if self.normalize:
            label_norm = self.sample_normalize_fn(label, 0. * label, coordinate)  # [bsz]
            loss = loss / (label_norm + self.normalize_eps)  # [bsz]
        if self.reduce_mean:
            loss = ops.mean(loss)  # shape []
        return loss


class DeltaWeightPenalty(nn.Cell):
    r"""
    To surpress overfitting during model fine-tuning, we penalize the model
    weight update, i.e. the distance between the fine-tuned model weights and
    the original ones.

    Args:
        - model (nn.Cell): Target model to be fine-tuned.
        - config_dw_penalty (DictConfig): Configurations. Typically
          `config.train.dw_penalty` in the config file.

    An (Incomplete) Usage Example:
        >>> dw_penalty = DeltaWeightPenalty(model, config.train.dw_penalty)
        >>> loss = loss_fn(pred, label, coordinate)
        >>> loss = loss + dw_penalty()
        >>> loss = loss_scaler.scale(loss)
    """

    def __init__(self, model: nn.Cell, config_dw_penalty: DictConfig) -> None:
        super().__init__()
        self.current_params = model.trainable_params()
        self.original_params = [Tensor(param).copy()
                                for param in self.current_params]

        self.mode = config_dw_penalty.mode.lower()
        self.coef = Parameter(Tensor(config_dw_penalty.init_coef, float32),
                              name="dw_penalty_coef",
                              requires_grad=False)
        # self.distance_value = Parameter(
        #     Tensor(0., float32), name="distance_value", requires_grad=False)
        distance_fn = config_dw_penalty.distance_fn.upper()
        if distance_fn == "MSE":
            self.entry_fn = ops.Square()
            self.output_fn = ops.Identity()
        elif distance_fn == "MAE":
            self.entry_fn = ops.Abs()
            self.output_fn = ops.Identity()
        elif distance_fn == "RMSE":
            self.entry_fn = ops.Square()
            self.output_fn = ops.Sqrt()

    def construct(self) -> Tensor:
        r"""construct"""
        if self.mode == "disabled":
            return 0.
        param_diff = 0.
        for init_param, curr_param in zip(self.original_params, self.current_params):
            param_diff = param_diff + ops.sum(self.entry_fn(curr_param - init_param))
        param_diff = self.output_fn(param_diff)
        # self.distance_value.set_data(param_diff)
        return self.coef * param_diff

    # def get_distance(self) -> float:
    #     r"""Get the last distance value."""
    #     return self.distance_value.asnumpy().item()


if __name__ == "__main__":  # unit test
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    # test_loss_fn = PerSampleMixedLoss(0.5, "nanmean")
    # x1 = np.random.rand(3, 2, 2)
    # x1[:, 0] = np.nan
    # x2 = Tensor(np.random.rand(3, 2, 2))
    test_loss_fn = PerSampleSpectralLoss()
    x1 = Tensor(np.random.rand(2, 128*128, 1))
    x2 = Tensor(np.random.rand(2, 128*128, 1))
    raw_loss = test_loss_fn(x1, x2)
    print(raw_loss)
