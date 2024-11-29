"""Loss functions."""
from typing import Optional, List
from abc import abstractmethod

import numpy as np
from omegaconf import OmegaConf, DictConfig
from mindspore import nn, ops, Tensor
from mindspore import context
from mindspore.ops import functional as F


class PerSampleLossBase(nn.LossBase):
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

    @abstractmethod
    def construct(self,
                  logits: Tensor,
                  labels: Tensor,
                  coordinate: Optional[Tensor]) -> Tensor:
        r"""construct"""

    def get_axis(self, x: Tensor):
        shape = F.shape(x)
        length = F.tuple_len(shape)
        # The only difference compared with nn.LossBase: The axis starts from 1
        # instead of 0, thereby avoid reduction over the samples (axis 0).
        perm = F.make_range(1, length)
        return perm


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


class PerSampleRMSELoss(PerSampleLossBase):
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

    def __init__(self) -> None:
        """Initialize PerSampleRMSELoss."""
        super().__init__()
        self.persample_mse_loss = PerSampleMSELoss()

    def construct(self,
                  logits: Tensor,
                  labels: Tensor,
                  coordinate: Optional[Tensor] = None) -> Tensor:
        rmse_loss = F.sqrt(self.persample_mse_loss(logits, labels))
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

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.mae_weight = weight
        self.rmse_weight = 1. - weight
        self.mae = PerSampleMAELoss()
        self.rmse = PerSampleRMSELoss()

    def construct(self,
                  logits: Tensor,
                  labels: Tensor,
                  coordinate: Optional[Tensor] = None) -> Tensor:
        rmse = self.rmse(logits, labels)
        mae = self.mae(logits, labels)
        loss = rmse * self.rmse_weight + mae * self.mae_weight
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

    def __init__(self, kmax: float = 10.0, knum: int = 8192):
        super().__init__()
        self.kmax = kmax
        self.knum = knum
        self.mse = PerSampleMSELoss()

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

    def __init__(self, samp_shape: List[int]):
        super().__init__()
        self.mse = PerSampleMSELoss()
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

    def __init__(self, samp_shape: List[int]):
        super().__init__()
        self.mse = PerSampleMSELoss()
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
                 reduce_mean: bool = True) -> None:
        super().__init__()
        config_loss = sub_config.get('loss', OmegaConf.create({}))
        config_normalize_loss = sub_config.get('normalize_loss', config_loss)
        self.normalize = config_loss.get('normalize', True)
        self.reduce_mean = reduce_mean
        self.normalize_eps = config_loss.get('normalize_eps', 1.e-6)
        if self.normalize_eps <= 0:
            raise ValueError(
                f"'normalize_eps' should be a positive float, but got '{self.normalize_eps}'.")

        def get_loss_fn(config_loss: DictConfig) -> PerSampleLossBase:
            loss_type = config_loss.get('type', 'RMSE').upper()
            if loss_type == 'MSE':
                loss_fn = PerSampleMSELoss()
            elif loss_type == 'RMSE':
                loss_fn = PerSampleRMSELoss()
            elif loss_type == 'MAE':
                loss_fn = PerSampleMAELoss()
            elif loss_type == 'MIXED':
                weight = OmegaConf.select(config_loss, 'mixed.weight', default=0.1)
                loss_fn = PerSampleMixedLoss(weight)
            elif loss_type == 'H1':
                samp_axes = sub_config.get('samp_axes', 'txyz').lower()
                if samp_axes == 'txyz':
                    kmax = OmegaConf.select(config_loss, 'h1.kmax', default=20.0)
                    knum = OmegaConf.select(config_loss, 'h1.knum', default=8192)
                    loss_fn = PerSampleH1Loss(kmax, knum)
                elif samp_axes == 't':
                    samp_shape = OmegaConf.select(sub_config, 'samp_shape', default=[-1, 128, 128])
                    if len(samp_shape) not in [3, 4]:
                        raise ValueError(
                            f"Invalid samp_shape {samp_shape}. The shape should be a list of 3"
                            f" integers [n_t, n_x, n_y] or 4 integers [n_t, n_x, n_y, n_z].")
                    if len(samp_shape) == 3:
                        samp_shape.append(1)
                    loss_fn = PerSampleH1LossT(samp_shape)
                elif samp_axes == 'xyz':
                    samp_shape = OmegaConf.select(sub_config, 'samp_shape', default=[101, -1])
                    if len(samp_shape) != 2:
                        raise ValueError(
                            f"Invalid samp_shape {samp_shape}. The shape should be a list of 2"
                            f" integers [n_t, n_xyz].")
                    loss_fn = PerSampleH1LossXYZ(samp_shape)
            else:
                raise ValueError(
                    "'loss_type' should be one of ['MSE', 'RMSE', 'MAE', 'MIXED', 'H1'], "
                    f"but got '{loss_type}'.")
            return loss_fn

        self.sample_loss_fn = get_loss_fn(config_loss)
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


if __name__ == "__main__":  # unit test
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    test_loss_fn = PerSampleMixedLoss()
    x1 = Tensor(np.random.rand(3, 2, 2))
    x2 = Tensor(np.random.rand(3, 2, 2))
    raw_loss = test_loss_fn(x1, x2)
    print(raw_loss)
