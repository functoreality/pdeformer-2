r"""Two-Dimensional U-Net"""
from mindspore import ops, nn, Tensor
import mindspore.common.dtype as mstype


class DoubleConv(nn.Cell):
    r"""(conv => BN => ReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int, compute_dtype=mstype.float32) -> None:
        super().__init__()
        self.conv = nn.SequentialCell(
            [nn.Conv2d(
                in_channels, out_channels, 3, pad_mode="same", has_bias=True,
                weight_init="HeUniform").to_float(compute_dtype),
             nn.BatchNorm2d(out_channels).to_float(mstype.float32),
             nn.ReLU(),
             nn.Conv2d(
                 out_channels, out_channels, 3, pad_mode="same", has_bias=True,
                 weight_init="HeUniform").to_float(compute_dtype),
             nn.BatchNorm2d(out_channels).to_float(mstype.float32),
             nn.ReLU()])

    def construct(self, x: Tensor) -> Tensor:
        r"""construct"""
        x = self.conv(x)
        return x


class InConv(nn.Cell):
    r"""input convolutional"""

    def __init__(self, in_channels: int, out_channels: int, compute_dtype=mstype.float32):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels, compute_dtype=compute_dtype)

    def construct(self, x: Tensor) -> Tensor:
        r"""construct"""
        x = self.conv(x)
        return x


class Down(nn.Cell):
    r"""downsample convolutional"""

    def __init__(self, in_channels: int, out_channels: int, compute_dtype=mstype.float32) -> None:
        super().__init__()
        self.mpconv = nn.SequentialCell([
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels, compute_dtype=compute_dtype)
        ])

    def construct(self, x: Tensor) -> Tensor:
        '''construct'''
        x = self.mpconv(x)
        return x


class Up(nn.Cell):
    r"""upsample convolutional"""

    def __init__(self, in_channels: int, out_channels: int, compute_dtype=mstype.float32) -> None:
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels, compute_dtype=compute_dtype)

    def construct(self, x1: Tensor, x2: Tensor) -> Tensor:
        r"""
        Args:
            - x1 (Tensor): Low resolution input tensor.
            - x2 (Tensor): High resolution input tensor.
        """
        height = x1.shape[2]
        width = x1.shape[3]
        x1 = ops.interpolate(x1, size=(2 * height, 2 * width), mode="bilinear")

        x = ops.concat([x2, x1], axis=1)
        x = self.conv(x)
        return x


class OutConv(nn.Cell):
    r"""output convolutional"""

    def __init__(self, in_channels: int, out_channels: int, compute_dtype=mstype.float32) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                              has_bias=True, weight_init="HeUniform").to_float(compute_dtype)

    def construct(self, x: Tensor) -> Tensor:
        r"""construct"""
        x = self.conv(x)
        return x


class UNet2D(nn.Cell):
    r"""
    The 2-dimensional U-Net model.
    U-Net is a U-shaped convolutional neural network for biomedical image segmentation.
    It has a contracting path that captures context and an expansive path that enables
    precise localization. The details can be found in `U-Net: Convolutional Networks for
    Biomedical Image Segmentation <https://arxiv.org/abs/1505.04597>`.

    Args:
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.
        data_nhwc (bool): whether to use NHWC data format. default: False, use NCHW.

    Inputs:
        - **x** (Tensor) - input tensor, shape is :math:`(batch\_size, C_{in}, H, W)`,
          or :math:`(batch\_size, H, W, C_{in})` if `data_nhwc` is True.
          `H` and `W` must be divisible by 16.

    Outputs:
        Tensor, the output of this network.

        - **output** (Tensor) - output tensor, shape is :math:`(batch\_size, C_{out}, H, W)`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from model.baseline.unet2d import UNet2D
        >>> unet = UNet2D(2, 2)
        >>> x = Tensor(np.random.rand(8, 2, 256, 256), mindspore.float32) # [B, C, H, W]
        >>> y = unet(x) # [B, C, H, W]
        >>> print(y.shape)
        (8, 2, 256, 256)

    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 data_nhwc: bool = False,
                 compute_dtype=mstype.float32) -> None:
        super().__init__()
        self.data_nhwc = data_nhwc

        self.inc = InConv(in_channels, 64, compute_dtype=compute_dtype)
        self.down1 = Down(64, 128, compute_dtype=compute_dtype)
        self.down2 = Down(128, 256, compute_dtype=compute_dtype)
        self.down3 = Down(256, 512, compute_dtype=compute_dtype)
        self.down4 = Down(512, 512, compute_dtype=compute_dtype)
        self.up1 = Up(1024, 256, compute_dtype=compute_dtype)
        self.up2 = Up(512, 128, compute_dtype=compute_dtype)
        self.up3 = Up(256, 64, compute_dtype=compute_dtype)
        self.up4 = Up(128, 64, compute_dtype=compute_dtype)
        self.outc = OutConv(64, out_channels, compute_dtype=compute_dtype)

    def construct(self, x: Tensor, _: Tensor) -> Tensor:
        r"""construct"""
        if self.data_nhwc:
            x = ops.transpose(x, (0, 3, 1, 2))  # [B, H, W, C] -> [B, C, H, W]
        x1 = self.inc(x)  # [B, C, H, W]
        x2 = self.down1(x1)  # [B, C, H/2, W/2]
        x3 = self.down2(x2)  # [B, C, H/4, W/4]
        x4 = self.down3(x3)  # [B, C, H/8, W/8]
        x5 = self.down4(x4)  # [B, C, H/16, W/16]
        x = self.up1(x5, x4)  # [B, C, H/8, W/8]
        x = self.up2(x, x3)  # [B, C, H/4, W/4]
        x = self.up3(x, x2)  # [B, C, H/2, W/2]
        x = self.up4(x, x1)  # [B, C, H, W]
        x = self.outc(x)  # [B, C, H, W]
        if self.data_nhwc:
            x = ops.transpose(x, (0, 2, 3, 1))  # [B, C, H, W] -> [B, H, W, C]
        return x
