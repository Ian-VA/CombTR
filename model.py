from typing import Sequence, Tuple, Union

import torch.nn as nn

from monai.networks.nets import UNETR
from monai.networks.nets import unet
from monai.networks.nets import swin_unetr

class CombTR(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Union[Sequence[int], int],
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "conv",
        norm_name: Union[Tuple, str] = "instance",
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        qkv_bias: bool = False,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dims.
        """

        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads")

        self.unetrmodel1 = UNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            feature_size=feature_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            pos_embed=pos_embed,
            norm_name=norm_name,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
            qkv_bias=qkv_bias
        )

        self.unetrmodel2 = UNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            feature_size=feature_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            pos_embed=pos_embed,
            norm_name=norm_name,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
            qkv_bias=qkv_bias
        )

        self.swinunetr = swin_unetr(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size
        )

        self.swinunetr2 = swin_unetr(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size
        )


    def forward(self, x_in):
        return x_in