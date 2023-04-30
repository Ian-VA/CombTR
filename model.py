from typing import Sequence, Tuple, Union
import torch.nn as nn
from monai.networks.nets import UNETR, SwinUNETR, SegResNet
from metalearner import CombTRMetaLearner
import torch as torch
import os
import numpy as np
from monai.inferers import sliding_window_inference
from monai.utils.misc import set_determinism

set_determinism(seed=0)

class CombTR(nn.Module):
    def __init__(
        self,
        in_channels: 1,
        out_channels: 14,
        img_size: (96, 96, 96),
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "conv",
        norm_name: Union[Tuple, str] = "instance",
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        qkv_bias: bool = False,
        segresnet_filters = 16
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
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads")

        self.segresnet = SegResNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            init_filters=segresnet_filters
        ).to(device)

        self.swinunetr = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=True,
        ).to(device)

        self.unetr = UNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            feature_size=feature_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0,
        ).to(device)
        
        self.meta = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=True,
        ).to(device)

        """
        Uncomment if reproducing results in research paper:

        self.swinunetr.load_state_dict(torch.load(os.path.join("./", "best_swinUNETR.pth")))
        self.segresnet.load_state_dict(torch.load(os.path.join("./", "bestSEGRESNET.pth")))
        self.unetr.load_state_dict(torch.load(os.path.join("./", "realunetrmodel.pth")), strict=False)
        self.meta.load_state_dict(torch.load(os.path.join("./", "best_CombTR.pth")))
        """


    def forward(self, x_in):
        x_out1 = self.unetr(x_in)
        x_out2 = self.swinunetr(x_in)
        x_out3 = self.segresnet(x_in)

        x_out = torch.stack((x_out1, x_out2, x_out3), 1)
        x_out = torch.mean(x_out, dim=1)

        return self.meta(x_out)


