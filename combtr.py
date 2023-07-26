from typing import Sequence, Tuple, Union
import torch.nn as nn
from monai.networks.nets import UNETR, SwinUNETR, SegResNet
import torch as torch
import os
import numpy as np
from monai.inferers import sliding_window_inference
from monai.utils.misc import set_determinism
set_determinism(seed=0)

def get_model_names():
    return ["swinUNETR", "SEGRESNET", "UNETR"]

def get_models(
        in_channels: 1,
        out_channels: 14,
        img_size: (96, 96, 96),
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        pos_embed: str = "conv",
        norm_name: Union[Tuple, str] = "instance",
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        segresnet_filters = 16,
        metalearnerinput = 14,
        swinfeaturesize = 48,
        use_checkpointing = True
    ):
    return [SegResNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            init_filters=segresnet_filters
        ),

        SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=swinfeaturesize,
            use_checkpoint=use_checkpointing
        ),

        UNETR(
            in_channels=in_channels,
            out_channels=out_channels
            img_size=img_size,
            feature_size=feature_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=dropout_rate
        ),
        
    )]

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
        segresnet_filters = 16,
        metalearnerinput = 14,
        swinfeaturesize = 48,
        use_checkpointing = True
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels for all base models.
            out_channels: dimension of output channels for all base models.
            img_size: dimension of input image for all models.
            num_heads: number of attention heads.
            dropout_rate: fraction of the input units to drop for all viable models.
            spatial_dims: number of spatial dims.

            """

        super().__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads")

        self.models = get_models(
            in_channels,
            out_channels,
            img_size,
            hidden_size,
            mlp_dim,
            num_heads,
            pos_embed,
            norm_name,
            dropout_rate,
            spatial_dims,
            segresnet_filters,
            metalearnerinput,
            swinfeaturesize,
            use_checkpointing
        )

        self.meta = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=swinfeaturesize,
            use_checkpoint=use_checkpointing
        ).to(self.device)

        [i.to(self.device) for i in self.models]

        for i in range(0, len(self.models)):
            self.models[i].load_state_dict(torch.load(os.path.join("./", "best" + get_model_names()[i])), strict=False)


    def forward(self, x_in):
        x_out = [i(x_in) for i in self.models]
        x_out = torch.stack(tuple(x_out), 1)
        x_out = torch.mean(x_out, dim=1)

        return self.meta(x_out)


