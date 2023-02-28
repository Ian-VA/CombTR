from monai.inferers import sliding_window_inference
from monai.networks.nets import UNETR, SegResNet, UNet, SwinUNETR
import torch
from datautils.getdataloader import getdataloaders, get_valds, get_noprocess
import os
from monai.data import decollate_batch
from os import path
import torch
import matplotlib.pyplot as plt
import json
import numpy as np
import nibabel as nib
import scipy.ndimage as ndimage
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from torch.cuda.amp import autocast
from monai.metrics import DiceMetric
from tqdm import tqdm
from metalearner import CombTRMetaLearner
import pandas as pd
from model import CombTR

from monai.transforms import (
    Compose,
    Activations,
    AsDiscrete,
    SaveImage
)

model = CombTR(
    in_channels=1,
    img_size=(96, 96, 96),
    out_channels=14
)

case_num = 4

def illustrate():
    slice_map = {
        "img0035.nii.gz": 170,
        "img0036.nii.gz": 230,
        "img0037.nii.gz": 204,
        "img0038.nii.gz": 204,
        "img0039.nii.gz": 204,
        "img0040.nii.gz": 180,
    }
    case_num = 0

    model.eval()
    val_ds = get_valds()

    with torch.no_grad():
        img_name = os.path.split(val_ds[case_num]["image"].meta["filename_or_obj"])[1]
        img = val_ds[case_num]["image"]
        label = val_ds[case_num]["label"]
        val_inputs = torch.unsqueeze(img, 1)
        val_labels = torch.unsqueeze(label, 1)
        val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model, overlap=0.8)
        plt.figure("check", (18, 6))
        plt.subplot(1, 3, 1)
        plt.title("image")
        plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_map[img_name]], cmap="gray")
        plt.subplot(1, 3, 2)
        plt.title("label")
        plt.imshow(val_labels.cpu().numpy()[0, 0, :, :, slice_map[img_name]])
        plt.subplot(1, 3, 3)
        plt.title("output")
        plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, slice_map[img_name]])
        plt.show()


illustrate()