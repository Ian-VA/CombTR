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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainmodel = SwinUNETR(
    img_size=(96, 96, 96),
    in_channels=14,
    out_channels=14,
    feature_size=48,
    use_checkpoint=True,
).to(device)


model2 = SegResNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=14,
    init_filters=16
).to(device)


model1 = SwinUNETR(
    img_size=(96, 96, 96),
    in_channels=1,
    out_channels=14,
    feature_size=48,
    use_checkpoint=True,
).to(device)

model3 = UNETR(
    in_channels=1,
    out_channels=14,
    img_size=(96, 96, 96),
    feature_size=16,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    pos_embed="perceptron",
    norm_name="instance",
    res_block=True,
    dropout_rate=0.0,
).to(device)

loss_function = DiceCELoss(to_onehot_y=True, softmax=True) 
optimizer = torch.optim.AdamW(trainmodel.parameters(), lr=1e-4, weight_decay=1e-5)

model1.load_state_dict(torch.load(os.path.join("C:/Users/mined/Downloads/", "best_swinUNETR.pth"), map_location=torch.device('cpu')), strict=False)
model3.load_state_dict(torch.load(os.path.join("C:/Users/mined/Downloads/", "realunetrmodel.pth"), map_location=torch.device('cpu')), strict=False)
model2.load_state_dict(torch.load(os.path.join("C:/Users/mined/Downloads/", "bestSEGRESNET.pth"), map_location=torch.device('cpu')), strict=False)
trainmodel.load_state_dict(torch.load(os.path.join("C:/Users/mined/Downloads/", "best_CombTR.pth"), map_location=torch.device('cpu')), strict=False)

case_num = 4
post_label = AsDiscrete(to_onehot=14)
post_pred = AsDiscrete(argmax=True, to_onehot=14)

dice_metric = DiceMetric(include_background=True, reduction="none", get_not_nans=False)

def illustrateseperateclasses():
    slice_map = {
        "img0035.nii.gz": 170,
        "img0036.nii.gz": 230,
        "img0037.nii.gz": 204,
        "img0038.nii.gz": 204,
        "img0039.nii.gz": 204,
        "img0040.nii.gz": 180,
    }
    case_num = 0

    model1.eval()
    model2.eval()
    model3.eval()
    trainmodel.eval()

    val_ds = get_valds()
    with torch.no_grad():
        img_name = os.path.split(val_ds[case_num]["image"].meta["filename_or_obj"])[1]
        img = val_ds[case_num]["image"]
        label = val_ds[case_num]["label"]
        val_inputs = torch.unsqueeze(img, 1)
        val_labels = torch.unsqueeze(label, 1)
        
        with torch.no_grad():
            img_name = os.path.split(val_ds[case_num]["image"].meta["filename_or_obj"])[1]
            img = val_ds[case_num]["image"]
            label = val_ds[case_num]["label"]
            val_inputs = torch.unsqueeze(img, 1)
            val_labels = torch.unsqueeze(label, 1)

            with autocast():
                val_outputs1 = sliding_window_inference(val_inputs, (96, 96, 96), 1, model1, device="cpu")
                val_outputs2 = sliding_window_inference(val_inputs, (96, 96, 96), 1, model2, device="cpu")
                val_outputs3 = sliding_window_inference(val_inputs, (96, 96, 96), 1, model3,  device="cpu")

                valalloutputs = torch.stack((val_outputs1, val_outputs2, val_outputs3), 1)
                val_outputs = torch.mean(valalloutputs, dim=1)
                
                val_outputs = sliding_window_inference(val_outputs, (96, 96, 96), 1, trainmodel, device="cpu")

        val_labelfordice = val_labels
        val_labels = val_labels.cpu().numpy()[0, 0, :, :, slice_map[img_name]]
        val_labels = np.ma.masked_where(val_labels == 0., val_labels)


        outputsallclasses = []
        diceallclasses = []

        for i in range(1, 14):
            outputsallclasses.append(torch.from_numpy(np.ma.masked_where(val_outputs != i, val_labelfordice)))
            corresponding_label = val_labelfordice
            corresponding_label = torch.from_numpy(np.ma.masked_where(corresponding_label != i, corresponding_label))

            dice_metric([post_pred(outputsallclasses[-1])], [post_label(corresponding_label)])
            mean_dice_val = dice_metric.aggregate()
            mean_dice_val = mean_dice_val.data[0]
            print(mean_dice_val.data[0])
            dice_metric.reset()
            diceallclasses.append(mean_dice_val.data[0].item())

            print(diceallclasses[-1])




        fig, axs = plt.subplots(2, 7, figsize=(20, 30))
        count = 0
        names = ["Spleen", "RKidney", "LKidney", "Gallbladder", "Esophagus", "Liver", "Stomach", "Aorta", "IVC", "PV & SV", "Pancreas", "RAdrenal", "LAdrenal"]

        for ax in axs.reshape(-1):
                if (count == 13):
                    ax.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_map[img_name]], cmap="gray")
                    ax.imshow(val_labels, cmap='jet', alpha=0.7)
                    ax.set_title("Ground Truth")
                    ax.set_aspect('auto')
                    break


                ax.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_map[img_name]], cmap="gray", interpolation='nearest')
                ax.imshow(torch.argmax(outputsallclasses[count], dim=1).detach().cpu()[0, :, :, slice_map[img_name]], cmap='jet', alpha=0.7)
                ax.set_title(names[count] + ": " + str(round(diceallclasses[count], 2)))
                ax.set_aspect('auto')

                count += 1

        

        plt.show()



        plt.figure("check", (18, 6))
        plt.subplot(1, 4, 1)
        plt.title("SegResNet")
        plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_map[img_name]], cmap="gray")
        plt.imshow(torch.argmax(val_outputs2, dim=1).detach().cpu()[0, :, :, slice_map[img_name]], cmap='jet', alpha=0.5)

        plt.subplot(1, 4, 2)
        plt.title("Swin UNETR")
        plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_map[img_name]], cmap="gray")
        plt.imshow(torch.argmax(val_outputs1, dim=1).detach().cpu()[0, :, :, slice_map[img_name]], cmap='jet', alpha=0.5)


        plt.subplot(1, 4, 3)
        plt.title("UNETR")
        plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_map[img_name]], cmap="gray")
        plt.imshow(torch.argmax(val_outputs3, dim=1).detach().cpu()[0, :, :, slice_map[img_name]], cmap='jet', alpha=0.5)


        plt.subplot(1, 4, 4)
        plt.title("CombTR")

        plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_map[img_name]], cmap="gray")
        plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, slice_map[img_name]], cmap='jet', alpha=0.5)


        plt.show()


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

    model1.eval()
    model2.eval()
    model3.eval()
    trainmodel.eval()

    val_ds = get_valds()

    with torch.no_grad():
        img_name = os.path.split(val_ds[case_num]["image"].meta["filename_or_obj"])[1]
        img = val_ds[case_num]["image"]
        label = val_ds[case_num]["label"]
        val_inputs = torch.unsqueeze(img, 1)
        val_labels = torch.unsqueeze(label, 1)
        val_labels = val_labels.cpu().numpy()[0, 0, :, :, slice_map[img_name]]
        val_labels = np.ma.masked_where(val_labels == 0., val_labels)


        with autocast():
            val_outputs1 = sliding_window_inference(val_inputs, (96, 96, 96), 1, model1, device="cpu")
            val_outputs2 = sliding_window_inference(val_inputs, (96, 96, 96), 1, model2, device="cpu")
            val_outputs3 = sliding_window_inference(val_inputs, (96, 96, 96), 1, model3,  device="cpu")

            valalloutputs = torch.stack((val_outputs1, val_outputs2, val_outputs3), 1)
            val_outputs = torch.mean(valalloutputs, dim=1)
            
            val_outputs = sliding_window_inference(val_outputs, (96, 96, 96), 1, trainmodel, device="cpu")

        plt.figure("check", (18, 6))
        plt.subplot(1, 3, 1)
        plt.title("Image")
        plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_map[img_name]], cmap="gray")
        plt.subplot(1, 3, 2)
        plt.title("Ground Truth")
        plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_map[img_name]], cmap="gray")
        plt.imshow(val_labels, cmap='jet', alpha=0.5)
        plt.subplot(1, 3, 3)
        plt.title("CombTR Output")
        plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_map[img_name]], cmap="gray")
        plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, slice_map[img_name]], cmap='jet', alpha=0.5)
        plt.show()


illustrateseperateclasses()