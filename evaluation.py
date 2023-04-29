from monai.inferers import sliding_window_inference
from monai.networks.nets import UNETR, SegResNet, UNet, SwinUNETR
import torch
from datautils.getdataloader import getdataloaders, get_valds, get_noprocess, get_valloader
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


device = torch.device("cpu")
torch.manual_seed(42)

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

control = SwinUNETR(
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

model1.load_state_dict(torch.load(os.path.join("./", "best_swinUNETR.pth"), map_location=torch.device('cpu')))
model3.load_state_dict(torch.load(os.path.join("./", "realunetrmodel.pth"), map_location=torch.device('cpu')), strict=False)
model2.load_state_dict(torch.load(os.path.join("./", "bestSEGRESNET.pth"), map_location=torch.device('cpu')))
trainmodel.load_state_dict(torch.load(os.path.join("./", "best_CombTR.pth"), map_location=torch.device('cpu')))

case_num = 5
post_label = AsDiscrete(to_onehot=14)
post_pred = AsDiscrete(argmax=True, to_onehot=14)

dice_metric = DiceMetric(reduction="mean_channel", include_background=True, get_not_nans=False, ignore_empty=True)


def checkdice():

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
    val_ds = get_valds()

    with torch.no_grad():
        img_name = os.path.split(val_ds[case_num]["image"].meta["filename_or_obj"])[1]
        img = val_ds[case_num]["image"]
        label = val_ds[case_num]["label"]
        val_inputs = torch.unsqueeze(img, 1)
        val_labels = torch.unsqueeze(label, 1)
        
        with autocast():
                val_outputs1 = sliding_window_inference(val_inputs, (96, 96, 96), 1, model1, device="cpu")


    segoutput = torch.argmax(val_outputs1, dim=1).detach().cpu()
    val_labelfordice = torch.squeeze(val_labelfordice, 1)
    dice_metric(post_pred(segoutput), post_label(val_labelfordice))
    print(dice_metric.aggregate())

def illustrateseperateclasses():

    control.eval()
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
    control.eval()

    val_ds = get_valds()
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
                controloutput = sliding_window_inference(val_inputs, (96, 96, 96), 1, control, device="cpu")

                valalloutputs = torch.stack((val_outputs1, val_outputs2, val_outputs3), 1)
                val_outputs = torch.mean(valalloutputs, dim=1)
                
                val_outputs = sliding_window_inference(val_outputs, (96, 96, 96), 1, trainmodel, device="cpu")

        val_labelfordice = val_labels
        val_labels = val_labels.cpu().numpy()[0, 0, :, :, slice_map[img_name]]
        val_labels = np.ma.masked_where(val_labels == 0., val_labels)

        """
        outputsallclasses = []
        diceallclasses = []

        for i in range(1, 14):
            with autocast():
                outputsallclasses.append(torch.from_numpy(np.ma.masked_where(val_outputs != i, val_outputs)))
                corresponding_label = torch.from_numpy(np.ma.masked_where(val_labelfordice != i, val_labelfordice))

                dice_metric([post_pred(outputsallclasses[-1])], [post_label(corresponding_label)])
                corresponding_label.detach()
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

        """



        plt.figure("check", (18, 6))
        plt.subplot(1, 5, 1)
        plt.title("Ground Truth")
        plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_map[img_name]], cmap="gray")
        plt.imshow(val_labels, cmap='jet', alpha=0.5)


        segoutput = torch.argmax(val_outputs2, dim=1).detach().cpu()
        val_labelfordice = torch.squeeze(val_labelfordice, 1)
        dice_metric(post_pred(segoutput), post_label(val_labelfordice))
        print(dice_metric.aggregate())

        plt.subplot(1, 6, 1)
        plt.title("SegResNet")
        plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_map[img_name]], cmap="gray")
        plt.imshow(torch.argmax(val_outputs2, dim=1).detach().cpu()[0, :, :, slice_map[img_name]], cmap='jet', alpha=0.5)


        swinoutput = torch.argmax(val_outputs1, dim=1).detach().cpu()

        plt.subplot(1, 6, 2)
        plt.title("Swin UNETR")
        plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_map[img_name]], cmap="gray")
        plt.imshow(torch.argmax(val_outputs1, dim=1).detach().cpu()[0, :, :, slice_map[img_name]], cmap='jet', alpha=0.5)


        unetroutput = torch.argmax(val_outputs3, dim=1).detach().cpu()

        plt.subplot(1, 6, 3)
        plt.title("UNETR")
        plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_map[img_name]], cmap="gray")
        plt.imshow(unetroutput[0, :, :, slice_map[img_name]], cmap='jet', alpha=0.5)

        combtroutput = torch.argmax(val_outputs, dim=1).detach().cpu()

        plt.subplot(1, 6, 4)
        plt.title("CombTR")
        plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_map[img_name]], cmap="gray")
        plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, slice_map[img_name]], cmap='jet', alpha=0.5)

        plt.subplot(1, 6, 4)
        plt.title("CombTR")
        plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_map[img_name]], cmap="gray")
        plt.imshow(torch.argmax(controloutput, dim=1).detach().cpu()[0, :, :, slice_map[img_name]], cmap='jet', alpha=0.5)



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

    valdl = get_valloader()
    epoch_iterator_val = tqdm(valdl, desc="Validation (dice=X.X)", dynamic_ncols=True)

    with autocast():
        with torch.no_grad():
            for batch in epoch_iterator_val:
                val_inputs, val_labels = (batch["image"].cpu(), batch["label"].cpu())
                val_outputs1 = sliding_window_inference(val_inputs, (96, 96, 96), 1, model1)
                val_labels_list = decollate_batch(val_labels)
                val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
                val_outputs_list = decollate_batch(val_outputs1)
                val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
                dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            mean_dice_val1 = dice_metric.aggregate()
            dice_metric.reset()

    print(mean_dice_val1)

    #SEGRESNET

    with torch.no_grad():
        for batch in epoch_iterator_val:
            val_inputs, val_labels = batch["image"].cpu(), batch["label"].cpu()
            val_outputs2 = sliding_window_inference(val_inputs, (96, 96, 96), 1, model2)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(val_outputs2)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
        mean_dice_val2 = dice_metric.aggregate()
        dice_metric.reset()

    print(mean_dice_val2)



    #UNETR
    with torch.no_grad():
        for batch in epoch_iterator_val:
            val_inputs, val_labels = batch["image"].cpu(), batch["label"].cpu()
            val_outputs3 = sliding_window_inference(val_inputs, (96, 96, 96), 1, model3)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(val_outputs3)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
        mean_dice_val3 = dice_metric.aggregate()
        dice_metric.reset()

    print(mean_dice_val3)

    #COMBTR
    with torch.no_grad():
        for batch in epoch_iterator_val:
            valalloutputs = torch.stack((val_outputs1, val_outputs2, val_outputs3), 1)
            val_inputs = torch.mean(valalloutputs, dim=1)

            val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 1, trainmodel)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
        mean_dice_val4 = dice_metric.aggregate()
        dice_metric.reset()

    print(mean_dice_val4)

    valds = get_valds()
    val_inputs, val_labels = valds[case_num]["image"], valds[case_num]["label"]
    img_name = os.path.split(valds[case_num]["image"].meta["filename_or_obj"])[1]

    plt.figure("check", (18, 6))
    plt.subplot(1, 5, 1)
    plt.title("Ground Truth")
    plt.imshow(val_inputs.cpu().numpy()[0, :, :, slice_map[img_name]], cmap="gray")
    plt.imshow(val_labels, cmap='jet', alpha=0.5)
    plt.subplot(1, 5, 2)
    plt.title("SwinUNETR Output: " + str(round(mean_dice_val1[0], 2)))
    plt.imshow(val_inputs.cpu().numpy()[0, :, :, slice_map[img_name]], cmap="gray")
    plt.imshow(val_outputs1, cmap='jet', alpha=0.5)
    plt.subplot(1, 5, 3)
    plt.title("SegResNet Output: " + str(round(mean_dice_val2[0])))
    plt.imshow(val_inputs.cpu().numpy()[0, :, :, slice_map[img_name]], cmap="gray")
    plt.imshow(val_outputs2, cmap='jet', alpha=0.5)
    plt.subplot(1, 5, 4)
    plt.title("UNETR Output: " + str(round(mean_dice_val3[0])))
    plt.imshow(val_inputs.cpu().numpy()[0, :, :, slice_map[img_name]], cmap="gray")
    plt.imshow(val_outputs3, cmap='jet', alpha=0.5)
    plt.subplot(1, 5, 5)
    plt.title("CombTR Output: " + str(round(mean_dice_val4[0])))
    plt.imshow(val_inputs.cpu().numpy()[0, :, :, slice_map[img_name]], cmap="gray")
    plt.imshow(val_outputs, cmap='jet', alpha=0.5)
    plt.show()


if __name__ == '__main__':
    illustrateseperateclasses()()