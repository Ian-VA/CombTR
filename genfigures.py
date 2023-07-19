from monai.inferers import sliding_window_inference
from monai.networks.nets import UNETR, SegResNet, UNet, SwinUNETR
import torch
from datautils.getdata import getdataloaders, get_valds, get_noprocess, get_valloader
import os
from monai.data import decollate_batch
from os import path
import torch
import matplotlib.pyplot as plt
import numpy as np
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from torch.cuda.amp import autocast
from monai.metrics import DiceMetric
from tqdm import tqdm
from model import CombTR
from monai.utils.misc import set_determinism
from monai.transforms import (
    Compose,
    Activations,
    AsDiscrete,
    SaveImage
)

#### THIS FILE IS USED FOR GENERATING FIGURES, NOT FOR MODEL EVALUATION

set_determinism(seed=0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = "/home/ian/Desktop/research/"

model4 = CombTR(in_channels=1, out_channels=14, img_size=(96, 96, 96)).to(device)

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

model1.load_state_dict(torch.load(os.path.join(path, "bestswinUNETR.pth")))
model3.load_state_dict(torch.load(os.path.join(path, "bestUNETR.pth")), strict=False)
model2.load_state_dict(torch.load(os.path.join(path, "bestSEGRESNET.pth")))
model_list = [model1, model2, model3, model4]

post_label = AsDiscrete(to_onehot=14)
post_pred = AsDiscrete(argmax=True, to_onehot=14)
dice_metric = DiceMetric(reduction="mean", include_background=True, get_not_nans=False)

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
    [i.eval() for i in model_list]

    val_ds = get_valds()

    with torch.no_grad():
        img_name = os.path.split(val_ds[case_num]["image"].meta["filename_or_obj"])[1]
        img = val_ds[case_num]["image"]
        label = val_ds[case_num]["label"]
        val_inputs = torch.unsqueeze(img, 1).cuda()
        val_labels = torch.unsqueeze(label, 1).cuda()
        
        with autocast():
                val_outputs = [sliding_window_inference(val_inputs, (96, 96, 96), 1, i, device="cuda") for i in model_list]

                valalloutputs = torch.stack((val_outputs), 1)
                val_outputs = torch.mean(valalloutputs, dim=1)
                
                val_outputs = sliding_window_inference(val_outputs, (96, 96, 96), 1, model4, device="cuda")

        val_labelfordice = val_labels
        val_labels = val_labels.cpu().numpy()[0, 0, :, :, slice_map[img_name]]
        val_labels = np.ma.masked_where(val_labels == 0., val_labels)

        plt.figure("check", (18, 6))
        plt.subplot(1, 5, 1)
        plt.title("Ground Truth")
        plt.imshow(val_inputs.cuda().numpy()[0, 0, :, :, slice_map[img_name]], cmap="gray")
        plt.imshow(val_labels, cmap='jet', alpha=0.5)

        plt.subplot(1, 6, 4)
        plt.title("CombTR")
        plt.imshow(val_inputs.cuda().numpy()[0, 0, :, :, slice_map[img_name]], cmap="gray")
        plt.imshow(torch.argmax(val_outputs, dim=1).detach().cuda()[0, 0, :, :, slice_map[img_name]], cmap='jet', alpha=0.5)

        plt.show()


def alldicescores():
    [i.eval() for i in model_list]
    valdl = get_valloader()
    epoch_iterator_val = tqdm(valdl, desc="Validation (dice=X.X)", dynamic_ncols=True)

    with torch.no_grad(), autocast():
        for batch in epoch_iterator_val:
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            val_outputs = [sliding_window_inference(val_inputs, (96, 96, 96), 4, i) for i in model_list]
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = [decollate_batch(i) for i in val_outputs]
            mean_dice_vals = []

            for i in val_outputs_list:
                val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in i]
                dice_metric(y_pred=val_output_convert, y=val_labels_convert)
                mean_dice_vals.append(dice_metric.aggregate().item())
                dice_metric.reset()
            epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (1, 10.0))

if __name__ == '__main__':
    illustrate()
