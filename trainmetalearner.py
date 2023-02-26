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
from monai.metrics import DiceMetric
from tqdm import tqdm
from metalearner import CombTRMetaLearner

from monai.transforms import (
    Compose,
    Activations,
    AsDiscrete,
    SaveImage
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNETR(
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

trainmodel = CombTRMetaLearner(
    n_channels=14,
    n_classes=14
).to(device)

loss_function = DiceCELoss(to_onehot_y=True, softmax=True) 
torch.backends.cudnn.benchmark = True
optimizer = torch.optim.AdamW(trainmodel.parameters(), lr=1e-4, weight_decay=1e-5)


#model.load_state_dict(torch.load(os.path.join("./", "realunetrmodel.pth"), strict=False))
#model1.load_state_dict(torch.load(os.path.join("./", "best_swinUNETR.pth"), strict=False))
#model2.load_state_dict(torch.load(os.path.join("./", "bestSEGRESNET.pth"), strict=False))

def validation(epoch_iterator_val):
    model.eval()
    with torch.no_grad():
        for batch in epoch_iterator_val:
            val_inputs, val_labels = batch["image"].cuda(), batch["label"].cuda()
            val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 1, trainmodel)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (global_step, 10.0))
        mean_dice_val = dice_metric.aggregate().item()
        dice_metric.reset()
    return mean_dice_val


def train(global_step, train_loader, val_loader, dice_val_best, global_step_best):
    trainmodel.train()
    epoch_loss = 0
    step = 0
    model2.eval()
    model1.eval()
    model.eval()

    epoch_iterator = tqdm(train_loader, desc="Training (loss=X.X)", dynamic_ncols=True)
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["image"], batch["label"])

        val_outputs3 = model(x)
        val_outputs1 = model1(x)
        val_outputs2 = model2(x)

        valalloutputs = torch.cat((val_outputs1, val_outputs2, val_outputs3), 0)
        val_outputs = torch.mean(valalloutputs, dim=0)
        val_outputs.unsqueeze(0)
        
        val_outputs = trainmodel(val_outputs)

        loss = loss_function(val_outputs, y)
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        epoch_iterator.set_description("Training (loss=%2.5f)" % (loss))
        if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
            epoch_iterator_val = tqdm(val_loader, desc="Validation (dice=X.X)", dynamic_ncols=True)
            dice_val = validation(epoch_iterator_val)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                torch.save(trainmodel.state_dict(), os.path.join(datadir, "best_CombTR.pth"))
                print(
                    "Saved Model! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best, dice_val)
                )
            else:
                print(
                    "Model was not saved. Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
        global_step += 1
    return global_step, dice_val_best, global_step_best



def make_stackingdata(jsonfilename="C:/Users/mined/Desktop/projects/segmentationv2/stackingdata.json"):
    trainloader, val_loader = getdataloaders()
    model.eval()
    model1.eval()
    model2.eval()
    i = 0
    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            if path.isfile(jsonfilename) is False:
                raise Exception("json datafile not found")

            val_inputs, val_labels = (batch["image"], batch["label"])
            val_outputs3 = sliding_window_inference(val_inputs, (96, 96, 96), 1, model)
            val_outputs1 = sliding_window_inference(val_inputs, (96, 96, 96), 1, model1)
            val_outputs2 = sliding_window_inference(val_inputs, (96, 96, 96), 1, model2)

            valalloutputs = torch.cat((val_outputs1, val_outputs2, val_outputs3), 0)
            val_outputs = torch.softmax(valalloutputs, 1).cpu().numpy()
            val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)[0]
            val_outputs = torch.from_numpy(val_outputs)
            val_labels = val_labels.cpu().numpy()[0, 0, :, :, :]






def plotdata(root_dir):
    val_ds = get_valds()
    no_process_ds = get_noprocess()
    slice_map = {
        "img0035.nii.gz": 170,
        "img0036.nii.gz": 230,
        "img0037.nii.gz": 204,
        "img0038.nii.gz": 204,
        "img0039.nii.gz": 204,
        "img0040.nii.gz": 180,
    }

    nifti = nib.load("C:/Users/mined/Downloads/stackingdata/labelsTr/label_2.nii.gz").get_fdata()
    
    plt.figure("image", (18, 6))
    plt.subplot(1, 2, 1)
    plt.title("After Preprocessing")
    plt.imshow(nifti[:, :, 59])
    plt.show()


if __name__ == "__main__":
    datadir = "C:/Users/mined/Desktop/projects/segmentationv2/" # replace with your dataset directory
    max_iterations = 25000
    eval_num = 500
    post_label = AsDiscrete(to_onehot=14)
    post_pred = AsDiscrete(argmax=True, to_onehot=14)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    global_step = 0
    dice_val_best = 0.0
    global_step_best = 0
    epoch_loss_values = []
    metric_values = []
    train_loader, val_loader = getdataloaders()

    train(global_step, train_loader, val_loader, dice_val_best, global_step_best)