import os
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import UNETR, SwinUNETR, UNet, SegResNet
from monai.apps import DecathlonDataset
import torch
from monai.data import decollate_batch
from tqdm import tqdm
from monai.transforms import AsDiscrete
from datautils.getdata import getdataloaders
from torchvision.models.segmentation import deeplabv3_resnet50
from model import CombTR
import pandas as pd
from monai.utils.misc import set_determinism
from model import CombTR

### FILE USED FOR CHECKING DICE SCORE OF ALL MODELS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
post_label = AsDiscrete(to_onehot=14)
post_pred = AsDiscrete(argmax=True, to_onehot=14)
dice_metric = DiceMetric(include_background=True, reduction="none", get_not_nans=False) # reduction set for class-wise dice score, use "mean" for mean of all classes
set_determinism(seed=0) 
loss_function = DiceCELoss(to_onehot_y=True, softmax=True) 
torch.backends.cudnn.benchmark = True

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

model3 = CombTR(1, 14, (96, 96, 96)).to(device)


train_loader, val_loader = getdataloaders()
epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)

model1.load_state_dict(torch.load(os.path.join("./", "bestswinUNETR.pth")))
model.load_state_dict(torch.load(os.path.join("./", "bestUNETR.pth"), strict=False))
model2.load_state_dict(torch.load(os.path.join("./", "bestSEGRESNET.pth")))


def validation(epoch_iterator_val):
    model.eval()
    with torch.no_grad():
        for batch in epoch_iterator_val:
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            with torch.cuda.amp.autocast():
                val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model)
                val_labels_list = decollate_batch(val_labels)
                val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
                val_outputs_list = decollate_batch(val_outputs)
                val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
                dice_metric(y_pred=val_output_convert, y=val_labels_convert)

                for tensoroutput in val_output_convert: # get this off the GPU, not needed if you have enough memory
                    tensoroutput.detach().cpu()

                for tensorlabel in val_labels_convert:
                    tensorlabel.detach().cpu()
                epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (1, 10.0))
        mean_dice_val = dice_metric.aggregate()
        dice_metric.reset()

    print(mean_dice_val)

    model1.eval()
    with torch.no_grad():
        for batch in epoch_iterator_val:
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            with torch.cuda.amp.autocast():
                val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model1)
                val_labels_list = decollate_batch(val_labels)
                val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
                val_outputs_list = decollate_batch(val_outputs)
                val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
                dice_metric(y_pred=val_output_convert, y=val_labels_convert)

                for tensoroutput in val_output_convert: 
                    tensoroutput.detach().cpu()

                for tensorlabel in val_labels_convert:
                    tensorlabel.detach().cpu()

            epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (1, 10.0))
        mean_dice_val = dice_metric.aggregate().item()
        dice_metric.reset()

    print(mean_dice_val)

    model2.eval()
    with torch.no_grad():
        for batch in epoch_iterator_val:
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            with torch.cuda.amp.autocast():
                val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model2)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)


            for tensoroutput in val_output_convert: 
                tensoroutput.detach().cpu()

            for tensorlabel in val_labels_convert:
                tensorlabel.detach().cpu()

            epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (1, 10.0))
        mean_dice_val = dice_metric.aggregate().item()
        dice_metric.reset()

    print(mean_dice_val)

    model3.eval()
    with torch.no_grad():
        for batch in epoch_iterator_val:
            val_inputs, val_labels = (batch["image"].cpu(), batch["label"].cpu())
            with torch.cuda.amp.autocast():
                val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model3)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)

            for tensoroutput in val_output_convert:
                    ensoroutput.detach().cpu()

            for tensorlabel in val_labels_convert:
                tensorlabel.detach().cpu()

            epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (1, 10.0))
        mean_dice_val = dice_metric.aggregate().item()
        dice_metric.reset()

    print(mean_dice_val)

validation(epoch_iterator_val)