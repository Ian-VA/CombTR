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
import csv
from torchvision.models.segmentation import deeplabv3_resnet50
from model import CombTR
import pandas as pd
from monai.utils.misc import set_determinism

### TRAINING FILE USED FOR ALL MODELS IN COMBTR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

set_determinism(seed=0)
loss_function = DiceCELoss(to_onehot_y=True, softmax=True) 
torch.backends.cudnn.benchmark = True

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


optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scaler = torch.cuda.amp.GradScaler()
set_determinism(seed=0)
root_dir = "./" # !! change if desired

def validation(epoch_iterator_val):
    model.eval() # !! change for model
    with torch.no_grad():
        for batch in epoch_iterator_val:
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            with torch.cuda.amp.autocast():
                val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model) # !! change for model
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
    model.train() # !! change for model
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)

    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["image"].cuda(), batch["label"].cuda())
        with torch.cuda.amp.autocast():
            logit_map = model(x) # !! change for model
            loss = loss_function(logit_map, y)
        scaler.scale(loss).backward()
        epoch_loss += loss.item()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        epoch_iterator.set_description(f"Training ({global_step} / {max_iterations} Steps) (loss={loss:2.5f})")
        if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
            epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
            dice_val = validation(epoch_iterator_val)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                torch.save(model1.state_dict(), os.path.join(root_dir, "best.pth")) # !! change if desired
                print(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best, dice_val)
                )
            else:
                print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
        global_step += 1
    return global_step, dice_val_best, global_step_best

if __name__ == '__main__':
    max_iterations = 10000
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

    while global_step < max_iterations:
        global_step, dice_val_best, global_step_best = train(global_step, train_loader, val_loader, dice_val_best, global_step_best)

    model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))

# UNETR: 0.793, SwinUNETR: 0.842, SegResNet: 0.80, CombTR: 0.853