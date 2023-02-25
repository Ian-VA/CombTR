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
from datautils.getdataloader import getdataloaders
import csv
from metalearner import CombTRMetaLearner
from torchvision.models.segmentation import deeplabv3_resnet50


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model1 = SwinUNETR(
    img_size=(96, 96, 96),
    in_channels=1,
    out_channels=14,
    feature_size=48,
    use_checkpoint=True,
).to(device)

model1.load_state_dict(torch.load(os.path.join("C:/Users/mined/Downloads/", "best_swinUNETR.pth"), map_location=torch.device('cpu')), strict=False)


loss_function = DiceCELoss(to_onehot_y=True, softmax=True) 
torch.backends.cudnn.benchmark = True
optimizer = torch.optim.AdamW(model1.parameters(), lr=1e-4, weight_decay=1e-5)



def validation(epoch_iterator_val):
    model1.eval()
    with torch.no_grad():
        for batch in epoch_iterator_val:
            val_inputs, val_labels = batch["image"].cuda(), batch["label"].cuda()
            val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 1, model1)
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
    model1.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(train_loader, desc="Training (loss=X.X)", dynamic_ncols=True)
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["image"].cuda(), batch["label"].cuda())
        logit_map = model1(x)
        loss = loss_function(logit_map, y)
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
                torch.save(model1.state_dict(), os.path.join(datadir, "best_swinUNETR.pth"))
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


if __name__ == '__main__':
    datadir = "C:/Users/mined/Desktop/projects/segmentationv2/" # replace with your dataset directory
    max_iterations = 25000
    eval_num = 1
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

    with open("validationdice.csv", 'w') as f:
        write = csv.writer(f)
        write.writerow(["DICE"])
        write.writerows(metric_values)

    with open("loss.csv", 'w') as f:
        write = csv.writer(f)
        write.writerow(["LOSS"])
        write.writerows(epoch_loss_values)


