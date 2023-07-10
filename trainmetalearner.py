from monai.inferers import sliding_window_inference
from monai.networks.nets import UNETR, SegResNet, SwinUNETR
import torch
from datautils.getdata import getdataloaders
import os
from monai.data import decollate_batch
import torch
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from torch.cuda.amp import autocast
from monai.metrics import DiceMetric
from tqdm import tqdm
from monai.utils.misc import set_determinism
from monai.transforms import (
    Compose,
    Activations,
    AsDiscrete,
    SaveImage
)

### FILE USED FOR TRAINING META LEARNERS AKA LEVEL-1 MODELS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") ## if you're not running cuda, you'll also have to change where sliding_window_inference is deployed and where inputs and labels are allocated (remove the .cuda())
set_determinism(seed=0)
image_size = (96, 96, 96)

metalearner = SwinUNETR(
    img_size=image_size,
    in_channels=14,
    out_channels=14,
    feature_size=48,
    use_checkpoint=True,
).to(device)

segresnet = SegResNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=14,
    init_filters=16
).to(device)

swinunetr = SwinUNETR(
    img_size=image_size,
    in_channels=1,
    out_channels=14,
    feature_size=48,
    use_checkpoint=True,
).to(device)

unetr = UNETR(
    in_channels=1,
    out_channels=14,
    img_size=image_size,
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
torch.backends.cudnn.benchmark = True

optimizer = torch.optim.AdamW(metalearner.parameters(), lr=1e-4, weight_decay=1e-5) # can experiment with lr and weight decay if desired, not the focus of the paper
scaler = torch.cuda.amp.GradScaler()

swinunetr.load_state_dict(torch.load(os.path.join("./", "bestswinUNETR.pth")))
unetr.load_state_dict(torch.load(os.path.join("./", "bestUNETR.pth")), strict=False) # strict has to be false for unetr's ViT
segresnet.load_state_dict(torch.load(os.path.join("./", "bestSEGRESNET.pth")))
model_list = [swinunetr, segresnet, unetr]

def validation(epoch_iterator_val):
    metalearner.eval()
    [i.eval() for i in model_list]

    with (torch.no_grad(), autocast()):
        for batch in epoch_iterator_val:
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            val_outputs = [sliding_window_inference(val_inputs, image_size, 1, i, device="cuda") for i in model_list]

            val_outputs = torch.mean(torch.stack((val_outputs), 1), dim=1)
        
            val_outputs = sliding_window_inference(val_outputs, image_size, 1, metalearner, device="cuda")
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
                            
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
        epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (global_step, 10.0))
    mean_dice_val = dice_metric.aggregate().item()
    dice_metric.reset()
        
    return mean_dice_val


def train(global_step, train_loader, val_loader, dice_val_best, global_step_best, loss_best):
    epoch_loss = 0
    step = 0

    epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
    epoch_iterator_val = tqdm(val_loader, desc="Validation (dice=X.X)", dynamic_ncols=True)

    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["image"].cuda(), batch["label"].cuda())
        
        with autocast(): # may omit if enough RAM

            ### run inputs through respective models

            val_outputs = [i(x) for i in model_list]
            
            valalloutputs = torch.stack((val_outputs), 1) # stack along n-thh dimension
            val_outputs = torch.mean(valalloutputs, dim=1) # take mean along n-th dimension

            val_outputs = metalearner(val_outputs) # input to the metalearner model
            loss = loss_function(val_outputs, y)
            
            scaler.scale(loss).backward()
            epoch_loss += loss.item()
            scaler.step(optimizer)
            scaler.update()

        epoch_iterator.set_description("Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss))
        if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
            epoch_iterator_val = tqdm(val_loader, desc="Validation (dice=X.X)", dynamic_ncols=True)
            dice_val = validation(epoch_iterator_val)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss) # for later graphing 
            metric_values.append(dice_val)
            if dice_val > dice_val_best: 
                dice_val_best = dice_val
                global_step_best = global_step
                torch.save(metalearner.state_dict(), os.path.join(datadir, "bestCombTRMetaLearner.pth"))
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
    return global_step, dice_val_best, global_step_best, loss_best

if __name__ == "__main__":
    datadir = "./" # replace with your dataset directory
    max_iterations = 10000
    eval_num = 500
    post_label = AsDiscrete(to_onehot=14) 
    post_pred = AsDiscrete(argmax=True, to_onehot=14)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False) # reduction is mean for overall dice score, can look at channel-wise dice if you wish with "mean_channel" reduction
    global_step = 0
    dice_val_best = 0.0
    global_step_best = 0
    loss_best = 99.9
    epoch_loss_values = []
    metric_values = []
    train_loader, val_loader = getdataloaders()
    
    while (global_step < max_iterations):
        global_step, dice_val_best, global_step_best, loss_best = train(global_step, train_loader, val_loader, dice_val_best, global_step_best, loss_best)
