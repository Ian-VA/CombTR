import os
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import UNETR
import torch
from monai.data import decollate_batch
from tqdm import tqdm
from monai.transforms import AsDiscrete
from datautils.getdata import getdataloaders
from monai.utils.misc import set_determinism
import gc
from torch.cuda.amp import autocast
from combtr import get_model

### TRAINING FILE USED FOR ALL MODELS IN COMBTR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

set_determinism(seed=0)
loss_function = DiceCELoss(to_onehot_y=True, softmax=True) 
torch.backends.cudnn.benchmark = True
image_size=(96, 96, 96)

model = UNETR(
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

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scaler = torch.cuda.amp.GradScaler()
set_determinism(seed=0)
root_dir = "./" # !! change if desired
filename = "best.pth"

def validation(epoch_iterator_val): # validation for dice
    model.eval() # !! change for model
    with torch.no_grad(), autocast():
        for batch in epoch_iterator_val:
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            val_outputs = sliding_window_inference(val_inputs, image_size, 4, model) # !! change for model
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list] # must have these transforms applied
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (global_step, 10.0))
        mean_dice_val = dice_metric.aggregate().item()
        dice_metric.reset()

    return mean_dice_val

def train(max_iterations):
    model.train() # !! change for model
    epoch_loss = 0
    step = 0
    global_step = 0
    epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)

    while global_step < max_iterations:
        for step, batch in enumerate(epoch_iterator):
            step += 1
            x, y = (batch["image"].cuda(), batch["label"].cuda())
            with autocast():
                logit_map = model(x) # !! change for model
                loss = loss_function(logit_map, y)

                scaler.scale(loss).backward() # scaler for memory improvements
                epoch_loss += loss.item()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
                    epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
                    dice_val = validation(epoch_iterator_val) # calculate dice score
                    epoch_loss /= step

                    epoch_loss_values.append(epoch_loss)
                    metric_values.append(dice_val)

                    if dice_val > dice_val_best:
                        dice_val_best = dice_val
                        torch.save(model.state_dict(), os.path.join(root_dir, filename)) # !! change if desired
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
            torch.cuda.empty_cache() # clean up cache for memory
            gc.collect()
    return model

if __name__ == '__main__':
    max_iterations = 10000 
    eval_num = 500 # evaluation is ran a total of max_iterations / eval_num times
    post_label = AsDiscrete(to_onehot=14) # transforms for validation and prediction 
    post_pred = AsDiscrete(argmax=True, to_onehot=14)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False) # reduction can be "mean_channel" for channel-wise results
    global_step = 0
    dice_val_best = 0.0
    global_step_best = 0
    epoch_loss_values = []
    metric_values = []

    train_loader, val_loader = getdataloaders()
