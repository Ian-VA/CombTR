from monai.inferers import sliding_window_inference
from monai.networks.nets import UNETR, SegResNet, UNet, SwinUNETR
import torch
from getdataloader import getdataloaders, get_valds, get_noprocess
import os
from monai.data import decollate_batch
from os import path
import torch
import matplotlib.pyplot as plt
import json
import numpy as np
import nibabel as nib
import scipy.ndimage as ndimage


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

model.load_state_dict(torch.load(os.path.join("./", "realunetrmodel.pth"), strict=False))
model1.load_state_dict(torch.load(os.path.join("./", "best_swinUNETR.pth"), strict=False))
model2.load_state_dict(torch.load(os.path.join("./", "bestSEGRESNET.pth"), strict=False))

def make_stackingdata(jsonfilename="./stackingdata.json"):
    trainloader, _= getdataloaders()
    model.eval()
    model1.eval()
    model2.eval()
    i = 0
    with torch.no_grad():
        for step, batch in enumerate(trainloader):
            if path.isfile(jsonfilename) is False:
                raise Exception("json datafile not found")


            original_affine = batch["label_meta_dict"]["affine"][0].numpy()
            img_name = os.path.split(batch["label"].meta["filename_or_obj"][0])[1]
            print(img_name)

            val_inputs, val_labels = (batch["image"], batch["label"])
            val_outputs3 = sliding_window_inference(val_inputs, (96, 96, 96), 1, model)
            val_outputs1 = sliding_window_inference(val_inputs, (96, 96, 96), 1, model1)
            val_outputs2 = sliding_window_inference(val_inputs, (96, 96, 96), 1, model2)

            valalloutputs = [val_outputs1, val_outputs2, val_outputs3]
            val_outputs = torch.cat(tuple(valalloutputs), dim=0)
            print(val_outputs1.shape)
            print(val_outputs.shape)
            val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
            val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)[0]
            val_labels = val_labels.cpu().numpy()[0, 0, :, :, :]


            nib.save(
                nib.Nifti1Image(val_outputs.astype(np.uint8), original_affine),
                "./imagesTr/image_" + str(i) + ".nii.gz"
            )

            #"C:/Users/mined/Downloads/stackingdata/imagesTr/image_" + str(i) + ".nii.gz"

            nib.save(
                nib.Nifti1Image(val_labels, original_affine),
                "./labelsTr/label_" + str(i) + ".nii.gz"
            )

            jsonfile = []


            with open(jsonfilename, "r+", encoding='utf-8') as js:
                dicttoappend = {
                    "image" : "image_" + str(i) + ".nii.gz",
                    "label" : "label_" + str(i) + ".nii.gz"
                }

                jsonfile = json.load(js)
            
            jsonfile['training'].append(dicttoappend)

            with open(jsonfilename, 'w') as js:
                json.dump(jsonfile, js, indent=4, separators=(',', ': '))

            i += 1
            




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
    make_stackingdata()