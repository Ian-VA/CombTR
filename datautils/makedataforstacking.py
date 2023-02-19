from monai.inferers import sliding_window_inference
from monai.networks.nets import UNETR
import torch
from getdataloader import get_valloader, get_valds
import os
from os import path
import torch
import matplotlib.pyplot as plt
import json
import numpy as np
import nibabel as nib


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
)


def make_stackingdata(jsonfilename="C:/Users/mined/Desktop/projects/segmentationv2/stackingdata.json"):
    valloader = get_valloader()
   # model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_unetr.pth")))
    model.eval()
    i = 0
    with torch.no_grad():
        for step, batch in enumerate(valloader):
            if path.isfile(jsonfilename) is False:
                raise Exception("json datafile not found")

            val_inputs, val_labels = (batch["image"], batch["label"])
            val_outputs = sliding_window_inference(val_inputs, (128, 128, 128), 4, model)
            np_arr = val_outputs.cpu().detach().numpy()
            np_labels = val_labels.cpu().detach().numpy()

            jsonfile = []
            nifti = nib.Nifti1Image(np_arr, affine=np.eye(4))
            niftilabel = nib.Nifti1Image(np_labels, affine=np.eye(4))

            print(nifti.get_shape())
            nib.save(nifti, "output_" + str(i) + ".nii.gz")
            nib.save(niftilabel, "label_" + str(i) + ".nii.gz")


            with open(jsonfilename, "r+", encoding='utf-8') as js:
                dicttoappend = {
                    "image" : "output_" + str(i) + ".nii.gz",
                    "label" : "label_" + str(i) + ".nii.gz"
                }

                jsonfile = json.load(js)
            
            jsonfile['training'].append(dicttoappend)

            with open(jsonfilename, 'w') as js:
                json.dump(jsonfile, js, indent=4, separators=(',', ': '))

            i += 1
            




def plotdata(root_dir):
    val_ds = get_valds()
    slice_map = {
        "img0035.nii.gz": 170,
        "img0036.nii.gz": 230,
        "img0037.nii.gz": 204,
        "img0038.nii.gz": 204,
        "img0039.nii.gz": 204,
        "img0040.nii.gz": 180,
    }
    case_num = 2
    img_name = os.path.split(val_ds[case_num]["image"].meta["filename_or_obj"])[1]
    img = val_ds[case_num]["image"]
    label = val_ds[case_num]["label"]
    val_inputs = torch.unsqueeze(img, 1)
    val_labels = torch.unsqueeze(label, 1)
    val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 1, model, overlap=0.8)
    plt.figure("image", (18, 6))
    plt.subplot(1, 2, 1)
    plt.title("image")
    plt.imshow(img[0, :, :, slice_map[img_name]].detach().cpu(), cmap="gray")
    print("yes")
    plt.subplot(1, 2, 2)
    plt.title("label")
    plt.imshow(label[0, :, :, slice_map[img_name]].detach().cpu())
    plt.subplot(1, 3, 3)
    plt.title("output")
    plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, slice_map[img_name]])
    plt.show()




if __name__ == "__main__":
    plotdata("C:/Users/mined/Downloads/")