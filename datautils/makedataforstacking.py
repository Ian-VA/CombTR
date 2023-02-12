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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNETR(
    in_channels=1,
    out_channels=3,
    img_size=(128, 128, 128),
    feature_size=16,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    pos_embed="perceptron",
    norm_name="instance",
    res_block=True,
    dropout_rate=0.0,
).to(device)


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
    valdataset = get_valds()
    case_num = 4
    model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
    model.eval()
    with torch.no_grad():
        img_name = os.path.split(valdataset[case_num]["image"].meta["filename_or_obj"])[1]
        img = valdataset[case_num]["image"]
        label = valdataset[case_num]["label"]
        val_inputs = torch.unsqueeze(img, 1).cuda()
        val_labels = torch.unsqueeze(label, 1).cuda()
        val_outputs = sliding_window_inference(val_inputs, (128, 128, 128), 4, model, overlap=0.8)
        plt.figure("check", (18, 6))
        plt.subplot(1, 3, 1)
        plt.title("image")
        plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, 4], cmap="gray")
        plt.subplot(1, 3, 2)
        plt.title("label")
        plt.imshow(val_labels.cpu().numpy()[0, 0, :, :, 4])
        plt.subplot(1, 3, 3)
        plt.title("output")
        plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, 4])
        plt.show()


if __name__ == "__main__":
    make_stackingdata()