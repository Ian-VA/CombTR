from monai.transforms import (
    Compose,
    CropForegroundd,
    LoadImaged,
    ScaleIntensityRanged,
    Spacingd,
    Resized,
    ToTensord,
    AddChanneld,
    EnsureChannelFirstd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    EnsureTyped
)

import torch

from monai.utils.misc import set_determinism
from monai.apps import CrossValidation
from monai.data import Dataset, DataLoader, load_decathlon_datalist, CacheDataset, ThreadDataLoader, set_track_meta
set_determinism(seed=0)

datadir = "/home/ian/Desktop/research/data/"
json = "dataset_0.json"

def getdataloaders(amin=-200, amax=200, bmin=0.0, bmax=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], ensure_channel_first=True, image_only=False),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            EnsureTyped(keys=["image", "label"], device=device),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=1,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[0],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[1],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[2],
                prob=0.10,
            ),
            RandRotate90d(
                keys=["image", "label"],
                prob=0.10,
                max_k=3,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.50,
            ),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            )
        ]
    )

    datasets = datadir + json
    datalist = load_decathlon_datalist(datasets, True, "training")
    val_files = load_decathlon_datalist(datasets, True, "validation")


    train_ds = CacheDataset(data = datalist, transform = train_transforms, cache_num=1, cache_rate=1.0, num_workers=1) # increase as you wish depending on your hardware
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_num=1, cache_rate=1.0, num_workers=1)
    train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=4, shuffle=True)
    val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)

    return train_loader, val_loader

def get_valloader(amin=-200, amax=200, bmin=0.0, bmax=1.0):
    val_transforms = Compose(
        [
            LoadImaged(keys=['image', 'label']),
            AddChanneld(keys=['image', 'label']),

            Spacingd(
                keys=['image', 'label'],
                pixdim=(1.5, 1.5, 2.0),
            ),

            ScaleIntensityRanged(
                keys=['image'], 
                a_min=amin, 
                a_max=amax, 
                b_min=bmin, 
                b_max=bmax, 
                clip=True
            ),

            CropForegroundd(keys=['image', 'label'], source_key='image'),
            Resized(keys=['image', 'label'], spatial_size=[96, 96, 96]),
            ToTensord(keys=['image', 'label'])
        ]
    )

    datasets = datadir + json
    val_files = load_decathlon_datalist(datasets, True, "validation")
    val_ds = Dataset(data=val_files, transform=val_transforms)

    val_loader = DataLoader(
        val_ds, 
        batch_size=1, 
        shuffle=False, 
        num_workers=1, 
        pin_memory=True
    )

    return val_loader


def get_valds():
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=["image", "label"], source_key="image"),
        ]
    )

    datasets = datadir + json
    val_files = load_decathlon_datalist(datasets, True, "validation")
    val_ds = Dataset(data=val_files, transform=val_transforms)
    
    return val_ds

def get_noprocess():
    transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
        ]
    )

    datasets = datadir + json
    files = load_decathlon_datalist(datasets, True, "validation")
    ds = Dataset(data=files, transform=transforms)
    return ds
