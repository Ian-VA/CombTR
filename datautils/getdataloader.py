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
    RandShiftIntensityd
)

from monai.data import Dataset, DataLoader, load_decathlon_datalist

def getdataloaders(amin=-200, amax=200, bmin=0.0, bmax=1.0):
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=4,
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


    datadir = "C:/Users/mined/Downloads/data/"
    json = "dataset_0.json"
    datasets = datadir + json
    datalist = load_decathlon_datalist(datasets, True, "training")
    val_files = load_decathlon_datalist(datasets, True, "validation")

    train_ds = Dataset(data = datalist, transform = train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)

    train_loader = DataLoader(
        train_ds, 
        batch_size=1, 
        shuffle=True,
        num_workers=4, 
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds, 
        batch_size=1, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )

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

    datadir = "C:/Users/mined/Downloads/data/"
    json = "dataset_0.json"
    datasets = datadir + json
    val_files = load_decathlon_datalist(datasets, True, "validation")
    val_ds = Dataset(data=val_files, transform=val_transforms)

    val_loader = DataLoader(
        val_ds, 
        batch_size=1, 
        shuffle=False, 
        num_workers=4, 
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

    datadir = "C:/Users/mined/Downloads/data/"
    json = "dataset_0.json"
    datasets = datadir + json
    val_files = load_decathlon_datalist(datasets, True, "validation")
    val_ds = Dataset(data=val_files, transform=val_transforms)
    
    return val_ds

def get_noprocess():
    transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),

            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),

        ]
    )

    datadir = "C:/Users/mined/Downloads/data/"
    json = "dataset_0.json"
    datasets = datadir + json
    files = load_decathlon_datalist(datasets, True, "validation")
    ds = Dataset(data=files, transform=transforms)
    return ds
