from monai.transforms import (
    Compose,
    CropForegroundd,
    LoadImaged,
    ScaleIntensityRanged,
    Spacingd,
    Resized,
    ToTensord,
    AddChanneld
)

from monai.data import Dataset, DataLoader, load_decathlon_datalist
from monai.apps import DecathlonDataset

def getdataloaders(amin=-200, amax=200, bmin=0.0, bmax=1.0):
    train_transforms = Compose(
        [
            LoadImaged(keys=['image', 'label']),
            AddChanneld(keys=['image', 'label']),

            Spacingd(
                keys=['image', 'label'], 
                pixdim=(1.5, 1.5, 2)
            ),

            ScaleIntensityRanged(
                keys=['image', 'label'], 
                a_min=amin, 
                a_max=amax, 
                b_min=bmin, 
                b_max=bmax, 
                clip=True
            ),

            CropForegroundd(keys=['image', 'label'], source_key='image'),
            Resized(keys=['image', 'label'], spatial_size=[128,128,128]),
            ToTensord(keys=['image', 'label'])
        ]
    )

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
            Resized(keys=['image', 'label'], spatial_size=[128,128,128]),
            ToTensord(keys=['image', 'label'])
        ]
    )

    datadir = "C:/Users/mined/Downloads/Task03_Liver/"
    json = "dataset.json"
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
            Resized(keys=['image', 'label'], spatial_size=[128,128,128]),
            ToTensord(keys=['image', 'label'])
        ]
    )

    datadir = "C:/Users/mined/Downloads/Task03_Liver/"
    json = "dataset.json"
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


def get_valds(amin=-200, amax=200, bmin=0.0, bmax=1.0):
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
            Resized(keys=['image', 'label'], spatial_size=[128,128,128]),
            ToTensord(keys=['image', 'label'])
        ]
    )

    datadir = "C:/Users/mined/Downloads/Task03_Liver/"
    json = "dataset.json"
    datasets = datadir + json
    val_files = load_decathlon_datalist(datasets, True, "validation")
    val_ds = Dataset(data=val_files, transform=val_transforms)
    
    return val_ds

def getdataloaderswdownload(amin=-200, amax=200, bmin=0.0, bmax=1.0):
    train_transforms = Compose(
        [
            LoadImaged(keys=['image', 'label']),
            AddChanneld(keys=['image', 'label']),

            Spacingd(
                keys=['image', 'label'], 
                pixdim=(1.5, 1.5, 2)
            ),

            ScaleIntensityRanged(
                keys=['image', 'label'], 
                a_min=amin, 
                a_max=amax, 
                b_min=bmin, 
                b_max=bmax, 
                clip=True
            ),

            CropForegroundd(keys=['image', 'label'], source_key='image'),
            Resized(keys=['image', 'label'], spatial_size=[128,128,128]),
            ToTensord(keys=['image', 'label'])
        ]
    )

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
            Resized(keys=['image', 'label'], spatial_size=[128,128,128]),
            ToTensord(keys=['image', 'label'])
        ]
    )

    traindata = DecathlonDataset(
        root_dir="./", task="Task03_Liver", section="training", transform=train_transforms, download=True, num_workers=4
    )

    valdata = DecathlonDataset(
        root_dir="./", task="Task03_Liver", section="validation", transform=train_transforms, download=True, num_workers=4
    )

    val_loader = DataLoader(
        valdata, 
        batch_size=1, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )

    train_loader = DataLoader(
        traindata, 
        batch_size=1, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )

    return train_loader, val_loader


