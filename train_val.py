from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRangePercentilesd, ResizeWithPadOrCropd, Resized,
    RandFlipd, RandRotate90d, RandGaussianNoised, EnsureTyped,
)
from sklearn.model_selection import train_test_split
import pandas as pd
df_data = pd.read_csv("./data/labels_duke.csv")
data = df_data.to_dict(orient="records")


train_data, val_data = train_test_split(
    data, test_size=0.2, stratify=[d["label"] for d in data], random_state=42
)

train_transforms = Compose([
    LoadImaged(keys="image", reader="PydicomReader"),          # reads the DICOM series
    EnsureChannelFirstd(keys="image"),                      # -> (C, H, W, D)
    Orientationd(keys="image", axcodes="RAS"),             # standardize orientation
    Spacingd(keys="image", pixdim=(1.0, 1.0, 1.0),         # resample to isotropic
             mode="bilinear"),
    ScaleIntensityRangePercentilesd(                       # MRI-appropriate normalization
        keys="image", lower=1, upper=99,
        b_min=0.0, b_max=1.0, clip=True),
    Resized(keys="image", spatial_size=(96, 96, 96)),      # fixed input size
    RandFlipd(keys="image", prob=0.5, spatial_axis=0),     # augmentation
    RandRotate90d(keys="image", prob=0.5, spatial_axes=(0, 1)),
    RandGaussianNoised(keys="image", prob=0.2),
    EnsureTyped(keys="image"),
])

val_transforms = Compose([                                  # no augmentation
    LoadImaged(keys="image", reader="PydicomReader"),
    EnsureChannelFirstd(keys="image"),
    Orientationd(keys="image", axcodes="RAS"),
    Spacingd(keys="image", pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
    ScaleIntensityRangePercentilesd(keys="image", lower=1, upper=99,
                                    b_min=0.0, b_max=1.0, clip=True),
    Resized(keys="image", spatial_size=(96, 96, 96)),
    EnsureTyped(keys="image"),
])

from monai.data import Dataset, DataLoader
# CacheDataset is much faster if your data fits in RAM:
# from monai.data import CacheDataset

train_ds = Dataset(data=train_data, transform=train_transforms)
val_ds   = Dataset(data=val_data,   transform=val_transforms)

train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=2, num_workers=0)


import torch
from monai.networks.nets import ViT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ViT(
    in_channels=1,
    img_size=(96, 96, 96),
    patch_size=(16, 16, 16),
    classification=True,
    num_classes=5,
    post_activation=None,      # return raw logits
    save_attn=True
).to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        x = batch["image"].to(device)
        y = batch["label"].to(device)
        optimizer.zero_grad()
        logits = model(x)
        if isinstance(logits, tuple):   # ViT returns (logits, hidden_states)
            logits = logits[0]
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

    # validation
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in val_loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            correct += (logits.argmax(1) == y).sum().item()
            total += y.numel()
    print(f"epoch {epoch}: val acc = {correct/total:.3f}")