import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import torch
from ContrastiveTransform import ContrastiveTransform


class MushroomDataset(Dataset):
    def __init__(self, csv_path=None, img_folder=None, transform=None, preload_df=None, classes=None):
        if preload_df is not None:
            # When loading from .pt
            self.df = preload_df
            self.classes = classes
        else:
            # When creating dataset for the first time
            self.df = pd.read_csv(csv_path)
            self.classes = sorted(self.df["species"].unique())

        self.img_folder = img_folder
        self.transform = transform
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.img_folder, row["filename"])
        image = Image.open(img_path).convert("RGB")

        label = self.class_to_idx[row["species"]]

        if isinstance(self.transform, ContrastiveTransform):
            v1, v2 = self.transform(image)
            return v1, v2

            # Otherwise → classification transform
        if self.transform:
            image = self.transform(image)

        return image, label
# ---------- Dataset defined above ----------

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

full_csv = "top5_species_data.csv"
img_folder = "top5_species_files/"

df = pd.read_csv(full_csv)

train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["species"])

train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

trainset = MushroomDataset("train.csv", img_folder, transform=transform)
testset = MushroomDataset("test.csv", img_folder, transform=transform)

torch.save({
    "df": train_df,
    "classes": trainset.classes,
    "img_folder": img_folder
}, "trainset.pt")

torch.save({
    "df": test_df,
    "classes": testset.classes,
    "img_folder": img_folder
}, "testset.pt")

print("Saved dataset!")
