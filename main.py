from torch.utils.data import DataLoader
from makeTorchdataset import MushroomDataset
from ContrastiveTransform import ContrastiveTransform
from simclr_train import train_simclr
from torchvision import transforms
import torch
from classifier import Classifier

if __name__ == "__main__":
   # model = Classifier(num_classes=5)
   # model.encoder.load_state_dict(torch.load("simclr_encoder.pth"))


    img_folder = "top5_species_files/"

    # ---------- Contrastive pretraining ----------
    contrastive_transform = ContrastiveTransform(size=128)

    trainset = MushroomDataset(
        csv_path="train.csv",
        img_folder=img_folder,
        transform=contrastive_transform
    )

    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, drop_last=False)

    # Train SimCLR + save encoder
    train_simclr(trainloader, epochs=20, lr=3e-4, device="cuda")

    print("Pretraining complete.")
