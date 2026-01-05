import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score

# ---------------------------------------------------
# Imports from your project
# ---------------------------------------------------
from makeTorchdataset import MushroomDataset
from simclr_model import SimCLR   # encoder definition


# ---------------------------------------------------
# Classification model (linear evaluation)
# ---------------------------------------------------
class SimCLRClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # load the SimCLR encoder WITHOUT projection head
        base = SimCLR()
        self.encoder = base.encoder   # CNN only

        # freeze encoder
        for p in self.encoder.parameters():
            p.requires_grad = False

        # linear head
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        h = self.encoder(x)
        h = torch.flatten(h, 1)
        return self.fc(h)


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
if __name__ == "__main__":

    device = "cuda"
    print("Running on:", device)

    img_folder = "top5_species_files/"

    # Classification transform (no SimCLR)
    classification_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    # Load labeled train and test sets
    trainset = MushroomDataset(
        csv_path="train.csv",
        img_folder=img_folder,
        transform=classification_transform
    )

    testset = MushroomDataset(
        csv_path="test.csv",
        img_folder=img_folder,
        transform=classification_transform
    )

    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader  = DataLoader(testset,  batch_size=32, shuffle=False)

    # Load pretrained encoder
    model = SimCLRClassifier(num_classes=len(trainset.classes))
    model.encoder.load_state_dict(torch.load("simclr_encoder.pth", map_location=device))
    model.to(device)

    # Loss and optimizer (only train the head)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

    # ---------------------------------------------------
    # Train classification head
    # ---------------------------------------------------
    epochs = 10
    for epoch in range(epochs):
        model.train()
        total = 0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += loss.item()

        print(f"Epoch {epoch+1}: loss = {total/len(trainloader):.4f}")

    # ---------------------------------------------------
    # Evaluate accuracy
    # ---------------------------------------------------
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            pred = outputs.argmax(dim=1)

            preds.extend(pred.cpu().numpy())
            trues.extend(labels.cpu().numpy())

    acc = accuracy_score(trues, preds)
    print("--------------------------------------------------")
    print(f"Test Accuracy: {acc:.4f}")
    print("--------------------------------------------------")
