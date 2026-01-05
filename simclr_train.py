import torch
from torch.optim import Adam
from tqdm import tqdm

from nt_xent_loss import nt_xent_loss
from simclr_model import SimCLR

def train_simclr(trainloader, epochs=20, lr=3e-4, device="cuda"):
    model = SimCLR().to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for v1, v2 in tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}"):
            v1, v2 = v1.to(device), v2.to(device)

            _, z1 = model(v1)
            _, z2 = model(v2)

            loss = nt_xent_loss(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {total_loss/len(trainloader):.4f}")

    torch.save(model.encoder.state_dict(), "simclr_encoder.pth")
    print("Encoder saved as simclr_encoder.pth")

    return model
