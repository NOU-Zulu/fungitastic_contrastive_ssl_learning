import torch
import torch.nn as nn
import torchvision.models as models

class SimCLR(nn.Module):
    def __init__(self, base_model='resnet18', projection_dim=128):
        super().__init__()

        resnet = getattr(models, base_model)(weights=None)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  # remove FC

        # Projection head (MLP)
        self.projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        h = torch.flatten(h, 1)
        z = self.projector(h)
        return h, z
