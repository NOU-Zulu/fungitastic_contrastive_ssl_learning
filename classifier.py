import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.encoder = SimCLR().encoder    # load encoder
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        h = self.encoder(x)
        h = torch.flatten(h, 1)
        out = self.fc(h)
        return out
