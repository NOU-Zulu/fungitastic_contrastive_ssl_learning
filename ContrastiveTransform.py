import torchvision.transforms as transforms
from torchvision.transforms import ColorJitter, RandomGrayscale, GaussianBlur
import random

class ContrastiveTransform:
    def __init__(self, size=128):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
            transforms.ColorJitter(0.6, 0.6, 0.4, 0.4),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])

    def __call__(self, x):
        xi = self.transform(x)
        xj = self.transform(x)
        return xi, xj
