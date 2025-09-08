import torch
import math
import timm
import torch.nn as nn
import numpy as np
from einops import rearrange




class SimpleCNN(nn.Module):
    def __init__(self, in_channels=1, img_size=256):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),  # 224 -> 224
            nn.ReLU(),
            nn.MaxPool2d(2),  # 224 -> 112

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 112 -> 112
            nn.ReLU(),
            nn.MaxPool2d(2),  # 112 -> 56

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 56 -> 56
            nn.ReLU(),
            nn.MaxPool2d(2),  # 56 -> 28
        )
        self.flatten_dim = 128 * (img_size//8) * (img_size//8)  # compute final flattened size

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)  # flatten


class DualCNN(nn.Module):
    def __init__(self, num_classes=1, mlp_dim=512):
        super(DualCNN, self).__init__()
        # Two CNN branches
        self.branch1 = SimpleCNN(in_channels=1)
        self.branch2 = SimpleCNN(in_channels=1)

        # Combined fully connected layers
        combined_dim = self.branch1.flatten_dim + self.branch2.flatten_dim
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_dim, mlp_dim//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_dim//2, num_classes),
        )

    def forward(self, x):
        f1 = self.branch1(x[:, 0:1, :, :])
        f2 = self.branch2(x[:, 1:2, :, :])
        combined = torch.cat([f1, f2], dim=1)  # concatenate along feature dim
        out = self.classifier(combined)
        return out