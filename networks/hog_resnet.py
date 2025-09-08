
import torch
import torch.nn as nn
from torchvision import models


class CustomFeedForward(nn.Module):
    def __init__(self, input_dim):
        super(CustomFeedForward, self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2), 
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.ff(x)

class Resnet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        old_conv = resnet.conv1
        resnet.conv1 = nn.Conv2d(
            in_channels=2,  # Change from 3 to 2
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias
        )

        # === OPTIONAL: COPY PRETRAINED WEIGHTS TO NEW CONV ===
        with torch.no_grad():
            resnet.conv1.weight[:, :2] = old_conv.weight[:, :2]
        # Freeze conv layers (optional)

        # === FREEZE ALL PARAMETERS FIRST ===
        for param in resnet.parameters():
            param.requires_grad = False

        # === UNFREEZE ONLY FIRST CONV AND FC ===
        for param in resnet.conv1.parameters():
            param.requires_grad = True

        for param in resnet.fc.parameters():
            param.requires_grad = True

        # Replace fc layer
        num_ftrs = resnet.fc.in_features
        resnet.fc = CustomFeedForward(input_dim=num_ftrs)
        self.model = resnet

    def forward(self, x):
        return self.model(x)
