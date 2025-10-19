import torch
import torch.nn as nn
from torchvision import models

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
PATH = "checkpoints/pretrained/resnet18.pth"
torch.save(model.state_dict(), PATH)
