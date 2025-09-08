import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import model.models as models

model = models.ViT2Channel
def get_first_layer(model):
    for layer in model.modules:
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            return layer
        else:
            return None
        
        break


print(get_first_layer(model))

