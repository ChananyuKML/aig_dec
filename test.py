import torch.optim as optim
import torch
import argparse
from utils import train_opt, create_dataloader, custom_transforms, create_dataset
from networks.hog_vit import *
from networks.hog_resnet import *
from networks.hog_cnn import *
import os
import matplotlib.pyplot as plt
import pathlib
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image

device = "cuda"

transform = custom_transforms.hogfft_256

train_loader, val_loader, test_loader= create_dataloader.createByDir("previous studies/CNNDetection/dataset", batch_size=32, transform=transform)








model = DualCNN()

model.load_state_dict(torch.load("checkpoints/dual_cnn_hogfft_256_norm/ckpt_30.pth", map_location="cpu"))

model.eval()

correct=0
test_loss=0
model.to(device)
out_path = 'output/test'
path = pathlib.Path(out_path)
train_loss_hist = []
val_loss_hist = []
criterion = nn.BCEWithLogitsLoss()


for data, labels in test_loader:
        data, labels = data.to(device), labels.to(device).float().unsqueeze(1)
        outputs = model(data)
        loss = criterion(outputs, labels)
        pred = (outputs > 0.5).float()
        test_loss += loss.item() * data.size(0)
        correct += (pred  == labels).sum().item()

acc = correct / len(test_loader.dataset)
test_loss /= len(test_loader.dataset)
print(f"Test Loss: {test_loss:.4f}, Accuracy: {acc:.4f}")    