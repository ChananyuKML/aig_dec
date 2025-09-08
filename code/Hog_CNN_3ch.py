import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

transform = transforms.Compose([
    transforms.Resize((256, 256)),   # Resize images
    transforms.ToTensor(),           # Convert to tensor
    transforms.Normalize([0.5], [0.5])  # Normalize (adjust for RGB if needed)
])

def train_model(model, train_loader, valid_loader, batch_size=32, num_epochs=10, lr=1e-4, device='cpu'):
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        best_val_loss = float('inf')
        correct = 0

        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            pred = (outputs > 0.5).float()
            train_loss += loss.item() * data.size(0)
            correct += (pred  == labels).sum().item()

        acc = correct / len(train_loader.dataset)
        train_loss /= len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Accuracy: {acc:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * data.size(0)
                pred = (outputs > 0.5).float()
                val_correct += (pred == labels).sum().item()

        val_acc = val_correct / len(valid_loader.dataset)
        val_loss /= len(valid_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
    
        if val_loss < best_val_loss:
            best_val_loss = val_loss + train_loss
            torch.save(model.state_dict(), f'../checkpoints/HoG_CNN_3ch/ckpt_{epoch+1}.pth')
            print(f"✅ Saved new best model at epoch {epoch+1} with val loss {val_loss:.4f}")

class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.feedforward = nn.Sequential(    
            nn.Linear(256 * 8 * 8, 1024),
            nn.LeakyReLU(0.2), 
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2), 
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
        x = self.pool(F.relu(self.conv1(x)))  # [B, 32, 128, 128]
        x = self.pool(F.relu(self.conv2(x)))  # [B, 64, 64, 64]
        x = self.pool(F.relu(self.conv3(x)))  # [B, 128, 32, 32]
        x = self.pool(F.relu(self.conv4(x)))  # [B, 128, 16, 16]
        x = self.pool(F.relu(self.conv5(x)))  # [B, 256, 8, 8]
        x = x.view(x.size(0), -1)
        x = self.feedforward(x)
        return x
    


num_epochs=30
train_dir = '../datasets/real-fake_faces/train'
val_dir = '../datasets/real-fake_faces/valid'

train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)

device='cuda' if torch.cuda.is_available() else 'cpu'
print(device)

data_root = '../datasets/real-fake_faces'
model = CNNClassifier()
train_model(model, train_loader, val_loader, num_epochs=num_epochs, device=device)

# torch.save(model.state_dict(), f'../checkpoints/CNN_3ch/ckpt_{num_epochs}.pt')