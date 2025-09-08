import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

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
        # Save loss history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss + train_loss
            torch.save(model.state_dict(), f'../checkpoints/res18_vanila/ckpt_{epoch+1}.pth')
            print(f"✅ Saved new best model at epoch {epoch+1} with val loss {val_loss:.4f}")

    


class CustomFeedForward(nn.Module):
    def __init__(self, input_dim):
        super(CustomFeedForward, self).__init__()
        self.ff = nn.Sequential(    
            # nn.Linear(256 * 8 * 8, 1024),
            # nn.LeakyReLU(0.2), 
            # nn.Linear(1024, 512),
            # nn.LeakyReLU(0.2), 
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
    


resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)



# Freeze conv layers (optional)
# for param in resnet.parameters():
#     param.requires_grad = False

# Replace fc layer
num_ftrs = resnet.fc.in_features
resnet.fc = CustomFeedForward(input_dim=num_ftrs)



num_epochs=30
train_dir = '../datasets/real-fake_faces/train'
val_dir = '../datasets/real-fake_faces/valid'
train_losses = []
val_losses = []

train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)

device='cuda' if torch.cuda.is_available() else 'cpu'
print(device)

data_root = '../datasets/real-fake_faces'
model = resnet.to(device='cuda')
train_model(model, train_loader, val_loader, num_epochs=num_epochs, device=device)

plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Resnet18 Vanila Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('loss_resnet18_vanila.png')  # Save the plot
plt.show()
# torch.save(model.state_dict(), f'../checkpoints/CNN_3ch/ckpt_{num_epochs}.pt')