import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class NPYDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.label_map = {'real': 0, 'fake': 1}

        for label_name in ['real', 'fake']:
            class_dir = os.path.join(root_dir, label_name)
            for fname in os.listdir(class_dir):
                if fname.endswith('.npy'):
                    fpath = os.path.join(class_dir, fname)
                    self.samples.append((fpath, self.label_map[label_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        data = np.load(fpath)
        data = np.transpose(data, (2, 0, 1))
        data_tensor = torch.from_numpy(data).float()  # convert to float tensor
        return data_tensor, label
    

def get_dataloaders(data_root, batch_size=32, num_workers=2):
    train_dataset = NPYDataset(os.path.join(data_root, 'train'))
    valid_dataset = NPYDataset(os.path.join(data_root, 'valid'))
    test_dataset  = NPYDataset(os.path.join(data_root, 'test'))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader, test_loader

def train_model(model, train_loader, valid_loader, batch_size=32, num_epochs=10, lr=1e-4, device='cpu'):
    model.to(device)
    train_acc_list = []
    val_acc_list = []
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
        train_acc_list.append(acc)
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
        val_acc_list.append(val_acc)
        val_loss /= len(valid_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'../checkpoints/HoG_CNN/ckpt_{epoch+1}.pth')
            print(f"✅ Saved new best model at epoch {epoch+1} with val loss {val_loss:.4f}")
    
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_acc_list, label='Train Accuracy', marker='o')
    plt.plot(range(1, num_epochs + 1), val_acc_list, label='Validation Accuracy', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("accuracy_plot.png")  # Optional: save as image
    plt.show()

class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
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
device='cuda' if torch.cuda.is_available() else 'cpu'
print(device)

data_root = '../datasets/real-fake_faces_processed'
train_loader, valid_loader, test_loader = get_dataloaders(data_root, batch_size=128)
model = CNNClassifier()
train_model(model, train_loader, valid_loader, num_epochs=num_epochs, device=device)






# torch.save(model.state_dict(), f'../checkpoints/Cnn/ckpt_CNN_{num_epochs}.pt')