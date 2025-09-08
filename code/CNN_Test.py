from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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


class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.feedforward = nn.Sequential(    
            nn.Linear(128 * 16 * 16, 256),
            nn.LeakyReLU(0.2), 
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 2),
        )
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [B, 32, 128, 128]
        x = self.pool(F.relu(self.conv2(x)))  # [B, 64, 64, 64]
        x = self.pool(F.relu(self.conv3(x)))  # [B, 128, 32, 32]
        x = self.pool(F.relu(self.conv4(x)))  # [B, 128, 16, 16]
        x = x.view(x.size(0), -1)
        x = self.feedforward(x)
        return x


def run_kfold_test(data_root, k=5, batch_size=16, model_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader = get_dataloaders(data_root, batch_size=128)

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold, (_, test_idx) in enumerate(kf.split(test_loader)):
        print(f"\n🧪 Fold {fold + 1}/{k} Test Evaluation:")

        model = CNNClassifier()
        if model_path:
            model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)

        model.eval()
        test_correct = 0
        with torch.no_grad():
            for test_data, test_labels in test_loader:
                test_data, test_labels = test_data.to(device), test_labels.to(device)
                test_outputs = model(test_data)
                _, test_preds = torch.max(test_outputs, 1)
                test_correct += (test_preds == test_labels).sum().item()
                test_acc = test_correct / len(test_loader.dataset)

        fold_accuracies.append(test_acc)
        print(f"✅ Fold {fold+1} Accuracy: {test_acc:.4f}")

    print(f"\n Final 5-Fold Average Accuracy: {np.mean(fold_accuracies):.4f}")

if __name__ == "__main__":
    run_kfold_test(
        data_root='../datasets/real-fake_faces_processed',
        model_path='../checkpoints/Cnn/ckpt_CNN_100.pt',  # or None
        k=5,
        batch_size=16
    )