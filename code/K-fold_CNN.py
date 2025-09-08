import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
import numpy as np

# ----- Config -----
batch_size = 32
k_folds = 5
data_root = '../datasets/real-fake_faces_processed'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    

# ----- Image Transform (same as training) -----
transform = transforms.Compose([
    transforms.ToTensor()
])

# ----- Load Dataset -----
test_dataset  = NPYDataset(os.path.join(data_root, 'test'))

# ----- Define CNN (same as training) -----
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
    
# ----- K-Fold Cross-Validation -----
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
indices = list(range(len(test_dataset)))
fold_accuracies = []

for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
    print(f"\n Fold {fold+1}/{k_folds}")

    val_subset = Subset(test_dataset, val_idx)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    # Load trained model
    model = CNNClassifier().to(device)
    model.load_state_dict(torch.load("../checkpoints/HoG_CNN/ckpt_28.pth", map_location=device))
    model.eval()

    # Evaluation
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)
            outputs = model(inputs)
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    fold_accuracies.append(acc)
    print(f"✅ Fold {fold+1} Accuracy: {acc * 100:.2f}%")

# ----- Final Results -----
mean_acc = np.mean(fold_accuracies)
std_acc = np.std(fold_accuracies)
print(f"\n🎯 Average Accuracy: {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%")

fold_labels = [1, 2, 3, 4, 5]


# Convert to percentages
accuracy_percent = [a * 100 for a in fold_accuracies]
average_accuracy = mean_acc*100
# Plot
plt.figure(figsize=(8, 5))
plt.bar(fold_labels, accuracy_percent, color='skyblue', edgecolor='black')
plt.ylim(95, 100)
plt.xlabel("Fold")
plt.ylabel("Accuracy (%)")
plt.title("K-Fold Cross-Validation Accuracy")
for i, acc in enumerate(accuracy_percent):
    plt.text(fold_labels[i], acc + 0.05, f"{acc:.2f}%", ha='center')
plt.axhline(y=average_accuracy, color='red', linestyle='--', linewidth=1.5, label=f'Average = {average_accuracy:.2f}%')
plt.legend()
plt.tight_layout()
plt.savefig("kfold_CNN_HoG.png")
print("📊 Saved as 'kfold_CNN_HoG.png'")
plt.show()