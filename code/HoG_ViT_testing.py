import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from einops import rearrange
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
import numpy as np

# ----- Config -----
matplotlib.use('Qt5Agg')
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
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, emb_dim, img_size):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, emb_dim, H', W']
        x = rearrange(x, 'b c h w -> b (h w) c')  # [B, num_patches, emb_dim]
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, emb_dim, heads, mlp_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = nn.MultiheadAttention(emb_dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, emb_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class ViT2Channel(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_channels=2, num_classes=2,
                 emb_dim=256, depth=8, heads=8, mlp_dim=512, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, emb_dim, img_size)
        self.num_patches = (img_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, emb_dim))
        self.dropout = nn.Dropout(dropout)

        self.transformer = nn.Sequential(*[
            TransformerEncoder(emb_dim, heads, mlp_dim, dropout) for _ in range(depth)
        ])
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # [B, num_patches, emb_dim]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, emb_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, num_patches + 1, emb_dim]
        x = x + self.pos_embed
        x = self.dropout(x)

        x = self.transformer(x)
        cls_output = x[:, 0]  # CLS token
        return self.mlp_head(cls_output)

    
# ----- K-Fold Cross-Validation -----
kf = KFold(n_splits=k_folds, shuffle=True, random_state=13)
indices = list(range(len(test_dataset)))
fold_accuracies = []

for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
    print(f"\n Fold {fold+1}/{k_folds}")

    val_subset = Subset(test_dataset, val_idx)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    # Load trained model
    model = ViT2Channel().to(device)
    model.load_state_dict(torch.load("../model/HoG_ViT/model_from_ckpt_100_1.pt", map_location=device))
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
print(accuracy_percent)
# Plot
plt.figure(figsize=(8, 5))
plt.bar(fold_labels, accuracy_percent, color='skyblue', edgecolor='black')
plt.ylim(90, 100)
plt.xlabel("Fold")
plt.ylabel("Accuracy (%)")
plt.title("K-Fold Cross-Validation Accuracy")
for i, acc in enumerate(accuracy_percent):
    plt.text(fold_labels[i], acc + 0.05, f"{acc:.2f}%", ha='center')
plt.axhline(y=average_accuracy, color='red', linestyle='--', linewidth=1.5, label=f'Average = {average_accuracy:.2f}%')
plt.legend()
plt.tight_layout()
plt.savefig("kfold_ViT.png")
print("📊 Saved as 'kfold_accuracy_bar_8.png'")
plt.show()