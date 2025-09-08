import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
import numpy as np
from einops import rearrange
transform = transforms.Compose([
    transforms.Resize((256, 256)),   # Resize images
    transforms.ToTensor(),           # Convert to tensor
    transforms.Normalize([0.5], [0.5])  # Normalize (adjust for RGB if needed)
])


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
        self.norm3 = nn.LayerNorm(emb_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return self.norm3(x)


class CustomDecoder(nn.Module):
    def __init__(self, emb_dim, heads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = nn.MultiheadAttention(emb_dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        return self.norm2(x)




class ViT2Channel(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_channels=1, num_classes=2,
                 emb_dim=256, depth=6, heads=8, mlp_dim=512, dropout=0.1):
        super().__init__()

        self.patch_embed_1 = PatchEmbedding(in_channels, patch_size, emb_dim, img_size)
        self.patch_embed_2 = PatchEmbedding(in_channels, patch_size, emb_dim, img_size)

        self.num_patches = (img_size // patch_size) ** 2

        self.cls_token_1 = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.cls_token_2 = nn.Parameter(torch.randn(1, 1, emb_dim))

        self.pos_embed_1 = nn.Parameter(torch.randn(1, self.num_patches + 1, emb_dim))
        self.pos_embed_2 = nn.Parameter(torch.randn(1, self.num_patches + 1, emb_dim))

        self.dropout = nn.Dropout(dropout)

        self.encoder = nn.Sequential(*[
            TransformerEncoder(emb_dim, heads, mlp_dim, dropout) for _ in range(depth)
        ])

       
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, 1),
            nn.Sigmoid()
        )

        self.depth = depth
        self.s_attn1 = CustomDecoder(emb_dim, heads, dropout=dropout)
        self.s_attn2 = CustomDecoder(emb_dim, heads, dropout=dropout)
        self.s_attn3 = CustomDecoder(emb_dim, heads, dropout=dropout)
        self.s_attn4 = CustomDecoder(emb_dim, heads, dropout=dropout)
        self.s_attn5 = CustomDecoder(emb_dim, heads, dropout=dropout)
        self.s_attn6 = CustomDecoder(emb_dim, heads, dropout=dropout)
        # self.s_attn7 = CustomDecoder(emb_dim, heads, dropout=dropout)
        # self.s_attn8 = CustomDecoder(emb_dim, heads, dropout=dropout)       


        self.c_attn1 = nn.MultiheadAttention(emb_dim, heads, dropout=dropout, batch_first=True)
        self.c_attn2 = nn.MultiheadAttention(emb_dim, heads, dropout=dropout, batch_first=True)
        self.c_attn3 = nn.MultiheadAttention(emb_dim, heads, dropout=dropout, batch_first=True)
        self.c_attn4 = nn.MultiheadAttention(emb_dim, heads, dropout=dropout, batch_first=True)
        self.c_attn5 = nn.MultiheadAttention(emb_dim, heads, dropout=dropout, batch_first=True)
        self.c_attn6 = nn.MultiheadAttention(emb_dim, heads, dropout=dropout, batch_first=True)
        # self.c_attn7 = nn.MultiheadAttention(emb_dim, heads, dropout=dropout, batch_first=True)
        # self.c_attn8 = nn.MultiheadAttention(emb_dim, heads, dropout=dropout, batch_first=True)

        self.norm11 = nn.LayerNorm(emb_dim)
        self.norm12 = nn.LayerNorm(emb_dim)
        self.norm13 = nn.LayerNorm(emb_dim)
        self.norm14 = nn.LayerNorm(emb_dim)
        self.norm15 = nn.LayerNorm(emb_dim)
        self.norm16 = nn.LayerNorm(emb_dim)
        # self.norm17 = nn.LayerNorm(emb_dim)
        # self.norm18 = nn.LayerNorm(emb_dim)

        self.mlp1 = nn.Sequential(
            nn.Linear(emb_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, emb_dim),
            nn.Dropout(dropout)
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(emb_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, emb_dim),
            nn.Dropout(dropout)
        )

        self.mlp3 = nn.Sequential(
            nn.Linear(emb_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, emb_dim),
            nn.Dropout(dropout)
        )

        self.mlp4 = nn.Sequential(
            nn.Linear(emb_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, emb_dim),
            nn.Dropout(dropout)
        )

        self.mlp5 = nn.Sequential(
            nn.Linear(emb_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, emb_dim),
            nn.Dropout(dropout)
        )

        self.mlp6 = nn.Sequential(
            nn.Linear(emb_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, emb_dim),
            nn.Dropout(dropout)
        )

        # self.mlp7 = nn.Sequential(
        #     nn.Linear(emb_dim, mlp_dim),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(mlp_dim, emb_dim),
        #     nn.Dropout(dropout)
        # )

        # self.mlp8 = nn.Sequential(
        #     nn.Linear(emb_dim, mlp_dim),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(mlp_dim, emb_dim),
        #     nn.Dropout(dropout)
        # )

        self.norm21 = nn.LayerNorm(emb_dim)
        self.norm22 = nn.LayerNorm(emb_dim)
        self.norm23 = nn.LayerNorm(emb_dim)
        self.norm24 = nn.LayerNorm(emb_dim)
        self.norm25 = nn.LayerNorm(emb_dim)
        self.norm26 = nn.LayerNorm(emb_dim)
        # self.norm27 = nn.LayerNorm(emb_dim)
        # self.norm28 = nn.LayerNorm(emb_dim)


    def decoder(self, x, y):
        x = self.s_attn1(x)
        x = self.norm11(x + self.c_attn1(y, x, x)[0])
        x = self.norm21(x + self.mlp1(x))
        
        x = self.s_attn2(x)
        x = self.norm12(x + self.c_attn2(y, x, x)[0])
        x = self.norm22(x + self.mlp1(x))

        x = self.s_attn3(x)
        x = self.norm13(x + self.c_attn3(y, x, x)[0])
        x = self.norm23(x + self.mlp3(x))

        x = self.s_attn4(x)
        x = self.norm14(x + self.c_attn4(y, x, x)[0])
        x = self.norm24(x + self.mlp4(x))

        x = self.s_attn5(x)
        x = self.norm15(x + self.c_attn5(y, x, x)[0])
        x = self.norm25(x + self.mlp5(x))

        x = self.s_attn6(x)
        x = self.norm16(x + self.c_attn6(y, x, x)[0])
        x = self.norm26(x + self.mlp6(x))

        # x = self.s_attn7(x)
        # x = self.norm17(x + self.c_attn7(y, x, x)[0])
        # x = self.norm27(x + self.mlp7(x))

        # x = self.s_attn8(x)
        # x = self.norm18(x + self.c_attn8(y, x, x)[0])
        # x = self.norm28(x + self.mlp8(x))

        return x

    def forward(self, x):
        B = x.shape[0]
        x1 = x[:, 0, :, :].unsqueeze(1)
        x2 = x[:, 1, :, :].unsqueeze(1)
        x1 = self.patch_embed_1(x1)  # [B, num_patches, emb_dim]
        x2 = self.patch_embed_2(x2)
        cls_tokens_1 = self.cls_token_1.expand(B, -1, -1)  # [B, 1, emb_dim]
        cls_tokens_2 = self.cls_token_2.expand(B, -1, -1)
        x1 = torch.cat((cls_tokens_1, x1), dim=1)  # [B, num_patches + 1, emb_dim]
        x2 = torch.cat((cls_tokens_2, x2), dim=1) 
        x1 = x1 + self.pos_embed_1
        x2 = x2 + self.pos_embed_2
        x1 = self.dropout(x1)
        x2 = self.dropout(x2)

        x1 = self.encoder(x1)
        x2 = self.decoder(x2, x1)
        cls_output = x2[:, 0]  # CLS token
        return self.mlp_head(cls_output)

    
# ----- K-Fold Cross-Validation -----
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
indices = list(range(len(test_dataset)))
fold_accuracies = []

for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
    print(f"\n Fold {fold+1}/{k_folds}")

    val_subset = Subset(test_dataset, val_idx)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    # Load trained model
    model = ViT2Channel().to(device)
    model.load_state_dict(torch.load("../checkpoints/HoG_ViT/ckpt_81.pth", map_location=device))
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
plt.savefig("kfold_ViT_ED.png")
print("📊 Saved as 'kfold_ViT_ED.png")
plt.show()