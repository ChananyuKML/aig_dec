import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from einops import rearrange

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
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
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
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'../checkpoints/HoG/ViT/ckpt_{epoch+1}.pth')
            print(f"✅ Saved new best model at epoch {epoch+1} with val loss {val_loss:.4f}")

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

num_epochs=100
device='cuda' if torch.cuda.is_available() else 'cpu'
print(device)

data_root = '../datasets/real-fake_faces_processed'
train_loader, valid_loader, test_loader = get_dataloaders(data_root, batch_size=128)
model = ViT2Channel()
train_model(model, train_loader, valid_loader, num_epochs=num_epochs, device=device)

torch.save(model.state_dict(), f'../model/HoG_ViT/model_from_ckpt_{num_epochs}.pt')