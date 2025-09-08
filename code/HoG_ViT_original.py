import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
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
    train_acc_list, train_loss_list, val_acc_list, val_loss_list=  [], [], [], []

    # Total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    # Trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")

    # Estimate memory for parameters
    mem_params = sum(param.nelement() * param.element_size() for param in model.parameters())

    # Estimate memory for buffers (e.g., BatchNorm statistics)
    mem_bufs = sum(buf.nelement() * buf.element_size() for buf in model.buffers())

    total_model_memory_bytes = mem_params + mem_bufs
    print(f"Estimated model memory: {total_model_memory_bytes / (1024**2):.2f} MB")    

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
        train_acc_list.append(acc)
        train_loss /= len(train_loader.dataset)
        train_loss_list.append(train_loss)
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
        val_acc_list.append
        val_loss /= len(valid_loader.dataset)
        val_loss_list.append(val_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'../checkpoints/HoG_ViT/ckpt_{epoch+1}.pth')
            print(f"✅ Saved new best model at epoch {epoch+1} with val loss {val_loss:.4f}")

    
    plt.figure(figsize=(12, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_acc_list, label='Train Accuracy')
    plt.plot(range(1, num_epochs + 1), val_acc_list, label='Validation Accuracy')
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_loss, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_loss, label='Validation Loss')
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("HOG-ViT_accuracy_loss.png")
    print("📊 Saved as 'HOG-ViT_accuracy_loss.png'")
    plt.show()

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
        x = self.norm11(x + self.c_attn1(x, y, y)[0])
        x = self.norm21(x + self.mlp1(x))
        
        x = self.s_attn2(x)
        x = self.norm12(x + self.c_attn2(x, y, y)[0])
        x = self.norm22(x + self.mlp1(x))

        x = self.s_attn3(x)
        x = self.norm13(x + self.c_attn3(x, y, y)[0])
        x = self.norm23(x + self.mlp3(x))

        x = self.s_attn4(x)
        x = self.norm14(x + self.c_attn4(x, y, y)[0])
        x = self.norm24(x + self.mlp4(x))

        x = self.s_attn5(x)
        x = self.norm15(x + self.c_attn5(x, y, y)[0])
        x = self.norm25(x + self.mlp5(x))

        x = self.s_attn6(x)
        x = self.norm16(x + self.c_attn6(x, y, y)[0])
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

num_epochs=100
device='cuda' if torch.cuda.is_available() else 'cpu'
print(device)

data_root = '../datasets/real-fake_faces_processed'
train_loader, valid_loader, test_loader = get_dataloaders(data_root, batch_size=64)
model = ViT2Channel()
train_model(model, train_loader, valid_loader, num_epochs=num_epochs, device=device)

torch.save(model.state_dict(), f'../model/HoG_ViT/model_from_ckpt_{num_epochs}.pt')