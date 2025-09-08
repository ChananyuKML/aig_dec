import torch
import math
import timm
import torch.nn as nn
import numpy as np
from einops import rearrange

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
            nn.Linear(mlp_dim, emb_dim)
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




class ViT2Channels(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=1, num_classes=2,
                 emb_dim=768, depth=6, heads=8, mlp_dim=512, dropout=0.1):
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


        self.c_attn1 = nn.MultiheadAttention(emb_dim, heads, dropout=dropout, batch_first=True)
        self.c_attn2 = nn.MultiheadAttention(emb_dim, heads, dropout=dropout, batch_first=True)
        self.c_attn3 = nn.MultiheadAttention(emb_dim, heads, dropout=dropout, batch_first=True)
        self.c_attn4 = nn.MultiheadAttention(emb_dim, heads, dropout=dropout, batch_first=True)
        self.c_attn5 = nn.MultiheadAttention(emb_dim, heads, dropout=dropout, batch_first=True)
        self.c_attn6 = nn.MultiheadAttention(emb_dim, heads, dropout=dropout, batch_first=True)

        self.norm11 = nn.LayerNorm(emb_dim)
        self.norm12 = nn.LayerNorm(emb_dim)
        self.norm13 = nn.LayerNorm(emb_dim)
        self.norm14 = nn.LayerNorm(emb_dim)
        self.norm15 = nn.LayerNorm(emb_dim)
        self.norm16 = nn.LayerNorm(emb_dim)


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


        self.norm21 = nn.LayerNorm(emb_dim)
        self.norm22 = nn.LayerNorm(emb_dim)
        self.norm23 = nn.LayerNorm(emb_dim)
        self.norm24 = nn.LayerNorm(emb_dim)
        self.norm25 = nn.LayerNorm(emb_dim)
        self.norm26 = nn.LayerNorm(emb_dim)



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

class HogHistTransformer(nn.Module):
    def __init__(self, input_len=512, input_dim=1, mlp_dim=512,
                 emb_dim=768, depth=6, heads=8, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Linear(input_dim, emb_dim)

        pos_emb = torch.zeros(input_len, emb_dim)
        position = torch.arange(0, input_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim))
        pos_emb[:, 0::2] = torch.sin(position * div_term)
        pos_emb[:, 1::2] = torch.cos(position * div_term)
        self.pos_emb = pos_emb.to("cuda")

        self.dropout = nn.Dropout(dropout)

        self.transformer = nn.Sequential(*[
            TransformerEncoder(emb_dim, heads, mlp_dim, dropout) for _ in range(depth)
        ])

        self.pooling = nn.Linear(emb_dim, 1)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(mlp_dim),
            nn.Linear(mlp_dim, mlp_dim//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim//2, 1),
            nn.Sigmoid()
        )

    def pos_embedding(self, x):
        return self.pos_emb + x
    
    def forward(self, x):
        x = rearrange(x, 'b d l -> b l d')
        x = self.token_emb(x)
        x = self.pos_embedding(x)
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.pooling(x).squeeze(-1)
        return self.mlp_head(x)
    

class HogHist8D(nn.Module):
    def __init__(self, input_len=64, input_dim=8, mlp_dim=512,
                 emb_dim=768, depth=6, heads=8, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Linear(input_dim, emb_dim)

        pos_emb = torch.zeros(input_len, emb_dim)
        position = torch.arange(0, input_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim))
        pos_emb[:, 0::2] = torch.sin(position * div_term)
        pos_emb[:, 1::2] = torch.cos(position * div_term)
        self.pos_emb = pos_emb.to("cuda")

        self.dropout = nn.Dropout(dropout)

        self.transformer = nn.Sequential(*[
            TransformerEncoder(emb_dim, heads, mlp_dim, dropout) for _ in range(depth)
        ])

        self.pooling = nn.Sequential(nn.Conv1d(in_channels=emb_dim, out_channels=mlp_dim, kernel_size=3, padding=1),
                                     nn.GELU(),
                                     nn.AdaptiveAvgPool1d(1))

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(mlp_dim),
            nn.Linear(mlp_dim, mlp_dim//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim//2, 1),
            nn.Sigmoid()
        )

    def pos_embedding(self, x):
        return self.pos_emb + x
    
    def forward(self, x):
        x = self.token_emb(x)
        x = self.pos_embedding(x)
        x = self.transformer(self.dropout(x))
        x = rearrange(x, 'b l d -> b d l')
        x = self.pooling(x).squeeze(-1)
        return self.mlp_head(x)
    
class HogHist8D_Convo(nn.Module):
    def __init__(self, input_len=64, input_dim=8, mlp_dim=512,
                 emb_dim=768, depth=6, heads=8, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Linear(input_dim, emb_dim)

        pos_emb = torch.zeros(input_len, emb_dim)
        position = torch.arange(0, input_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim))
        pos_emb[:, 0::2] = torch.sin(position * div_term)
        pos_emb[:, 1::2] = torch.cos(position * div_term)
        self.pos_emb = pos_emb.to("cuda")

        self.dropout = nn.Dropout(dropout)

        self.transformer = nn.Sequential(*[
            TransformerEncoder(emb_dim, heads, mlp_dim, dropout) for _ in range(depth)
        ])

        self.pooling = nn.Sequential(nn.Conv1d(in_channels=emb_dim, out_channels=emb_dim, kernel_size=3, padding=1),
                                     nn.AdaptiveAvgPool1d(1))

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, 1),
            nn.Sigmoid()
        )

    def pos_embedding(self, x):
        return self.pos_emb + x
    
    def forward(self, x):
        print(x.size())
        x = self.token_emb(x)
        x = self.pos_embedding(x)
        x = self.transformer(self.dropout(x))
        x = rearrange(x, 'b l d -> b d l')
        x = self.pooling(x).squeeze(-1)
        return self.mlp_head(x)
    


class ViT2Ch(nn.Module):
    def __init__(self, img_size=224, patch_size=16, input_dim=2, mlp_dim=512,
                 emb_dim=768, depth=6, heads=8, dropout=0.1):
        super().__init__()

        self.patch_embed = PatchEmbedding(input_dim, patch_size, emb_dim, img_size)

        self.num_patches = (img_size // patch_size) ** 2

        pos_emb = torch.zeros(self.num_patches, emb_dim)
        position = torch.arange(0, self.num_patches, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim))
        pos_emb[:, 0::2] = torch.sin(position * div_term)
        pos_emb[:, 1::2] = torch.cos(position * div_term)
        self.pos_emb = pos_emb.to("cuda")

        self.dropout = nn.Dropout(dropout)

        self.transformer = nn.Sequential(*[
            TransformerEncoder(emb_dim, heads, mlp_dim, dropout) for _ in range(depth)
        ])

        self.pooling = nn.Sequential(nn.Conv1d(in_channels=emb_dim, out_channels=mlp_dim, kernel_size=3, padding=1),
                                     nn.GELU(),
                                     nn.AdaptiveAvgPool1d(1))

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(mlp_dim),
            nn.Linear(mlp_dim, mlp_dim//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim//2, 1),
            nn.GELU(),
            nn.Sigmoid()
        )

    
    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_emb
        # x = rearrange(x, 'b l d -> b d l')
        x = self.transformer(self.dropout(x))
        x = rearrange(x, 'b l d -> b d l')
        x = self.pooling(x).squeeze(-1)
        return self.mlp_head(x)

class PretrainViT(nn.Module):
    def __init__(self, vit_model_name='vit_base_patch16_224', num_classes=2, emb_dim=768, depth=6, heads=8, dropout=0.1):
        super().__init__()

        self.vit = timm.create_model(vit_model_name, pretrained=True, num_classes=num_classes)

        self.vit.patch_embed.proj = nn.Conv2d(2, emb_dim,
                                                  kernel_size=self.vit.patch_embed.proj.kernel_size,
                                                  stride=self.vit.patch_embed.proj.stride)
        self.cls = nn.Sequential(
            nn.LayerNorm(2),
            nn.Linear(2, 1),
            nn.GELU(),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.cls(self.vit(x))