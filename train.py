import torch.optim as optim
import torch
import argparse
from utils import train_opt, create_dataloader, custom_transforms, create_dataset
from networks.hog_vit import *
from networks.hog_resnet import *
from networks.hog_cnn import *
import os
import matplotlib.pyplot as plt
import pathlib


def train_model(model,model_name, transform, train_loader, valid_loader, num_epochs=30, lr=1e-4, device="cuda"):
    model.to(device)
    out_path = f'checkpoints/{model_name}_{transform}_norm'
    path = pathlib.Path(out_path)
    train_loss_hist = []
    val_loss_hist = []
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if path.exists() and path.is_dir():
        print("Directory exists:", path)
    else:
        os.mkdir(out_path)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        best_val_loss = float('inf')
        correct = 0
        iter = 1

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
            if iter%10==1:
                print(f"currently training at iteration : {iter} / {len(train_loader)}", end="\r", flush=True)
            iter += 1

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
            torch.save(model.state_dict(), f'{out_path}/ckpt_{epoch+1}.pth')
            print(f"✅ Saved new best model at epoch {epoch+1} with val loss {val_loss:.4f}")
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)

    return train_loss_hist, val_loss_hist

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available else 'cpu'

    opt = train_opt.TrainOptions().parse()

    match opt.network:
        case "ViT2Channels":
            model = ViT2Ch()
        case "dual_cnn":
            model = DualCNN()
        case "Res18_2Channels":
            model = Resnet()
        case "HogHistTrans":
            model = HogHistTransformer()
        case "pretrain_vit":
            model = PretrainViT()
        case "HogHist8D":
            model = HogHist8D()
        case "_":
            print("Unknown Loader")

    match opt.transform:
        case "hog_224":
            transform = custom_transforms.hog_224
        case "hogfft_224":
            transform = custom_transforms.hogfft_224
        case "hogfft_256":
            transform = custom_transforms.hogfft_256
        case "hog_256":
            transform = custom_transforms.hog_256
        case "hog_hist":
            transform = custom_transforms.hog_hist
        case "hog_hist_8D":
            transform = custom_transforms.hog_hist_8D
        case "_":
            print("Unknown Loader")

    match opt.loader:
        case "createByDir":
            train_loader, val_loader, _ = create_dataloader.createByDir(opt.dataroot, opt.batch_size, transform=transform)
        case "createByDir2":
            train_loader, val_loader = create_dataloader.createByDir2(opt.dataroot, opt.batch_size, transform=transform)
        case "_":
            print("Unknown Loader")

    print(f"dataset created\ndevice={device}")
    train_hist, val_hist = train_model(model=model, model_name=opt.network, transform=opt.transform, train_loader=train_loader, valid_loader=val_loader, num_epochs=opt.epochs, lr=opt.lr, device=device)

    plt.figure(figsize=(8, 5))
    plt.plot(train_hist, label='Train Loss')
    plt.plot(val_hist, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{opt.network}_{opt.transform} Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'output/graph/{opt.network}_{opt.transform}.png')  # Save the plot
    plt.show()