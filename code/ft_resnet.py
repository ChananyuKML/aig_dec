import torch
import torch.nn as nn
from torchvision import datasets, models

class FullyConnected(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.feedforward = nn.Sequential(    
            nn.Linear(in_features, 1024),
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
            nn.Linear(32, out_features),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.feedforward(x)

resnet = models.resnet18(pretrained=True)

# Freeze all convolutional weights (everything except the final FC layer)
for param in resnet.parameters():
    param.requires_grad = False

in_features = resnet.fc.in_features
resnet.fc = FullyConnected(in_features, 1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = resnet.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(resnet.fc.parameters(), lr=1e-4)  # Only fc layer is trainable

def train_one_epoch(model, dataloader):
    model.train()
    running_loss = 0.0
    correct = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        running_loss += loss.item() * images.size(0)

    accuracy = correct / len(dataloader.dataset)
    return running_loss / len(dataloader.dataset), accuracy

if __name__ == "__main__":

    torch.save(resnet.state_dict(), "resnet18_frozen_conv.pt")