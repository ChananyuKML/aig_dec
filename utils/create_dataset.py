import os
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image

class CNNDetectionataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.data = []
        self.transform = transform
        self.label_map = {"0_real": 0, "1_fake": 1}
        # print(os.listdir(root_dir))
        # walk through dataset/classX/{real,fake}
        for class_dir in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_dir)
            # print(class_path)
            if not os.path.isdir(class_path):
                continue

            for label_name in ["0_real", "1_fake"]:
                label_path = os.path.join(class_path, label_name)
                # print(label_path)
                if not os.path.isdir(label_path):
                    continue
                
                for file_name in os.listdir(label_path):
                    file_path = os.path.join(label_path, file_name)
                    if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                        self.data.append((file_path, self.label_map[label_name]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # print(len(self.data))
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
    


class CNNDetectionataset2(Dataset):
    def __init__(self, root_dir, transform=None):
        self.data = []
        self.transform = transform
        self.label_map = {"real": 0, "fake": 1}
        # print(os.listdir(root_dir))
        # walk through dataset/classX/{real,fake}
        for class_dir in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_dir)
            # print(class_path)
            if not os.path.isdir(class_path):
                continue

        for label_name in ["real", "fake"]:
            for file_name in os.listdir(class_path):
                file_path = os.path.join(class_path, file_name)
                if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.data.append((file_path, self.label_map[label_name]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


class genImage(Dataset):
    def __init__(self, root_dir, transform=None):
        self.data = []
        self.transform = transform
        self.label_map = {"nature": 0, "ai": 1}
        
        # Iterate through the label directories ('nature', 'ai')
        for label_name in self.label_map.keys():
            class_path = os.path.join(root_dir, label_name)

            if not os.path.isdir(class_path):
                continue

            # Get the correct label index
            label_index = self.label_map[label_name]

            # Add each file in the directory with its correct label
            for file_name in os.listdir(class_path):
                file_path = os.path.join(class_path, file_name)
                if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.data.append((file_path, label_index))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # print(len(self.data))
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
