import torch
import torch.nn.functional as f
from torchvision import transforms
from PIL import Image, ImageFilter
from einops import rearrange



class PatchesSplitting(object):
    def __init__(self, img_size=256, patch_size=32):
        self.patch_num = img_size//patch_size
        self.patch_size = patch_size

    def __call__(self, object):
        assert isinstance(object, torch.Tensor)
        patches = []
        for i in range(self.patch_num):
            for j in range(self.patch_num):
                patch = object[:, i*self.patch_size:(i+1)*self.patch_size, j*self.patch_size:(j+1)*self.patch_size]
                patches.append(patch)
        return torch.stack(patches)

class HogPatch(object):
    def __init__(self):
        self.Gx = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0 ,1]
        ],dtype=torch.float32).reshape(1, 1, 3, 3)

        self.Gy = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ],dtype=torch.float32).reshape(1, 1, 3, 3)

    def __call__(self, object):
        assert isinstance(object, torch.Tensor)
        hist_list = []
        for patch in object:
            x_filtered = f.conv2d(patch, self.Gx, padding='same', groups=1)
            y_filtered = f.conv2d(patch, self.Gy, padding='same', groups=1)
            dir = torch.atan2(y_filtered, x_filtered)
            dir_hist = torch.histc(dir, max=torch.max(dir), min=torch.min(dir), bins=8)
            hist_list.append(dir_hist)
        return torch.stack(hist_list)

class HogTransformScaled(object):
    def __init__(self):

        self.Gx = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0 ,1]
        ],dtype=torch.float32).reshape(1, 1, 3, 3)

        self.Gy = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ],dtype=torch.float32).reshape(1, 1, 3, 3)

    def normalize(self, x):
        denominator = torch.max(x) - torch.min(x)
        return (x - torch.min(x)) / (denominator + 1e-8)
    
    def __call__(self, object):
        assert isinstance(object, torch.Tensor)
        x_filtered = f.conv2d(object, self.Gx, padding='same', groups=1)
        y_filtered = f.conv2d(object, self.Gy, padding='same', groups=1)
        mag, dir = (x_filtered.square() + y_filtered.square()).sqrt(), torch.atan2(y_filtered, x_filtered)
        stacked = torch.cat((self.normalize(mag), self.normalize(dir)), dim=0)
        return stacked
    
class HogTransform(object):
    def __init__(self):

        self.Gx = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0 ,1]
        ],dtype=torch.float32).reshape(1, 1, 3, 3)

        self.Gy = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ],dtype=torch.float32).reshape(1, 1, 3, 3)

    def __call__(self, object):
        assert isinstance(object, torch.Tensor)
        x_filtered = f.conv2d(object, self.Gx, padding='same', groups=1)
        y_filtered = f.conv2d(object, self.Gy, padding='same', groups=1)
        mag, dir = (x_filtered.square() + y_filtered.square()).sqrt(), torch.atan2(y_filtered, x_filtered)
        stacked = torch.stack((mag, dir), dim=1)
        return stacked.squeeze(0)
    
class HogFFT(object):
    def __init__(self):

        self.Gx = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0 ,1]
        ],dtype=torch.float32).reshape(1, 1, 3, 3)

        self.Gy = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ],dtype=torch.float32).reshape(1, 1, 3, 3)

    def fft_map(self, input_tensor):
        f = torch.fft.fft2(input_tensor)
        fshift = torch.fft.fftshift(f)
        magnitude = torch.abs(fshift)
        spectrum = torch.log1p(magnitude)
        return (spectrum/(torch.max(spectrum) + 1e-8))

    def __call__(self, object):
        assert isinstance(object, torch.Tensor)
        x_filtered = f.conv2d(object, self.Gx, padding='same', groups=1)
        y_filtered = f.conv2d(object, self.Gy, padding='same', groups=1)
        mag, dir = (x_filtered.square() + y_filtered.square()).sqrt(), torch.atan2(y_filtered, x_filtered)
        mag_fft = self.fft_map(mag)
        dir_fft = self.fft_map(dir)
        stacked = torch.stack((mag_fft, dir_fft), dim=1)
        return stacked.squeeze(0)

    
        
class Hog2Hist(object):
    def __init__(self):

        self.Gx = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0 ,1]
        ],dtype=torch.float32).reshape(1, 1, 3, 3)

        self.Gy = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ],dtype=torch.float32).reshape(1, 1, 3, 3)

    def normalize(self, x):
        return (x-torch.min(x))/(torch.max(x)-torch.min(x))
    
    def __call__(self, object):
        assert isinstance(object, torch.Tensor)
        x_filtered = f.conv2d(object, self.Gx, padding='same', groups=1)
        y_filtered = f.conv2d(object, self.Gy, padding='same', groups=1)
        mag, dir = (x_filtered.square() + y_filtered.square()).sqrt(), torch.atan2(y_filtered, x_filtered)
        mag_norm, dir_norm = self.normalize(mag), self.normalize(dir)
        mag_norm_hist, dir_norm_hist = torch.histc(mag_norm*255, bins=256, min=0, max=255), torch.histc(dir_norm*255, bins=256, min=0, max=255)
        # stacked = torch.stack((mag_norm_hist, dir_norm_hist), dim=1)
        # return stacked.unsqueeze(0)
    
        concatenated = torch.concatenate((mag_norm_hist, dir_norm_hist), dim=0)
        return concatenated.unsqueeze(0)



hog_hist = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        Hog2Hist(),
    ])

hog_hist_8D = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        PatchesSplitting(),
        HogPatch()
    ])

hog_256 = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        HogTransform(),
    ])

hog_224 = transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        HogTransformScaled(),
    ])


hog_224_vit = transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        HogTransformScaled(),
        transforms.Normalize([0.5, 0.5], [0.5,0.5])
    ])

hogfft_224 = transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        HogFFT()
    ])

hogfft_256 = transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomResizedCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        HogFFT()
    ])

rgb_224 = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])
