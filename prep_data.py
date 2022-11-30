from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os


class MyDataset(Dataset):
    def __init__(self, image_paths, transform=None, transform_target=None):
        self.image_paths = image_paths
        self.transform = transform
        self.transform_target = transform_target

    def __getitem__(self, index):
        x = Image.open(self.image_paths[index])
        if self.transform_target:
            y = self.transform_target(x)
        else:
            y = x.copy()

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.image_paths)
