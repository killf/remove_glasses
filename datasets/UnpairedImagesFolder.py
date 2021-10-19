from torch.utils.data import Dataset
from PIL import Image

import random
import os


class UnpairedImagesFolder(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        self.filesA = os.listdir(os.path.join(root, "trainA"))
        self.filesB = os.listdir(os.path.join(root, "trainB"))

    def __len__(self):
        return len(self.filesA)

    def __getitem__(self, idx):
        fileA = os.path.join(self.root, "trainA", self.filesA[idx])
        fileB = os.path.join(self.root, "trainB", random.choice(self.filesB))

        imgA = Image.open(fileA)
        imgB = Image.open(fileB)

        if self.transform is not None:
            imgA = self.transform(imgA)
            imgB = self.transform(imgB)

        return imgA, imgB
