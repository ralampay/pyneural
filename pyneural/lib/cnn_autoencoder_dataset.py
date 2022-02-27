import sys
import os
import torch
from torch.utils.data import Dataset
import cv2

class CnnAutoencoderDataset(Dataset):
    def __init__(self, img_dir, img_width, img_height):
        self.img_dir    = img_dir
        self.img_width  = img_width
        self.img_height = img_height
        self.images     = os.listdir(img_dir)

        self.dim = (img_width, img_height)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.images[index])

        img = (cv2.resize(cv2.imread(img_path), self.dim) / 255).transpose((2, 0, 1))

        # Input is the Output
        return torch.Tensor(img), torch.Tensor(img)
