import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import os
import random
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CnnAutoencoder(nn.Module):
    def __init__(self, scale=2, channel_maps=[], padding=1, kernel_size=3, num_channels=3, img_width=100, img_height=100, device=torch.device("cpu")):
        super().__init__()

        self.device = device

        self.img_width      = img_width
        self.img_height     = img_height
        self.num_channels   = num_channels
        self.kernel_size    = kernel_size
        self.padding        = padding
        self.channel_maps   = channel_maps
        self.scale          = scale

        self.reversed_channel_maps = list(reversed(channel_maps))

        # Build convolutional layers
        self.convolutional_layers = nn.ModuleList([])

        for i in range(len(self.channel_maps) - 1):
            self.convolutional_layers.append(nn.Conv2d(self.channel_maps[i], self.channel_maps[i+1], kernel_size=self.kernel_size, padding=self.padding))

        # Build deconvolutional layers
        self.deconvolutional_layers = nn.ModuleList([])

        for i in range(len(self.reversed_channel_maps) - 1):
            self.deconvolutional_layers.append(nn.ConvTranspose2d(self.reversed_channel_maps[i], self.reversed_channel_maps[i+1], 2, stride=2))

    def conv(self, x):
        for i in range(len(self.convolutional_layers)):
            conv_layer = self.convolutional_layers[i]

            x = F.max_pool2d(F.relu(conv_layer(x)), self.scale)
        
        return x

    def compress(self, x):
        x = self.conv(x)
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])

        return x

    def deconv(self, x):
        for i in range(len(self.deconvolutional_layers)):
            deconv_layer = self.deconvolutional_layers[i]
            x = deconv_layer(x)

            if i != len(self.deconvolutional_layers) - 1:
                x = F.relu(x)
            else:
                x = torch.sigmoid(x)

        return x

    def forward(self, x):
        x = self.conv(x)
        x = self.deconv(x)

        return x
