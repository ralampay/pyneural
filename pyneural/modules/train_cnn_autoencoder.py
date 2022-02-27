import sys
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib.cnn_autoencoder import CnnAutoencoder
from lib.cnn_autoencoder_dataset import CnnAutoencoderDataset

class TrainCnnAutoencoder:
    def __init__(self, params={}):
        self.params = params

        self.device         = params.get('device')          or 'cpu' 
        self.gpu_index      = params.get('gpu_index')       or 0
        self.epochs         = params.get('epochs')          or 100
        self.learning_rate  = params.get('learning_rate')   or 0.00001
        self.chunk_size     = params.get('chunk_size')      or 100
        self.batch_size     = params.get('batch_size')      or 50
        self.cont           = params.get('cont')            or False
        self.kernel_size    = params.get('kernel_size')     or 3
        self.model_file     = params.get('model_file')
        self.channel_maps   = params.get('channel_maps')
        self.padding        = params.get('padding')
        self.scale          = params.get('scale')
        self.img_height     = params.get('img_height')
        self.img_width      = params.get('img_width')
        self.train_img_dir  = params.get('train_img_dir')
        self.num_channels   = params.get('num_channels')

    def execute(self):
        print("Training CNN Autoencoder model...")
        
        if self.device == 'cuda':
            print("CUDA Device: {}".format(torch.cuda.get_device_name(self.gpu_index)))
            self.device = "cuda:{}".format(self.gpu_index)

        model   =   CnnAutoencoder(
                        scale=self.scale,
                        channel_maps=self.channel_maps,
                        padding=self.padding,
                        kernel_size=self.kernel_size,
                        num_channels=self.num_channels,
                        img_width=self.img_width,
                        img_height=self.img_height
                    ).to(self.device)

        if self.cont:
            print("Loading model from {}".format(self.model_file))
            state = torch.load(self.model_file)
            model.load_state_dict(state['state_dict'])
            model.optimizer = state['optimizer']

        loss_fn     = nn.BCELoss()
        optimizer   = optim.Adam(model.parameters(), lr=self.learning_rate)
        scaler      = torch.cuda.amp.GradScaler()

        train_ds = CnnAutoencoderDataset(
            img_dir=self.train_img_dir,
            img_width=self.img_width,
            img_height=self.img_height
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False
        )

        for epoch in range(self.epochs):
            print("Epoch: {}".format(epoch))
            self.train_fn(train_loader, model, optimizer, loss_fn, scaler)

            # Save model after every epoch
            print("Saving model to {}...".format(self.model_file))

            state = {
                'params':       self.params,
                'state_dict':   model.state_dict(),
                'optimizer':    optimizer.state_dict()
            }

            torch.save(state, self.model_file)

        print("Done.")

    def train_fn(self, loader, model, optimizer, loss_fn, scaler):
        loop = tqdm(loader)

        for batch_idx, (data, targets) in enumerate(loop):
            data    = data.float().to(device=self.device)
            targets = targets.float().to(device=self.device)

            # Forward
            predictions = model.forward(data)

            loss = loss_fn(predictions, targets)

            # Backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # update tqdm
            loop.set_postfix(loss=loss.item())
