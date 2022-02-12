import sys
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib.autoencoder import Autoencoder
from lib.autoencoder_dataset import AutoencoderDataset

class TrainAutoencoder:
    def __init__(self, params={}):
        self.params = params

        self.layers         = params.get('layers')          or [10, 6]
        self.h_activation   = params.get('h_activation')    or 'relu'
        self.o_activation   = params.get('o_activation')    or 'sigmoid'
        self.device         = params.get('device')          or 'cpu' 
        self.error_type     = params.get('error_type')      or 'mse'
        self.optimizer_type = params.get('optimizer_type')  or 'adam'
        self.gpu_index      = params.get('gpu_index')       or 0
        self.epochs         = params.get('epochs')          or 100
        self.learning_rate  = params.get('learning_rate')   or 0.001
        self.chunk_size     = params.get('chunk_size')      or 100
        self.batch_size     = params.get('batch_size')      or 50
        self.cont           = params.get('cont')            or False
        self.model_file     = params.get('model_file')
        self.training_file  = params.get('training_file')

    def execute(self):
        print("Training model...")
        
        if self.device == 'cuda':
            print("CUDA Device: {}".format(torch.cuda.get_device_name(self.gpu_index)))
            self.device = "cuda:{}".format(self.gpu_index)

        model   =   Autoencoder(
                        layers=self.layers,
                        h_activation=self.h_activation,
                        o_activation=self.o_activation,
                        device=self.device
                    )

        if self.cont:
            print("Loading model from {}".format(self.model_file))
            model = torch.load(self.model_file)
            model.load_state_dict(state['state_dict'])
            model.optimizer = state['optimizer']

        if self.error_type == "mse":
            loss_fn = nn.MSELoss()
        else:
            raise Exception("Invalid error_type {}".format(self.error_type))

        if self.optimizer_type == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate) 
        else:
            raise Exception("Invalid optimizer_type {}".format(self.optimizer_type))

        # Read training file
        data = pd.DataFrame()
        for i, chunk in enumerate(pd.read_csv(self.training_file, header=None, chunksize=self.chunk_size)):
            data = data.append(chunk)

        print("Storing data to tensor...")
        x = torch.tensor(data.values).float().to(self.device)

        train_ds = AutoencoderDataset(
            x=x
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False
        )

        for epoch in range(self.epochs):
            print("Epoch: {}".format(epoch))
            self.train_fn(train_loader, model, optimizer, loss_fn)

            print("Saving model to {}...".format(self.model_file))

            state = {
                'params': {
                    'o_activation': self.o_activation,
                    'h_activation': self.h_activation,
                    'layers':       self.layers,
                    'device':       self.device
                },
                'state_dict':       model.state_dict(),
                'optimizer':        optimizer.state_dict()
            }

            torch.save(state, self.model_file)
    
    def train_fn(self, loader, model, optimizer, loss_fn):
        loop = tqdm(loader)

        for batch_idx, (data, targets) in enumerate(loop):
            data    = data.to(device=self.device)
            targets = targets.to(device=self.device)

            # Forward
            predictions = model.forward(data)

            loss = loss_fn(predictions, targets)

            # Backward
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            # update tqdm
            loop.set_postfix(loss=loss.item())
