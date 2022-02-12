import sys
import os
import pandas as pd
import numpy as np
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib.autoencoder import Autoencoder

class Compress:
    def __init__(self, params={}):
        self.params = params

        self.model_file     = params.get('model_file')
        self.output_file    = params.get('output_file')
        self.data_file      = params.get('data_file')
        self.chunk_size     = params.get('chunk_size')  or 100

    def execute(self):
        print("Loading model from {}...".format(self.model_file))
        state   = torch.load(self.model_file)
        params  = state['params']

        self.layers         = params.get('layers')          or [10, 6]
        self.h_activation   = params.get('h_activation')    or 'relu'
        self.o_activation   = params.get('o_activation')    or 'sigmoid'
        self.device         = params.get('device')          or 'cpu' 

        model = Autoencoder(
            layers=self.layers,
            h_activation=self.h_activation,
            o_activation=self.o_activation,
            device=self.device
        )

        model.load_state_dict(state['state_dict'])
        model.optimizer = state['optimizer']

        print("Reading data from file {}...".format(self.data_file))
        data = pd.DataFrame()
        for i, chunk in enumerate(pd.read_csv(self.data_file, header=None, chunksize=self.chunk_size)):
            data = data.append(chunk)


        print("Storing data to tensor...")
        x = torch.tensor(data.values).float().to(self.device)

        print("Compressing...")
        x_hat = model.encode(x)
        x_hat = x_hat.detach().cpu().numpy().astype(np.float32)

        print("Writing to file {}...".format(self.output_file))
        df_x = pd.DataFrame(x_hat, columns=None)
        df_x.to_csv(self.output_file, header=None)

        print("Done.")
