import sys
import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib.cnn_autoencoder import CnnAutoencoder
from lib.cnn_autoencoder_dataset import CnnAutoencoderDataset

class CompressCnnAutoencoder:
    def __init__(self, params={}):
        self.params = params

        self.model_file     = params.get('model_file')
        self.output_file    = params.get('output_file')
        self.chunk_size     = params.get('chunk_size') or 100
        self.img_dir        = params.get('img_dir')
        self.normalize      = params.get('normalize')

    def execute(self):
        print("Loading model from {}...".format(self.model_file))
        state   = torch.load(self.model_file)
        params  = state['params']

        self.scale          = params.get('scale')
        self.channel_maps   = params.get('channel_maps')
        self.padding        = params.get('padding')
        self.kernel_size    = params.get('kernel_size')
        self.num_channels   = params.get('num_channels')
        self.img_width      = params.get('img_width')
        self.img_height     = params.get('img_height')
        self.device         = params.get('device')

        model   =   CnnAutoencoder(
                        scale=self.scale,
                        channel_maps=self.channel_maps,
                        padding=self.padding,
                        kernel_size=self.kernel_size,
                        num_channels=self.num_channels,
                        img_width=self.img_width,
                        img_height=self.img_height
                    ).to(self.device)

        model.load_state_dict(state['state_dict'])
        model.optimizer = state['optimizer']

        # Compress data
        print("Compressing data...")

        ds_compress = CnnAutoencoderDataset(
            img_dir=self.img_dir,
            img_width=self.img_width,
            img_height=self.img_height
        )

        loader = DataLoader(
            ds_compress,
            batch_size=1,
            shuffle=False,
            drop_last=False
        )

        raw_data = []

        for batch_idx, (data, targets) in enumerate(loader):
            data = data.float().to(device=self.device)
            compressed_data = model.compress(data)

            for d in  compressed_data:
                line = d.detach().cpu().numpy().astype(np.float32)
                raw_data.append(line)

        # Build the data frame
        columns = []
        for i in range(len(raw_data[0])):
            columns.append("x{}".format(i))

        df_x = pd.DataFrame(raw_data, columns=columns)

        print("Latent Size: {}".format(len(columns)))

        # Normalize
        if self.normalize:
            print("Normalizing data...")
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(df_x.values)

            df_x = pd.DataFrame(x_scaled, columns=columns)

        print("Writing to file {}...".format(self.output_file))
        df_x.to_csv(self.output_file, index=False)
