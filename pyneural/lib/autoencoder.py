import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class Autoencoder(nn.Module):
    def __init__(self, layers=[], h_activation="relu", o_activation="sigmoid", device=torch.device("cpu")):
        super().__init__()

        self.device = device

        self.h_activation = h_activation
        self.o_activation = o_activation

        self.encoding_layers = nn.ModuleList([])
        self.decoding_layers = nn.ModuleList([])

        self.layers = layers
        
        reversed_layers = list(reversed(layers))

        for i in range(len(layers) - 1):
            self.encoding_layers.append(nn.Linear(layers[i], layers[i+1]))
            self.decoding_layers.append(nn.Linear(reversed_layers[i], reversed_layers[i+1]))

        # Initialize model to device
        self.to(self.device)

    def encode(self, x):
        for i in range(len(self.encoding_layers)):
            if self.h_activation == "relu":
                x = F.relu(self.encoding_layers[i](x))
            else:
                raise Exception("Invalid hidden activation {}".format(self.h_activation))

        return x

    def decode(self, x):
        for i in range(len(self.decoding_layers)):
            if i != len(self.decoding_layers) - 1:
                if self.h_activation == "relu":
                    x = F.relu(self.decoding_layers[i](x))
                else:
                    raise Exception("Invalid hidden activation {}".format(self.h_activation))
            else:
                if self.o_activation == "sigmoid":
                    x = torch.sigmoid(self.decoding_layers[i](x))
                else:
                    raise Exception("Invalid output activation {}".format(self.o_activation))

        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)

        return x

    def load(self, filename):
        state = torch.load(filename)

        self.load_state_dict(state['state_dict'])

        self.optimizer = state['optimizer']

        params = state.get('params')

        self.o_activation   = params['o_activation']
        self.h_activation   = params['h_activation']
        self.layers         = params['layers']
        self.device         = params['device']
