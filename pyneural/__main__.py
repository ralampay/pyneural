import sys
import argparse
import os
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from modules.train_autoencoder import TrainAutoencoder
from modules.compress import Compress

mode_choices = [
    "train-autoencoder",
    "compress"
]

def main():
    parser = argparse.ArgumentParser(description="PyNeural: Python neural network implementation")
    parser.add_argument("--mode", help="Mode to be used", choices=mode_choices, type=str, required=True)
    parser.add_argument("--layers", help="Layers for neural network", type=int, nargs='+')
    parser.add_argument("--h-activation", help="Hidden activation", type=str, default="relu")
    parser.add_argument("--o-activation", help="Output activation", type=str, default="sigmoid")
    parser.add_argument("--error-type", help="Error type", type=str, default="mse")
    parser.add_argument("--optimizer-type", help="Optimizer type", type=str, default="adam")
    parser.add_argument("--gpu-index", help="GPU index", type=int, default=0)
    parser.add_argument("--learning-rate", help="Learning rate", type=float, default=0.001)
    parser.add_argument("--chunk-size", help="Chunk size", type=int, default=100)
    parser.add_argument("--batch-size", help="Batch size", type=int, default=50)
    parser.add_argument("--cont", help="Continue training", type=bool, default=False)
    parser.add_argument("--epochs", help="Epoch count", type=int, default=100)
    parser.add_argument("--model-file", help="Model file for saving", type=str)
    parser.add_argument("--training-file", help="Training file", type=str)
    parser.add_argument("--device", help="Device", type=str, default="cpu")
    parser.add_argument("--data-file", help="Data file", type=str)
    parser.add_argument("--output-file", help="Output file", type=str)

    args            = parser.parse_args()
    mode            = args.mode
    layers          = args.layers
    h_activation    = args.h_activation
    o_activation    = args.o_activation
    error_type      = args.error_type
    optimizer_type  = args.optimizer_type
    gpu_index       = args.gpu_index
    learning_rate   = args.learning_rate
    chunk_size      = args.chunk_size
    batch_size      = args.batch_size
    cont            = args.cont
    epochs          = args.epochs
    model_file      = args.model_file
    training_file   = args.training_file
    device          = args.device
    data_file       = args.data_file
    output_file     = args.output_file

    if mode == "train-autoencoder":
        params = {
            'layers':           layers,
            'h_activation':     h_activation,
            'o_activation':     o_activation,
            'device':           device,
            'error_type':       error_type,
            'optimizer_type':   optimizer_type,
            'gpu_index':        gpu_index,
            'epochs':           epochs,
            'learning_rate':    learning_rate,
            'chunk_size':       chunk_size,
            'batch_size':       batch_size,
            'cont':             cont,
            'model_file':       model_file,
            'training_file':    training_file
        }

        cmd = TrainAutoencoder(params=params)
        cmd.execute()

    elif mode == "compress":
        params = {
            'model_file':   model_file,
            'output_file':  output_file,
            'data_file':    data_file,
            'chunk_size':   chunk_size
        }

        cmd = Compress(params=params)
        cmd.execute()

if __name__ == '__main__':
    main()
