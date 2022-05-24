import sys
import argparse
import os
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from modules.train_autoencoder import TrainAutoencoder
from modules.train_cnn_autoencoder import TrainCnnAutoencoder
from modules.compress_autoencoder import CompressAutoencoder
from modules.compress_cnn_autoencoder import CompressCnnAutoencoder
from modules.anomaly_data_partitioner import AnomalyDataPartitioner
from modules.extract_frames import ExtractFrames

mode_choices = [
    "train-autoencoder",
    "train-cnn-autoencoder",
    "compress-autoencoder",
    "compress-cnn-autoencoder",
    "anomaly-data-partitioner",
    "extract-frames"
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
    parser.add_argument("--channel-maps", help="Channel maps", type=int, nargs='+')
    parser.add_argument("--padding", help="Padding", type=int, default=1)
    parser.add_argument("--scale", help="Scale", type=int, default=2)
    parser.add_argument("--img-height", help="IMG Height", type=int, default=100)
    parser.add_argument("--img-width", help="IMG Width", type=int, default=100)
    parser.add_argument("--train-img-dir", help="Train Image Directory", type=str)
    parser.add_argument("--test-img-dir", help="Test Image Directory", type=str)
    parser.add_argument("--num-channels", help="Num channels", type=int, default=3)
    parser.add_argument("--kernel-size", help="CNN kernel size", type=int, default=3)
    parser.add_argument("--normalize", help="Normalize compression", type=bool, default=True)
    parser.add_argument("--contamination-ratio", help="Contamination Ratio", type=float, default=0.05)
    parser.add_argument("--normal-count", help="Normal Count", type=int, default=1000)
    parser.add_argument("--label-column", help="Label Column", type=str, default='y')
    parser.add_argument("--label-anomaly", help="Label Anomaly", type=int, default=-1)
    parser.add_argument("--label-normal", help="Label Normal", type=int, default=1)
    parser.add_argument("--output-train-file", help="Output Train File", type=str, default='output.csv')
    parser.add_argument("--output-validation-file", help="Output Validation File", type=str, default='validation.csv')
    parser.add_argument("--output-img-dir", help="Output image directory", type=str, required=False)
    parser.add_argument("--video-file", help="Video file", type=str, required=False)

    args                    = parser.parse_args()
    mode                    = args.mode
    layers                  = args.layers
    h_activation            = args.h_activation
    o_activation            = args.o_activation
    error_type              = args.error_type
    optimizer_type          = args.optimizer_type
    gpu_index               = args.gpu_index
    learning_rate           = args.learning_rate
    chunk_size              = args.chunk_size
    batch_size              = args.batch_size
    cont                    = args.cont
    epochs                  = args.epochs
    model_file              = args.model_file
    training_file           = args.training_file
    device                  = args.device
    data_file               = args.data_file
    output_file             = args.output_file
    channel_maps            = args.channel_maps
    padding                 = args.padding
    scale                   = args.scale
    img_height              = args.img_height
    img_width               = args.img_width
    train_img_dir           = args.train_img_dir
    test_img_dir            = args.test_img_dir
    num_channels            = args.num_channels
    kernel_size             = args.kernel_size
    normalize               = args.normalize
    contamination_ratio     = args.contamination_ratio
    normal_count            = args.normal_count
    label_column            = args.label_column
    label_anomaly           = args.label_anomaly
    label_normal            = args.label_normal
    output_train_file       = args.output_train_file
    output_validation_file  = args.output_validation_file
    output_img_dir          = args.output_img_dir
    video_file              = args.video_file

    if mode == "train-cnn-autoencoder":
        params = {
            'gpu_index':        gpu_index,
            'epochs':           epochs,
            'learning_rate':    learning_rate,
            'chunk_size':       chunk_size,
            'batch_size':       batch_size,
            'cont':             cont,
            'model_file':       model_file,
            'channel_maps':     channel_maps,
            'padding':          padding,
            'scale':            scale,
            'img_height':       img_height,
            'img_width':        img_width,
            'train_img_dir':    train_img_dir,
            'kernel_size':      kernel_size,
            'device':           device,
            'normalize':        normalize
        }

        cmd = TrainCnnAutoencoder(params=params)
        cmd.execute()

    elif mode == "train-autoencoder":
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

    elif mode == "compress-autoencoder":
        params = {
            'model_file':   model_file,
            'output_file':  output_file,
            'data_file':    data_file,
            'chunk_size':   chunk_size
        }

        cmd = CompressAutoencoder(params=params)
        cmd.execute()
    
    elif mode == "compress-cnn-autoencoder":
        params = {
            'model_file':   model_file,
            'output_file':  output_file,
            'chunk_size':   chunk_size,
            'img_dir':      test_img_dir,
            'normalize':    normalize
        }

        cmd = CompressCnnAutoencoder(params=params)
        cmd.execute()

    elif mode == "anomaly-data-partitioner":
        params = {
            'data_file':                data_file,
            'normal_count':             normal_count,
            'contamination_ratio':      contamination_ratio,
            'normal_count':             normal_count,
            'label_column':             label_column,
            'label_anomaly':            label_anomaly,
            'label_normal':             label_normal,
            'output_train_file':        output_train_file,
            'output_validation_file':   output_validation_file
        }

        cmd = AnomalyDataPartitioner(params=params)
        cmd.execute()

    elif mode == "extract-frames":
        params = {
            'video_file':       video_file,
            'output_img_dir':   output_img_dir
        }

        cmd = ExtractFrames(params=params)
        cmd.execute()

if __name__ == '__main__':
    main()
