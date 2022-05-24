# PyNeural

Neural Network tooling written in Python w/ Pytorch

## Modes

### Train CNN Autoencoder

Train a vanilla CNN Autoencoder and save the model.

```
python -m pyneural --model train-cnn-autoencoder \
--gpu-index 0 \
--epochs 100 \
--learning-rate 0.0001 \
--chunk-size 1 \
--batch-size 1 \
--cont False \
--model-file output-model.pth \
--channel-maps 3 16 8 4 \
--padding 1 \
--scale 2 \
--img-height 128 \
--img-width 128 \
--train-img-dir frames \
--kernel-size 3 \
--device cuda \
```

### Video Frame Extraction

Extract frames from a video.

```
python -m pyneural --mode extract-frames --video-file video.mp4 --output-img-dir output
```
