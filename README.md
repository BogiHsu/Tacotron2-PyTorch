# Tacotron2-PyTorch
Yet another PyTorch implementation of [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](https://arxiv.org/pdf/1712.05884.pdf). The project is highly based on [these](#References). I made some modification to improve speed and performance of both training and inference.

## TODO
- [x] Add Colab demo.
- [x] Update README.
- [x] Upload pretrained models.
- [x] Compatible with [WaveGlow](https://github.com/NVIDIA/waveglow) and [Hifi-GAN](https://github.com/jik876/hifi-gan).

## Requirements
- Python >= 3.5.2
- torch >= 1.0.0
- numpy
- scipy
- pillow
- inflect
- librosa
- Unidecode
- matplotlib
- tensorboardX

## Preprocessing
Currently only support [LJ Speech](https://keithito.com/LJ-Speech-Dataset/). You can modify `hparams.py` for different sampling rates. `prep` decides whether to preprocess all utterances before training or online preprocess. `pth` sepecifies the path to store preprocessed data.

## Training
1. For training Tacotron2, run the following command.
```bash
python3 train.py \
    --data_dir=<dir/to/dataset> \
    --ckpt_dir=<dir/to/models>
```

2. If you have multiple GPUs, try [distributed.launch](https://pytorch.org/docs/stable/distributed.html#launch-utility).
```bash
python -m torch.distributed.launch --nproc_per_node <NUM_GPUS> train.py \
    --data_dir=<dir/to/dataset> \
    --ckpt_dir=<dir/to/models>
```
Note that the training batch size will become <NUM_GPUS> times larger.

3. For training using a pretrained model, run the following command.
```bash
python3 train.py \
    --data_dir=<dir/to/dataset> \
    --ckpt_dir=<dir/to/models> \
    --ckpt_pth=<pth/to/pretrained/model>
```

4. For using Tensorboard (optional), run the following command.
```bash
python3 train.py \
    --data_dir=<dir/to/dataset> \
    --ckpt_dir=<dir/to/models> \
    --log_dir=<dir/to/logs>
```
You can find alinment images and synthesized audio clips during training. The text to synthesize can be set in `hparams.py`.

## Inference
- For synthesizing wav files, run the following command.

```bash
python3 inference.py \
    --ckpt_pth=<pth/to/model> \
    --img_pth=<pth/to/save/alignment> \
    --npy_pth=<pth/to/save/mel> \
    --wav_pth=<pth/to/save/wav> \
    --text=<text/to/synthesize>
```

## Pretrained Model
You can download pretrained models from [Realeases](https://github.com/BogiHsu/Tacotron2-PyTorch/releases). The hyperparameter for training is also in the directory. All the models were trained using 8 GPUs.

## Vocoder
A vocoder is not implemented. But the model is compatible with [WaveGlow](https://github.com/NVIDIA/waveglow) and [Hifi-GAN](https://github.com/jik876/hifi-gan). Check the Colab demo for more information. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BogiHsu/Tacotron2-PyTorch/blob/master/inference.ipynb)

## References
This project is highly based on the works below.
- [Tacotron2 by NVIDIA](https://github.com/NVIDIA/tacotron2)
- [Tacotron by r9y9](https://github.com/r9y9/tacotron_pytorch)
- [Tacotron by keithito](https://github.com/keithito/tacotron)
