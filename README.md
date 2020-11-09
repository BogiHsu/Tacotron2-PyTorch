# Tacotron2-PyTorch
Yet another PyTorch implementation of [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](https://arxiv.org/pdf/1712.05884.pdf). The project is highly based on [these](#References). I made some modification to improve speed and performance of both training and inference.

## TODO
- [ ] Combining with [WG-WaveNet](https://arxiv.org/abs/2005.07412).

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
Currently only support [LJSpeech dataset](https://keithito.com/LJ-Speech-Dataset/). You can modify `hparams.py` for different sampling rates. `prep` decides whether to preprocess all utterances before training or online preprocess. `pth` sepecifies the path to store preprocessed data.

## Training
1. For training Tacotron2, run the following command.

```bash
python3 train.py --data_dir=<dir/to/dataset> --ckpt_dir=<dir/to/models>
```

2. For training using a pretrained model, run the following command.

```bash
python3 train.py --data_dir=<dir/to/dataset> --ckpt_dir=<dir/to/models> --ckpt_pth=<pth/to/pretrained/model>
```

3. For using Tensorboard (optional), run the following command.

```bash
python3 train.py --data_dir=<dir/to/dataset> --ckpt_dir=<dir/to/models> --log_dir=<dir/to/logs>
```

You can find alinment images and synthesized audio clips during training. Recording freqency and text to synthesize can be set in `hparams.py`.

## Inference
- For synthesizing wav files, run the following command.

```bash
python3 inference.py --ckpt_pth=<pth/to/model> --img_pth=<pth/to/save/alignment> --wav_pth=<pth/to/save/wavs> --text=<text/to/synthesize>
```

## Pretrained Model
You can download pretrained models from [here](https://www.dropbox.com/sh/vk2erozpkoltao6/AABCk4WryQtrt4BYthIKzbK7a?dl=0) (git commit: [9e7c26d](https://github.com/BogiHsu/Tacotron2-PyTorch/commit/9e7c26d93ea9d93332b1c316ac85c58771197d4f)). The hyperparameter for training is also in the directory.

## Vocoder
A vocoder is not implemented yet. For now it just reconstucts the linear spectrogram from the Mel spectrogram directly and uses Griffim-Lim to synthesize the waveform. A pipeline for [WG-WaveNet](https://arxiv.org/abs/2005.07412) is in progress. Or you can refer to [WaveNet](https://github.com/r9y9/wavenet_vocoder), [FFTNet](https://github.com/syang1993/FFTNet), or [WaveGlow](https://github.com/NVIDIA/waveglow).

## Results
You can find some samples in [results](https://github.com/BogiHsu/Tacotron2-PyTorch/tree/master/results). These results are generated using either pseudo inverse or WaveNet.

The alignment of the attention is pretty well now (about 100k training steps), the following figure is one sample.

<img src="https://github.com/BogiHsu/Tacotron2-PyTorch/blob/master/results/tmp.png">

This figure shows the Mel spectrogram from the decoder without the postnet, the Mel spectrgram with the postnet, and the alignment of the attention.

## References
This project is highly based on the works below.
- [Tacotron2 by NVIDIA](https://github.com/NVIDIA/tacotron2)
- [Tacotron by r9y9](https://github.com/r9y9/tacotron_pytorch)
- [Tacotron by keithito](https://github.com/keithito/tacotron)
