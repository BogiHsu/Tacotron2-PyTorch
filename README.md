# Tacotron2-PyTorch
Yet another PyTorch implementation of [Natural TTS Synthesis By Conditioning
Wavenet On Mel Spectrogram Predictions](https://arxiv.org/pdf/1712.05884.pdf). The project is highly based on [these](#References). I made some modification to improve speed and performance of both training and inference.

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
Currently only support LJSpeech dataset. No need to do preprocessing if you use the dataset with 22050 sample rate.

For traing with different sample rate, you should deal with the audio files yourself and modified `hparams.py`.

## Training
1. For training Tacotron2, run the following command.

```bash
python3 train.py --data_dir=<dir/to/dataset> --log_dir-root=<dir/to/logs> --ckpt_dir=<dir/to/models>
```

2. For training using a pretrained model, run the following command.

```bash
python3 train.py --data_dir=<dir/to/dataset> --log_dir-root=<dir/to/logs> --ckpt_dir=<dir/to/models> --ckpt_pth=<pth/to/pretrained/model>
```

3. For using Tensorboard (optional), run the following command.

```bash
python3 train.py --data_dir=<dir/to/dataset> --log_dir-root=<dir/to/logs> --ckpt_dir=<dir/to/models> --ckpt_pth=<pth/to/pretrained/model>
```

You can find alinment images and synthesized audio clips during training. Recording freqency and text to synthesize can be set in `hparams.py`.

## Inference
Work in progress.

## Vocoder
Vocoder is not implemented yet. For now I just reconstuct the linear spectrogram from the mel-spectrogram directly and use Griffim-Lim to synthesize the waveform. A neural vocoder will be implemented in the future. Or you can refer to [Wavenet](https://github.com/r9y9/wavenet_vocoder), [FFTNet](https://github.com/syang1993/FFTNet), or [WaveGlow](https://github.com/NVIDIA/waveglow).

## TODO
- [x] LJSpeech dataloader.
- [x] Reduce factor.
- [x] Implement codes for inference.
- [ ] Add some samples and training detail.
- [ ] Add pretrained vocoder.

## References
This project is highly based on the works below.
- [Tacotron2 implement by NVIDIA](https://github.com/NVIDIA/tacotron2)
- [Tacotron implement by r9y9](https://github.com/r9y9/tacotron_pytorch)
- [Tacotron implement by keithito](https://github.com/keithito/tacotron)
