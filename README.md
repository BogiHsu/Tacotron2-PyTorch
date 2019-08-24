# Tacotron2-PyTorch
Yet another PyTorch implementation of [Natural TTS Synthesis By Conditioning
Wavenet On Mel Spectrogram Predictions](https://arxiv.org/pdf/1712.05884.pdf). The project is highly based on [these](#References). I made some modification to improve speed and performance of both training and inference.

## Update
Pretrained models are available now!

## TODO
- [x] LJSpeech dataloader.
- [x] Reduction factor.
- [x] Implement codes for inference.
- [x] Add some temporary samples.
- [X] Add some samples and a pertrained model.
- [ ] Add a pretrained vocoder.

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
Currently only support [LJSpeech dataset](https://keithito.com/LJ-Speech-Dataset/). No need to do preprocessing if you use the dataset with 22050 sample rate.

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
- For synthesizing wav files, run the following command.

```bash
python3 inference.py --ckpt_pth=<pth/to/model> --img_pth=<pth/to/save/alignment> --wav_pth=<pth/to/save/wavs> --text=<text/to/synthesize>
```

## Pretrained Model
Now you can download pretrained models from [here](https://www.dropbox.com/sh/vk2erozpkoltao6/AABCk4WryQtrt4BYthIKzbK7a?dl=0). The hyperparameter(git commit: [301943f](https://github.com/BogiHsu/Tacotron2-PyTorch/commit/301943f4c9d1de7d6c819be08ebd401a059127c3)) for training the pretrained models is also in the directory.

## Vocoder
Vocoder is not implemented yet. For now I just reconstuct the linear spectrogram from the mel spectrogram directly and use Griffim-Lim to synthesize the waveform. A neural vocoder will be implemented in the future. Or you can refer to [Wavenet](https://github.com/r9y9/wavenet_vocoder), [FFTNet](https://github.com/syang1993/FFTNet), or [WaveGlow](https://github.com/NVIDIA/waveglow).

## Results
You can find TEMPORARY samples [here](https://github.com/BogiHsu/Tacotron2-PyTorch/tree/master/results). Please note that the training has not done yet and for now the reconstruction [process](#Vocoder) still causes a lot of distortion (a vocoder implementation will be a future work), so these are NOT the best results of this work.

But the alignment of the attention is pretty well now (about 100k training steps), the following figure is one sample.

<img src="https://github.com/BogiHsu/Tacotron2-PyTorch/blob/master/results/tmp.png">

This figure shows the mel spectrogram from the decoder without the postnet, the mel spectrgram with the postnet, and the alignment of the attention.

## References
This project is highly based on the works below.
- [Tacotron2 implement by NVIDIA](https://github.com/NVIDIA/tacotron2)
- [Tacotron implement by r9y9](https://github.com/r9y9/tacotron_pytorch)
- [Tacotron implement by keithito](https://github.com/keithito/tacotron)
