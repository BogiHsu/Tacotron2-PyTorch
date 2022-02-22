# Tacotron2-PyTorch
Yet another PyTorch implementation of [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](https://arxiv.org/pdf/1712.05884.pdf). The project is highly based on [these](#References). I made some modification to improve speed and performance of both training and inference.

## TODO
- [ ] Add Colab demo.
- [ ] Update README.
- [ ] Upload pretrained models.
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

## Vocoder
A vocoder is not implemented. But the model is compatible with [WaveGlow](https://github.com/NVIDIA/waveglow) and [Hifi-GAN](https://github.com/jik876/hifi-gan). Please refer to these great repositories.

## References
This project is highly based on the works below.
- [Tacotron2 by NVIDIA](https://github.com/NVIDIA/tacotron2)
- [Tacotron by r9y9](https://github.com/r9y9/tacotron_pytorch)
- [Tacotron by keithito](https://github.com/keithito/tacotron)
