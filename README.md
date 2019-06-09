# Tacotron2-PyTorch
Yet another implementation of Tacotron2. The codes are executable but still work in progress.

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

## TODO
- [x] LJSpeech dataloader.
- [x] Reduce factor.
- [ ] Implement codes for inference.
- [ ] Add some samples and training detail.
- [ ] Add pretrained vocoder.

## Reference
This project is highly based on the works below.
- [Tacotron2 implement by NVIDIA](https://github.com/NVIDIA/tacotron2)
- [Tacotron implement by r9y9](https://github.com/r9y9/tacotron_pytorch)
- [Tacotron implement by keithito](https://github.com/keithito/tacotron)
