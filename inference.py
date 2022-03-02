import torch
import argparse
import numpy as np
import matplotlib.pylab as plt
from text import text_to_sequence
from model.model import Tacotron2
from hparams import hparams as hps
from utils.util import mode, to_arr
from utils.audio import save_wav, inv_melspectrogram


def load_model(ckpt_pth):
    ckpt_dict = torch.load(ckpt_pth)
    model = Tacotron2()
    model.load_state_dict(ckpt_dict['model'])
    model = mode(model, True).eval()
    return model


def infer(text, model):
    sequence = text_to_sequence(text, hps.text_cleaners)
    sequence = mode(torch.IntTensor(sequence)[None, :]).long()
    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
    return (mel_outputs, mel_outputs_postnet, alignments)


def plot_data(data, figsize = (16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize = figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect = 'auto', origin = 'bottom')


def plot(output, pth):
    mel_outputs, mel_outputs_postnet, alignments = output
    plot_data((to_arr(mel_outputs[0]),
                to_arr(mel_outputs_postnet[0]),
                to_arr(alignments[0]).T))
    plt.savefig(pth+'.png')


def audio(output, pth):
    mel_outputs, mel_outputs_postnet, _ = output
    wav_postnet = inv_melspectrogram(to_arr(mel_outputs_postnet[0]))
    save_wav(wav_postnet, pth+'.wav')


def save_mel(output, pth):
    mel_outputs, mel_outputs_postnet, _ = output
    np.save(pth+'.npy', to_arr(mel_outputs_postnet))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--ckpt_pth', type = str, default = '',
                        required = True, help = 'path to load checkpoints')
    parser.add_argument('-i', '--img_pth', type = str, default = '',
                        help = 'path to save images')
    parser.add_argument('-w', '--wav_pth', type = str, default = '',
                        help = 'path to save wavs')
    parser.add_argument('-n', '--npy_pth', type = str, default = '',
                        help = 'path to save mels')
    parser.add_argument('-t', '--text', type = str, default = 'Tacotron is awesome.',
                        help = 'text to synthesize')

    args = parser.parse_args()
    
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    model = load_model(args.ckpt_pth)
    output = infer(args.text, model)
    if args.img_pth != '':
        plot(output, args.img_pth)
    if args.wav_pth != '':
        audio(output, args.wav_pth)
    if args.npy_pth != '':
        save_mel(output, args.npy_pth)
