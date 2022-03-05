import os
import torch
import argparse
import numpy as np
import matplotlib.pylab as plt
from text import text_to_sequence
from model.model import Tacotron2
from hparams import hparams as hps
from utils.util import mode, to_var, to_arr
from utils.audio import load_wav, save_wav, melspectrogram


def files_to_list(fdir = 'data'):
    f_list = []
    with open(os.path.join(fdir, 'metadata.csv'), encoding = 'utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            wav_path = os.path.join(fdir, 'wavs', '%s.wav' % parts[0])
            f_list.append([wav_path, parts[1]])
    return f_list


def load_model(ckpt_pth):
    ckpt_dict = torch.load(ckpt_pth)
    model = Tacotron2()
    model.load_state_dict(ckpt_dict['model'])
    model = mode(model, True).eval()
    model.decoder.train()
    model.postnet.train()
    return model


def infer(wav_path, text, model):
    sequence = text_to_sequence(text, hps.text_cleaners)
    sequence = to_var(torch.IntTensor(sequence)[None, :]).long()
    mel = melspectrogram(load_wav(wav_path))
    mel_in = to_var(torch.Tensor([mel]))
    r = mel_in.shape[2]%hps.n_frames_per_step
    if r != 0:
        mel_in = mel_in[:, :, :-r]
    sequence = torch.cat([sequence, sequence], 0)
    mel_in = torch.cat([mel_in, mel_in], 0)
    _, mel_outputs_postnet, _, _ = model.teacher_infer(sequence, mel_in)
    ret = mel
    if r != 0:
        ret[:, :-r] = to_arr(mel_outputs_postnet[0])
    else:
        ret = to_arr(mel_outputs_postnet[0])
    return ret


def save_mel(res, pth, name):
    out = os.path.join(pth, name)
    np.save(out, res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--ckpt_pth', type = str, default = '',
                        required = True, help = 'path to load checkpoints')
    parser.add_argument('-n', '--npy_pth', type = str, default = 'dump',
                        help = 'path to save mels')

    args = parser.parse_args()
    
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    model = load_model(args.ckpt_pth)
    flist = files_to_list()
    for x in flist:
        ret = infer(x[0], x[1], model)
        name = x[0].split('/')[-1].split('.wav')[0]
        if args.npy_pth != '':
            save_mel(ret, args.npy_pth, name)
