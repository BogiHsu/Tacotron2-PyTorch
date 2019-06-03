import torch
import numpy as np
from audio import save_wav
import matplotlib.pylab as plt
from text import text_to_sequence
from model.model import Tacotron2
from hparams import hparams as hps
from utils import mode, to_var, to_arr


def plot_data(data, figsize = (16, 4)):
	fig, axes = plt.subplots(1, len(data), figsize = figsize)
	for i in range(len(data)):
		axes[i].imshow(data[i], aspect = 'auto', origin = 'bottom')


def load_model(ckpt_pth):
	ckpt_dict = torch.load(ckpt_pth)
	model = Tacotron2()
	model.load_state_dict(ckpt_dict['model'])
	return model


ckpt_pth = 'ckpt/ckpt_16000'
ckpt_dict = torch.load(ckpt_pth)
model = Tacotron2()
model.load_state_dict(ckpt_dict['model'])
mode(model, True).eval()


text = 'Waveglow is really awesome!'
sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
sequence = to_var(torch.Tensor(sequence)).long()


mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
plot_data((to_arr(mel_outputs[0]),
			to_arr(mel_outputs_postnet[0]),
			to_arr(alignments[0]).T))
plt.savefig('infer.png')