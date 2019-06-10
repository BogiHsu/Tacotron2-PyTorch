import os
import torch
import random
import numpy as np
from text import text_to_sequence
from hparams import hparams as hps
from torch.utils.data import Dataset
from utils.audio import load_wav, melspectrogram
random.seed(0)


def files_to_list(fdir):
	f_list = []
	with open(os.path.join(fdir, 'metadata.csv'), encoding = 'utf-8') as f:
		for line in f:
			parts = line.strip().split('|')
			wav_path = os.path.join(fdir, 'wavs', '%s.wav' % parts[0])
			f_list.append([wav_path, parts[1]])
	return f_list


class ljdataset(Dataset):
	def __init__(self, fdir):
		self.f_list = files_to_list(fdir)
		random.shuffle(self.f_list)

	def get_mel_text_pair(self, filename_and_text):
		filename, text = filename_and_text[0], filename_and_text[1]
		text = self.get_text(text)
		mel = self.get_mel(filename)
		return (text, mel)

	def get_mel(self, filename):
		wav = load_wav(filename)
		mel = melspectrogram(wav).astype(np.float32)
		return torch.Tensor(mel)

	def get_text(self, text):
		text_norm = torch.IntTensor(text_to_sequence(text, hps.text_cleaners))
		return text_norm

	def __getitem__(self, index):
		return self.get_mel_text_pair(self.f_list[index])

	def __len__(self):
		return len(self.f_list)


class ljcollate():
	def __init__(self, n_frames_per_step):
		self.n_frames_per_step = n_frames_per_step

	def __call__(self, batch):
		# Right zero-pad all one-hot text sequences to max input length
		input_lengths, ids_sorted_decreasing = torch.sort(
			torch.LongTensor([len(x[0]) for x in batch]),
			dim=0, descending=True)
		max_input_len = input_lengths[0]

		text_padded = torch.LongTensor(len(batch), max_input_len)
		text_padded.zero_()
		for i in range(len(ids_sorted_decreasing)):
			text = batch[ids_sorted_decreasing[i]][0]
			text_padded[i, :text.size(0)] = text

		# Right zero-pad mel-spec
		num_mels = batch[0][1].size(0)
		max_target_len = max([x[1].size(1) for x in batch])
		if max_target_len % self.n_frames_per_step != 0:
			max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
			assert max_target_len % self.n_frames_per_step == 0

		# include mel padded and gate padded
		mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
		mel_padded.zero_()
		gate_padded = torch.FloatTensor(len(batch), max_target_len)
		gate_padded.zero_()
		output_lengths = torch.LongTensor(len(batch))
		for i in range(len(ids_sorted_decreasing)):
			mel = batch[ids_sorted_decreasing[i]][1]
			mel_padded[i, :, :mel.size(1)] = mel
			gate_padded[i, mel.size(1)-1:] = 1
			output_lengths[i] = mel.size(1)

		return text_padded, input_lengths, mel_padded, gate_padded, output_lengths
