import numpy as np
from utils.util import to_arr
from hparams import hparams as hps
from tensorboardX import SummaryWriter
from utils.audio import inv_melspectrogram
from utils.plot import plot_alignment_to_numpy, plot_spectrogram_to_numpy


class Tacotron2Logger(SummaryWriter):
	def __init__(self, logdir):
		super(Tacotron2Logger, self).__init__(logdir)

	def log_training(self, reduced_loss, grad_norm, learning_rate, iteration):
			self.add_scalar("training.loss", reduced_loss, iteration)
			self.add_scalar("grad.norm", grad_norm, iteration)
			self.add_scalar("learning.rate", learning_rate, iteration)

	def sample_training(self, output, iteration):
			mel_outputs = to_arr(output[0][0])
			mel_outputs_postnet = to_arr(output[1][0])
			alignments = to_arr(output[2][0]).T
			
			# plot alignment, mel and postnet output
			self.add_image(
				"alignment",
				plot_alignment_to_numpy(alignments),
				iteration)
			self.add_image(
				"mel_outputs",
				plot_spectrogram_to_numpy(mel_outputs),
				iteration)
			self.add_image(
				"mel_outputs_postnet",
				plot_spectrogram_to_numpy(mel_outputs_postnet),
				iteration)
			
			# save audio
			try: # sometimes error
				wav = inv_melspectrogram(mel_outputs)
				wav /= max(0.01, np.max(np.abs(wav)))
				wav_postnet = inv_melspectrogram(mel_outputs_postnet)
				wav_postnet /= max(0.01, np.max(np.abs(wav_postnet)))
				self.add_audio('pred', wav, iteration, hps.sample_rate)
				self.add_audio('pred_postnet', wav_postnet, iteration, hps.sample_rate)
			except:
				pass
