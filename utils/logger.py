import numpy as np
from utils.util import to_arr
from hparams import hparams as hps
from tensorboardX import SummaryWriter
from utils.audio import inv_melspectrogram
from utils.plot import plot_alignment_to_numpy, plot_spectrogram_to_numpy


class Tacotron2Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Tacotron2Logger, self).__init__(logdir, flush_secs = 5)

    def log_training(self, items, grad_norm, learning_rate, iteration):
            self.add_scalar('loss.mel', items[0], iteration)
            self.add_scalar('loss.gate', items[1], iteration)
            self.add_scalar('grad.norm', grad_norm, iteration)
            self.add_scalar('learning.rate', learning_rate, iteration)

    def sample_train(self, outputs, iteration):
            mel_outputs = to_arr(outputs[0][0])
            mel_outputs_postnet = to_arr(outputs[1][0])
            alignments = to_arr(outputs[3][0]).T
            
            # plot alignment, mel and postnet output
            self.add_image(
                'train.align',
                plot_alignment_to_numpy(alignments),
                iteration)
            self.add_image(
                'train.mel',
                plot_spectrogram_to_numpy(mel_outputs),
                iteration)
            self.add_image(
                'train.mel_post',
                plot_spectrogram_to_numpy(mel_outputs_postnet),
                iteration)

    def sample_infer(self, outputs, iteration):
            mel_outputs = to_arr(outputs[0][0])
            mel_outputs_postnet = to_arr(outputs[1][0])
            alignments = to_arr(outputs[2][0]).T
            
            # plot alignment, mel and postnet output
            self.add_image(
                'infer.align',
                plot_alignment_to_numpy(alignments),
                iteration)
            self.add_image(
                'infer.mel',
                plot_spectrogram_to_numpy(mel_outputs),
                iteration)
            self.add_image(
                'infer.mel_post',
                plot_spectrogram_to_numpy(mel_outputs_postnet),
                iteration)
            
            # save audio
            wav = inv_melspectrogram(mel_outputs)
            wav_postnet = inv_melspectrogram(mel_outputs_postnet)
            self.add_audio('infer.wav', wav, iteration, hps.sample_rate)
            self.add_audio('infer.wav_post', wav_postnet, iteration, hps.sample_rate)
