import scipy
import librosa
import librosa.filters
import numpy as np
from scipy.io import wavfile
from hparams import hparams as hps


def load_wav(path):
	sr, wav = wavfile.read(path)
	wav = wav.astype(np.float32)
	wav = wav/np.max(np.abs(wav))
	try:
		assert sr == hps.sample_rate
	except:
		print('Error:', path, 'has wrong sample rate.')
	return wav


def save_wav(wav, path):
	wav *= 32767 / max(0.01, np.max(np.abs(wav)))
	wavfile.write(path, hps.sample_rate, wav.astype(np.int16))


def preemphasis(x):
	return scipy.signal.lfilter([1, -hps.preemphasis], [1], x)


def inv_preemphasis(x):
	return scipy.signal.lfilter([1], [1, -hps.preemphasis], x)


def spectrogram(y):
	D = _stft(preemphasis(y))
	S = _amp_to_db(np.abs(D)) - hps.ref_level_db
	return _normalize(S)


def inv_spectrogram(spectrogram):
	'''Converts spectrogram to waveform using librosa'''
	S = _db_to_amp(_denormalize(spectrogram) + hps.ref_level_db)	# Convert back to linear
	return inv_preemphasis(_griffin_lim(S ** hps.power))			# Reconstruct phase

def melspectrogram(y):
	D = _stft(preemphasis(y))
	S = _amp_to_db(_linear_to_mel(np.abs(D))) - hps.ref_level_db
	return _normalize(S)


def inv_melspectrogram(spectrogram):
	mel = _db_to_amp(_denormalize(spectrogram) + hps.ref_level_db)
	S = _mel_to_linear(mel)
	return inv_preemphasis(_griffin_lim(S ** hps.power))


def find_endpoint(wav, threshold_db=-40, min_silence_sec=0.8):
	window_length = int(hps.sample_rate * min_silence_sec)
	hop_length = int(window_length / 4)
	threshold = _db_to_amp(threshold_db)
	for x in range(hop_length, len(wav) - window_length, hop_length):
		if np.max(wav[x:x+window_length]) < threshold:
			return x + hop_length
	return len(wav)


def _griffin_lim(S):
	'''librosa implementation of Griffin-Lim
	Based on https://github.com/librosa/librosa/issues/434
	'''
	angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
	S_complex = np.abs(S).astype(np.complex)
	y = _istft(S_complex * angles)
	for i in range(hps.gl_iters):
		angles = np.exp(1j * np.angle(_stft(y)))
		y = _istft(S_complex * angles)
	return y


def _stft(y):
	n_fft, hop_length, win_length = _stft_parameters()
	return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _istft(y):
	_, hop_length, win_length = _stft_parameters()
	return librosa.istft(y, hop_length=hop_length, win_length=win_length)


def _stft_parameters():
	n_fft = (hps.num_freq - 1) * 2
	hop_length = int(hps.frame_shift_ms / 1000 * hps.sample_rate)
	win_length = int(hps.frame_length_ms / 1000 * hps.sample_rate)
	return n_fft, hop_length, win_length


# Conversions:

_mel_basis = None

def _linear_to_mel(spectrogram):
	global _mel_basis
	if _mel_basis is None:
		_mel_basis = _build_mel_basis()
	return np.dot(_mel_basis, spectrogram)
	

def _mel_to_linear(spectrogram):
	global _mel_basis
	if _mel_basis is None:
		_mel_basis = _build_mel_basis()
	inv_mel_basis = np.linalg.pinv(_mel_basis)
	inverse = np.dot(inv_mel_basis, spectrogram)
	inverse = np.maximum(1e-10, inverse)
	return inverse


def _build_mel_basis():
	n_fft = (hps.num_freq - 1) * 2
	return librosa.filters.mel(hps.sample_rate, n_fft, n_mels=hps.num_mels)

def _amp_to_db(x):
	return 20 * np.log10(np.maximum(1e-5, x))

def _db_to_amp(x):
	return np.power(10.0, x * 0.05)

def _normalize(S):
	return np.clip((S - hps.min_level_db) / -hps.min_level_db, 0, 1)

def _denormalize(S):
	return (np.clip(S, 0, 1) * -hps.min_level_db) + hps.min_level_db

