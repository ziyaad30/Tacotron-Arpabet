import scipy
import librosa
import numpy as np
from scipy.io import wavfile
from librosa.util import normalize


_mel_basis = None
num_mels = 80
num_freq = 513
sample_rate = 22050
frame_shift = 256
frame_length = 1024
fmin = 0
fmax = 8000
power = 1.5
gl_iters = 30


def spectrogram(y):
    D = _stft(y)
    S = _amp_to_db(np.abs(D))
    return S


def inv_spectrogram(S):
    S = _db_to_amp(S)
    return _griffin_lim(S ** power)


def melspectrogram(y):
    D = _stft(y)
    S = _amp_to_db(_linear_to_mel(np.abs(D)))
    return S


def inv_melspectrogram(mel):
    mel = _db_to_amp(mel)
    S = _mel_to_linear(mel)
    return _griffin_lim(S**power)


def _griffin_lim(S):
    '''librosa implementation of Griffin-Lim
    Based on https://github.com/librosa/librosa/issues/434
    '''
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(S_complex * angles)
    for i in range(gl_iters):
        angles = np.exp(1j * np.angle(_stft(y)))
        y = _istft(S_complex * angles)
    return np.clip(y, a_max = 1, a_min = -1)


# Conversions:
def _stft(y):
    n_fft, hop_length, win_length = _stft_parameters()
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, 
            win_length=win_length, pad_mode='reflect')


def _istft(y):
    _, hop_length, win_length = _stft_parameters()
    return librosa.istft(y, hop_length=hop_length, win_length=win_length)


def _stft_parameters():
    return (num_freq - 1) * 2, frame_shift, frame_length


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
    n_fft = (num_freq - 1) * 2
    return librosa.filters.mel(sample_rate, n_fft, n_mels=num_mels, fmin = fmin, fmax = fmax)


def _amp_to_db(x):
    return np.log(np.maximum(1e-5, x))


def _db_to_amp(x):
    return np.exp(x)
