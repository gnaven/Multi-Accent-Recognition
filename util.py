from __future__ import print_function, division, absolute_import, unicode_literals
import six
import numpy as np
from scipy.io import wavfile
import scipy.signal 
import sys
from matplotlib import pyplot as plt
import librosa

def wavread(filepath):
    if '.mp3' in filepath:
        return mp3read(filepath)
    
    
    fs, x = wavfile.read(filepath)    
    if (len(x.shape)) == 1:
        x = x.reshape(-1,1)
    # print(x.shape)

    if x.dtype != np.float32:
        x = x/np.iinfo(x.dtype).max

    return x,fs

def wavwrite(filepath, x, fs):
    wavfile.write(filepath, fs, x)

def mp3read(filepath):
    y, sr = librosa.load(filepath)
    if (len(y.shape)) == 1:
        y = y.reshape(-1,1)    
    return y,sr

def stft_real(x, blockSize, hopSize, window='hamming'):
    # print(x.shape)
    _,_, S = scipy.signal.stft(
        x,
            window = window,
            nperseg=blockSize,
            noverlap = blockSize-hopSize,
            return_onesided=True)
    # print(S.shape)
    return S

def istft_real(S, blockSize, hopSize, window='hamming'):
    _,x = scipy.signal.istft(S, window= window, noverlap= blockSize- hopSize, input_onesided=True)
    return x