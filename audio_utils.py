"""
datautils.py:  Just some routines that we use for moving data around
"""
import librosa
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram


def load_audio(audio_path, mono=None, sr=None):  # wrapper for librosa.load
    signal, sr = librosa.load(audio_path, mono=mono, sr=sr)
    return signal, sr


def mean_features(file_name):
    X, sample_rate = librosa.load(file_name)
    # print("Features :",len(X), "sampled at ", sample_rate, "hz")
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
    ftrz = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
    return ftrz


def cnn_features(fn, bands=60, frames=41):
    def windows(data, window_size):
        start = 0
        while start < len(data):
            yield start, start + window_size
            start += (window_size // 2)

    window_size = 512 * (frames - 1)
    log_specgrams = []

    sound_clip, s = librosa.load(fn)
    for (start, end) in windows(sound_clip, window_size):
        if len(sound_clip[start:end]) == int(window_size):
            signal = sound_clip[start:end]
            melspec = librosa.feature.melspectrogram(signal, n_mels=bands)
            logspec = librosa.amplitude_to_db(melspec)
            logspec = logspec.T.flatten()[:, np.newaxis].T
            log_specgrams.append(logspec)

    log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams), bands, frames, 1)
    features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis=3)
    for i in range(len(features)):
        features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])

    return np.array(features)


def seq_features(fn, bands=20, frames=41):
    def windows(data, window_size):
        start = 0
        while start < len(data):
            yield start, start + window_size
            start += (window_size // 2)

    window_size = 512 * (frames - 1)
    mfccs = []
    sound_clip, s = librosa.load(fn)
    for (start, end) in windows(sound_clip, window_size):
        if len(sound_clip[start:end]) == window_size:
            signal = sound_clip[start:end]
            mfcc = librosa.feature.mfcc(y=signal, sr=s, n_mfcc=bands).T.flatten()[:, np.newaxis].T
            mfccs.append(mfcc)
    features = np.asarray(mfccs).reshape(len(mfccs), bands, frames)
    return np.array(features)


if __name__ == '__main__':
    pass
    # from dataset import audiodata
    #
    # sample_filename = os.path.join(audiodata, 'background_0001.wav')  #"samples/us8k/siren.wav"
    #
    # mfccs, chroma, mel, contrast, tonnetz = extract_feature(sample_filename)
    # all_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
    # print("MFCSS  = ", len(mfccs))
    # print("Chroma = ", len(chroma))
    # print("Mel = ", len(mel))
    # print("Contrast = ", len(contrast))
    # print("Tonnetz = ", len(tonnetz))
    #
    # data_points, _ = librosa.load(sample_filename)
    # print("IN: Initial Data Points =", len(data_points), np.shape(data_points))
    # print("OUT: Total features =", len(all_features))