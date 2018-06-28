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


def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    # print("Features :",len(X), "sampled at ", sample_rate, "hz")
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
    return mfccs, chroma, mel, contrast, tonnetz


def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode

def assure_path_exists(path):
    mydir = os.path.join(os.getcwd(), path)
    if not os.path.exists(mydir):
        os.makedirs(mydir)


if __name__ == '__main__':
    from dataset import audiodata

    sample_filename = os.path.join(audiodata, 'background_0001.wav')  #"samples/us8k/siren.wav"

    mfccs, chroma, mel, contrast, tonnetz = extract_feature(sample_filename)
    all_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
    print("MFCSS  = ", len(mfccs))
    print("Chroma = ", len(chroma))
    print("Mel = ", len(mel))
    print("Contrast = ", len(contrast))
    print("Tonnetz = ", len(tonnetz))

    data_points, _ = librosa.load(sample_filename)
    print("IN: Initial Data Points =", len(data_points), np.shape(data_points))
    print("OUT: Total features =", len(all_features))