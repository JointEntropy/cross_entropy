import pandas as pd
import os
from audio_utils import load_audio, extract_feature
import numpy as np
from tqdm import tqdm
from utils import save_obj, load_obj

metadata = '/media/grigory/Диск/ITMO_DATA/data_v_7_stc/meta/meta.txt'
audiodata = '/media/grigory/Диск/ITMO_DATA/data_v_7_stc/audio'
extracted_data = '/media/grigory/Диск/ITMO_DATA/data_v_7_stc/extracted'
extracted_data_name = 'features_labels_test'

if not os.path.exists(extracted_data):
    os.mkdir(extracted_data)




def prepare_generator(meta_path, audio_path):
    df = pd.read_csv(meta_path, sep='\t', header=None)
    df.columns = ['name', 'col1', 'col2', 'col3', 'label']

    for name, label in zip(df['name'], df['label']):
        filedata = load_audio(os.path.join(audio_path, name))
        yield filedata, label


def parse_audio_files(meta_path, audio_path):
    df = pd.read_csv(meta_path, sep='\t', header=None)
    df.columns = ['name', 'col1', 'col2', 'col3', 'label']

    features, labels = np.empty((0,193)), np.empty(0)
    for i, (name, label) in enumerate(zip(tqdm(df['name']), df['label'])):
        fpth = os.path.join(audio_path, name)
        try:
            mfccs, chroma, mel, contrast, tonnetz = extract_feature(fpth)
            ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
            features = np.vstack([features,ext_features])
            labels = np.append(labels, label)
        except BaseException as e:  # да, baseException это плохо
            print("Error processing " + fpth + " - skipping")
    return np.array(features), np.array(labels)


def prepare_submit():
    pass

if __name__ == '__main__':
    ftrs, labels = parse_audio_files(metadata, audiodata)
    save_obj((ftrs, labels), os.path.join(extracted_data, extracted_data_name))

    # ftr,labels = load_obj(os.path.join(extracted_data, 'features_labels'))
    # pass
    # u_labels = set()
    # for i, (comp, label) in enumerate(tqdm(prepare_generator(metadata, audiodata))):
    #     u_labels.add(label)
    #     if len(u_labels) > 1:
    #         print('Not only one!!')
    #         break

