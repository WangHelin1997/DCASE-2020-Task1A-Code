import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], 'utils'))
import numpy as np
import argparse
import h5py
import librosa
from scipy import signal
import matplotlib.pyplot as plt
import time
import math
import pandas as pd
import random
from gtg_example import gtg_in_dB
from utilities import (create_folder, read_audio, calculate_scalar_of_tensor, 
    pad_truncate_sequence, get_subdir, read_metadata, read_audio_gamm)
import config


class LogMelExtractor(object):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax):
        '''Log mel feature extractor. 
        
        Args:
          sample_rate: int
          window_size: int
          hop_size: int
          mel_bins: int
          fmin: int, minimum frequency of mel filter banks
          fmax: int, maximum frequency of mel filter banks
        '''
        
        self.window_size = window_size
        self.hop_size = hop_size
        self.window_func = np.hanning(window_size)
        
        self.melW = librosa.filters.mel(
            sr=sample_rate, 
            n_fft=window_size, 
            n_mels=mel_bins, 
            fmin=fmin, 
            fmax=fmax).T
        '''(n_fft // 2 + 1, mel_bins)'''

    def transform(self, audio):
        '''Extract feature of a singlechannel audio file. 
        
        Args:
          audio: (samples,)
          
        Returns:
          feature: (frames_num, freq_bins)
        '''
    
        window_size = self.window_size
        hop_size = self.hop_size
        window_func = self.window_func
        
        # Compute short-time Fourier transform
        stft_matrix = librosa.core.stft(
            y=audio, 
            n_fft=window_size, 
            hop_length=hop_size, 
            window=window_func, 
            center=True, 
            dtype=np.complex64, 
            pad_mode='reflect').T
        '''(N, n_fft // 2 + 1)'''
    
        # Mel spectrogram
        mel_spectrogram = np.dot(np.abs(stft_matrix) ** 2, self.melW)
        
        # Log mel spectrogram
        logmel_spectrogram = librosa.core.power_to_db(
            mel_spectrogram, ref=1.0, amin=1e-10, 
            top_db=None)
        
        logmel_spectrogram = logmel_spectrogram.astype(np.float32)
        
        return logmel_spectrogram


def calculate_feature_for_all_audio_files(args):
    '''Calculate feature of audio files and write out features to a hdf5 file. 
    
    Args:
      dataset_dir: string
      workspace: string
      subtask: 'a' | 'b' | 'c'
      data_type: 'development' | 'evaluation'
      mini_data: bool, set True for debugging on a small part of data
    '''

    # Arguments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    subtask = args.subtask
    data_type = args.data_type
    mini_data = args.mini_data
    
    sample_rate = config.sample_rate
    window_size = config.window_size
    hop_size = config.hop_size
    mel_bins = config.mel_bins
    fmin = config.fmin
    fmax = config.fmax
    frames_per_second = config.frames_per_second
    frames_num = config.frames_num
    total_samples = config.total_samples
    lb_to_idx = config.lb_to_idx
    mfcc_frames = config.mfcc_frames
    n_mfcc = config.n_mfcc
    mfcc_hop_size = config.mfcc_hop_size
    gamm_frames = config.gamm_frames
    n_gamm = config.n_gamm
    # Paths
    if mini_data:
        prefix = 'minidata_'
    else:
        prefix = ''
        
    sub_dir = get_subdir(subtask, data_type)
    audios_dir = os.path.join(dataset_dir, sub_dir, 'audio')

    if data_type == 'development':
        metadata_path = os.path.join(dataset_dir, sub_dir, 'meta.csv')
    elif data_type == 'leaderboard':
        metadata_path = os.path.join(dataset_dir, sub_dir, 'evaluation_setup', 'test.csv')
    elif data_type == 'evaluation':
        metadata_path = os.path.join(dataset_dir, sub_dir, 'evaluation_setup', 'fold1_test.csv')
    else:
        raise Exception('Incorrect data_type!')
    
    feature_path = os.path.join(workspace, 'features', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        '{}.h5'.format(sub_dir))
    create_folder(os.path.dirname(feature_path))
        
    # Feature extractor
    feature_extractor = LogMelExtractor(
        sample_rate=sample_rate, 
        window_size=window_size, 
        hop_size=hop_size, 
        mel_bins=mel_bins, 
        fmin=fmin, 
        fmax=fmax)

    # Read metadata
    meta_dict = read_metadata(metadata_path)

    # Extract features and targets 
    if mini_data:
        mini_num = 10
        total_num = len(meta_dict['audio_name'])
        random_state = np.random.RandomState(1234)
        indexes = random_state.choice(total_num, size=mini_num, replace=False)
        for key in meta_dict.keys():
            meta_dict[key] = meta_dict[key][indexes]
        
    print('Extracting features of all audio files ...')
    extract_time = time.time()
    
    # Hdf5 file for storing features and targets
    hf = h5py.File(feature_path, 'w')

    hf.create_dataset(
        name='audio_name', 
        data=[audio_name.encode() for audio_name in meta_dict['audio_name']], 
        dtype='S80')

    if 'scene_label' in meta_dict.keys():
        hf.create_dataset(
            name='scene_label', 
            data=[scene_label.encode() for scene_label in meta_dict['scene_label']], 
            dtype='S24')
            
    if 'identifier' in meta_dict.keys():
        hf.create_dataset(
            name='identifier', 
            data=[identifier.encode() for identifier in meta_dict['identifier']], 
            dtype='S24')
            
    if 'source_label' in meta_dict.keys():
        hf.create_dataset(
            name='source_label', 
            data=[source_label.encode() for source_label in meta_dict['source_label']], 
            dtype='S8')

    hf.create_dataset(
        name='feature', 
        shape=(0, total_samples), 
        maxshape=(None, total_samples), 
        dtype=np.float32)
    hf.create_dataset(
        name='feature_gamm', 
        shape=(0, gamm_frames, n_gamm), 
        maxshape=(None, gamm_frames, n_gamm), 
        dtype=np.float32)
    hf.create_dataset(
        name='feature_mfcc', 
        shape=(0, mfcc_frames, n_mfcc), 
        maxshape=(None, mfcc_frames, n_mfcc), 
        dtype=np.float32)
    hf.create_dataset(
        name='feature_panns', 
        shape=(0, 320000), 
        maxshape=(None, 320000), 
        dtype=np.float32)
    
    for (n, audio_name) in enumerate(meta_dict['audio_name']):
        audio_path = os.path.join(audios_dir, audio_name)
        print(n, audio_path)
        
        # Read audio
        (audio, _) = read_audio(
            audio_path=audio_path, 
            target_fs=sample_rate)
        audio = audio[:sample_rate*10]
        (audio_gamm, _) = read_audio_gamm(
            audio_path=audio_path, 
            target_fs=sample_rate)
        fea_gamm, _ = gtg_in_dB(audio_gamm, sample_rate) 
        fea_gamm = fea_gamm.transpose(1, 0)
        sound, fs = librosa.load(audio_path)
        fea_mfcc = librosa.feature.mfcc(y=sound, sr=fs, hop_length=mfcc_hop_size, n_mfcc=n_mfcc)
        fea_mfcc = fea_mfcc.transpose(1, 0)
        (waveform, _) = librosa.core.load(audio_path, sr=32000, mono=True)
        waveform = waveform[:320000]
        
        hf['feature'].resize((n + 1, total_samples))
        hf['feature'][n] = audio
        hf['feature_gamm'].resize((n + 1, gamm_frames, n_gamm))
        hf['feature_gamm'][n] = fea_gamm
        hf['feature_mfcc'].resize((n + 1, mfcc_frames, n_mfcc))
        hf['feature_mfcc'][n] = fea_mfcc
        hf['feature_panns'].resize((n + 1, 320000))
        hf['feature_panns'][n] = waveform
            
    hf.close()
        
    print('Write hdf5 file to {} using {:.3f} s'.format(
        feature_path, time.time() - extract_time))
    
    
def calculate_scalar(args):
    '''Calculate and write out scalar of features. 
    
    Args:
      workspace: string
      subtask: 'a' | 'b' | 'c'
      data_type: 'train'
      mini_data: bool, set True for debugging on a small part of data
    '''

    # Arguments & parameters
    workspace = args.workspace
    subtask = args.subtask
    data_type = args.data_type
    mini_data = args.mini_data
    
    mel_bins = config.mel_bins
    frames_per_second = config.frames_per_second
    
    # Paths
    if mini_data:
        prefix = 'minidata_'
    else:
        prefix = ''
    
    sub_dir = get_subdir(subtask, data_type)
    
    feature_path = os.path.join(workspace, 'features', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        '{}.h5'.format(sub_dir))
        
    scalar_path = os.path.join(workspace, 'scalars', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        '{}.h5'.format(sub_dir))
    create_folder(os.path.dirname(scalar_path))
        
    # Load data
    load_time = time.time()
    
    with h5py.File(feature_path, 'r') as hf:
        features = hf['feature'][:]
        features_gamm = hf['feature_gamm'][:]
        features_mfcc = hf['feature_mfcc'][:]
        features_panns = hf['feature_panns'][:]
    # Calculate scalar
    features = np.concatenate(features[None,:], axis=0)
    (mean, std) = calculate_scalar_of_tensor(features)
    features_gamm = np.concatenate(features_gamm, axis=0)
    (mean_gamm, std_gamm) = calculate_scalar_of_tensor(features_gamm)
    features_mfcc = np.concatenate(features_mfcc, axis=0)
    (mean_mfcc, std_mfcc) = calculate_scalar_of_tensor(features_mfcc)
    features_panns = np.concatenate(features_panns[None,:], axis=0)
    (mean_panns, std_panns) = calculate_scalar_of_tensor(features_panns)
    with h5py.File(scalar_path, 'w') as hf:
        hf.create_dataset('mean', data=mean, dtype=np.float32)
        hf.create_dataset('std', data=std, dtype=np.float32)
        hf.create_dataset('mean_gamm', data=mean_gamm, dtype=np.float32)
        hf.create_dataset('std_gamm', data=std_gamm, dtype=np.float32)
        hf.create_dataset('mean_mfcc', data=mean_mfcc, dtype=np.float32)
        hf.create_dataset('std_mfcc', data=std_mfcc, dtype=np.float32)
        hf.create_dataset('mean_panns', data=mean_panns, dtype=np.float32)
        hf.create_dataset('std_panns', data=std_panns, dtype=np.float32)
    
    print('Write out scalar to {}'.format(scalar_path))
            

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='mode')

    # Calculate feature for all audio files
    parser_logmel = subparsers.add_parser('calculate_feature_for_all_audio_files')    
    parser_logmel.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')    
    parser_logmel.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')        
    parser_logmel.add_argument('--subtask', type=str, choices=['a', 'b', 'c'], required=True, help='Correspond to 3 subtasks in DCASE2019 Task1')        
    parser_logmel.add_argument('--data_type', type=str, choices=['development', 'leaderboard', 'evaluation'], required=True)        
    parser_logmel.add_argument('--mini_data', action='store_true', default=False, help='Set True for debugging on a small part of data.')
    
    # Calculate scalar
    parser_scalar = subparsers.add_parser('calculate_scalar')    
    parser_scalar.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')    
    parser_scalar.add_argument('--subtask', type=str, choices=['a', 'b', 'c'], required=True, help='Correspond to 3 subtasks in DCASE2019 Task1')        
    parser_scalar.add_argument('--data_type', type=str, choices=['development', 'evaluation'], required=True)        
    parser_scalar.add_argument('--mini_data', action='store_true', default=False, help='Set True for debugging on a small part of data.')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.mode == 'calculate_feature_for_all_audio_files':
        calculate_feature_for_all_audio_files(args)
        
    elif args.mode == 'calculate_scalar':
        calculate_scalar(args)
        
    else:
        raise Exception('Incorrect arguments!')