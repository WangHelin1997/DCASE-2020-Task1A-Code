import os
import sys
import numpy as np
import soundfile
import librosa
import h5py
import math
import pandas as pd
from sklearn import metrics
import logging
import matplotlib.pyplot as plt
import torch
import config
import scipy.io.wavfile as wf
from pydub import AudioSegment
import csv

'''
def mixup_data(x, y, alpha=0.2):
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
'''
def mixup_data(x1,x2,x3,x4, y, alpha=0.2):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x1.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x1 = lam * x1 + (1 - lam) * x1[index, :]
    mixed_x2 = lam * x2 + (1 - lam) * x2[index, :]
    mixed_x3 = lam * x3 + (1 - lam) * x3[index, :]
    mixed_x4 = lam * x4 + (1 - lam) * x4[index, :]
    y_a, y_b = y, y[index]
    return mixed_x1, mixed_x2, mixed_x3,mixed_x4, y_a, y_b, lam

def mixup_criterion(class_criterion, pred, y_a, y_b, lam):
    return lam * class_criterion(pred, y_a) + (1 - lam) * class_criterion(pred, y_b)

def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)

def get_filename(path):
    path = os.path.realpath(path)
    name_ext = path.split('/')[-1]
    name = os.path.splitext(name_ext)[0]
    return name

def create_logging(log_dir, filemode):
    create_folder(log_dir)
    i1 = 0

    while os.path.isfile(os.path.join(log_dir, '{:04d}.log'.format(i1))):
        i1 += 1
        
    log_path = os.path.join(log_dir, '{:04d}.log'.format(i1))
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename=log_path,
        filemode=filemode)
    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logger = logging.getLogger('')
    while logger.handlers:
        logger.handlers.pop()
    logger.addHandler(console)
    
    return logging
    
def read_audio(audio_path, target_fs=None):
    (audio, fs) = soundfile.read(audio_path)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
        
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
        
    return audio, fs
    
def read_audio_gamm(audio_path, target_fs=None):
    audio1='temp_16.wav'
    s = AudioSegment.from_file(audio_path, format = "wav" )
    s.export(audio1 , bitrate="16", format="wav")
    fs, audio = wf.read(audio1)
    return audio, fs

def pad_truncate_sequence(x, max_len):
    if len(x) < max_len:
        return np.concatenate((x, np.zeros(max_len - len(x))))
    else:
        return x[0 : max_len]
    
    
def calculate_scalar_of_tensor(x):
    if x.ndim == 2:
        axis = 0
    elif x.ndim == 3:
        axis = (0, 1)

    mean = np.mean(x, axis=axis)
    std = np.std(x, axis=axis)

    return mean, std


def load_scalar(scalar_path):
    with h5py.File(scalar_path, 'r') as hf:
        mean = hf['mean'][:]
        std = hf['std'][:]
        mean_gamm = hf['mean_gamm'][:]
        std_gamm = hf['std_gamm'][:]
        mean_mfcc = hf['mean_mfcc'][:]
        std_mfcc = hf['std_mfcc'][:]
        mean_panns = hf['mean_panns'][:]
        std_panns = hf['std_panns'][:]
        
    scalar = {'mean': mean, 'std': std,'mean_gamm': mean_gamm, 'std_gamm': std_gamm,'mean_mfcc': mean_mfcc, 'std_mfcc': std_mfcc,'mean_panns': mean_panns, 'std_panns': std_panns}
    return scalar
    
    
def scale(x, mean, std):
    return (x - mean) / std
    
    
def inverse_scale(x, mean, std):
    return x * std + mean
    
    
def get_subdir(subtask, data_type):
    if subtask == 'a':
        subdir = 'TAU-urban-acoustic-scenes-2020-mobile-{}'.format(data_type)
    elif subtask == 'b':
        subdir = 'TAU-urban-acoustic-scenes-2019-mobile-{}'.format(data_type)
    elif subtask == 'c':
        subdir = 'TAU-urban-acoustic-scenes-2019-openset-{}'.format(data_type)
    else:
        raise Exception('Incorrect argument!')
    
    return subdir
        
        
def read_metadata(metadata_path):
    '''Read metadata from a csv file. 
    
    Returns:
      meta_dict: dict of meta data, e.g.:
        {'audio_name': np.array(['a.wav', 'b.wav', ...]), 
         'scene_label': np.array(['airport', 'bus', ...]), 
         ...}
    '''
    df = pd.read_csv(metadata_path, sep='\t')
    
    meta_dict = {}
    
    meta_dict['audio_name'] = np.array(
        [name.split('/')[1] for name in df['filename'].tolist()])
    
    if 'scene_label' in df.keys():
        meta_dict['scene_label'] = np.array(df['scene_label'])
        
    if 'identifier' in df.keys():
        meta_dict['identifier'] = np.array(df['identifier'])
        
    if 'source_label' in df.keys():
        meta_dict['source_label'] = np.array(df['source_label'])
    
    return meta_dict
    
    
def sparse_to_categorical(x, n_out):
    x = x.astype(int)
    shape = x.shape
    x = x.flatten()
    N = len(x)
    x_categ = np.zeros((N,n_out))
    x_categ[np.arange(N), x] = 1
    return x_categ.reshape((shape)+(n_out,))
    
    
def get_sources(subtask):
    if subtask in ['a', 'c']:
        return ['a', 'b', 'c', 's1', 's2', 's3']
    elif subtask == 'b':
        return ['a', 'b', 'c']
    else:
        raise Exception('Incorrect argument!')
        

def write_submission(output_dict, subtask, data_type, submission_path):
    '''Write output to submission. 

    Args:
      output_dict: 
        {'audio_name': (audios_num,), 'output': (audios_num, classes_num)}
      subtask: 'a' | 'b' | 'c'
      data_type: 'leaderboard' | 'evaluation'
      submission_path: string
    '''
    fw = open(submission_path, 'w')
    csv_writer = csv.writer(fw,delimiter='\t')
    csv_writer.writerow(['filename','scene_label','airport','bus','metro','metro_station','park','public_square','shopping_mall','street_pedestrian','street_traffic','tram'])
    #0,7,8,2,9,4,1,3,5,6

    for n in range(len(output_dict['audio_name'])):
   
        audio_name = output_dict['audio_name'][n]
        pred_label = config.idx_to_lb[np.argmax(output_dict['output'][n])]
        logloss = output_dict['loss'][n]
        print(n,pred_label,logloss)
        csv_writer.writerow([audio_name,pred_label,logloss[0],logloss[7],logloss[8],logloss[2],logloss[9],logloss[4],logloss[1],logloss[3],logloss[5],logloss[6]])                  
    fw.close()
    logging.info('Write submission to {}'.format(submission_path))
