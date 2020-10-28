#!/bin/bash
# You need to modify this path to your downloaded dataset directory
DATASET_DIR='data/'

# You need to modify this path to your workspace to store features and models
WORKSPACE='workspace/'

# Hyper-parameters
GPU_ID=0
MODEL_TYPE='Logmel_Cnn'
# MODEL_TYPE='Cqt_Cnn'
# MODEL_TYPE='Gamm_Cnn'
# MODEL_TYPE='Mfcc_Cnn'
# MODEL_TYPE='DCMR'
# MODEL_TYPE='Logmel_DCMF'
# MODEL_TYPE='Cqt_DCMF'
# MODEL_TYPE='Gamm_DCMF'
# MODEL_TYPE='Mfcc_DCMF'
# MODEL_TYPE='Logmel_DCMT'
# MODEL_TYPE='Cqt_DCMT'
# MODEL_TYPE='Gamm_DCMT'
# MODEL_TYPE='Mfcc_DCMT'
# MODEL_TYPE='DCMR_DCMF'
# MODEL_TYPE='DCMR_DCMT'
# MODEL_TYPE='DCMR_DCMF_DCMT'


#### Origanl Train (Other Models)
BATCH_SIZE=128
ITE_TRAIN=15000
ITE_EVA=12000
ITE_STORE=12000

############ Train and validate on development dataset ############
# Calculate feature
python utils/features.py calculate_feature_for_all_audio_files --dataset_dir=$DATASET_DIR --subtask='a' --data_type='development' --workspace=$WORKSPACE

# Calculate scalar
python utils/features.py calculate_scalar --subtask='a' --data_type='development' --workspace=$WORKSPACE

# Subtask A
CUDA_VISIBLE_DEVICES=$GPU_ID python pytorch/main.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --subtask='a' --data_type='development' --holdout_fold=1 --model_type=$MODEL_TYPE --batch_size=$BATCH_SIZE --ite_train=$ITE_TRAIN --ite_eva=$ITE_EVA --ite_store=$ITE_STORE --fixed=$FIXED --finetune=$FINETUNE --cuda

CUDA_VISIBLE_DEVICES=$GPU_ID python pytorch/main.py inference_validation --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --subtask='a' --data_type='development' --holdout_fold=1 --model_type=$MODEL_TYPE --iteration=$ITE_TRAIN --batch_size=$BATCH_SIZE --cuda
