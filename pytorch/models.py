import math
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
from nnAudio import Spectrogram as Spec
from torchlibrosa.augmentation import SpecAugmentation
import config

def init_layer(layer, nonlinearity='leaky_relu'):
    """Initialize a Linear or Convolutional layer. """
    nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
    
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    
    bn.bias.data.fill_(0.)
    bn.running_mean.data.fill_(0.)
    bn.weight.data.fill_(1.)
    bn.running_var.data.fill_(1.)
    
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.init_weights()
        
    def init_weights(self):
        
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)
        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        else:
            raise Exception('Incorrect argument!')
        
        return x

class Logmel_Cnn(nn.Module):
    
    def __init__(self, classes_num, activation):
        super(Logmel_Cnn, self).__init__()

        self.activation = activation
        self.spec_layer = Spec.MelSpectrogram(sr=config.sample_rate, n_fft=config.window_size, n_mels=config.mel_bins, hop_length=config.hop_size, window='hann', center=True, pad_mode='reflect', htk=False, fmin=0.0, fmax=None, norm=1, trainable_mel=False, trainable_STFT=False)
        #self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, freq_drop_width=5, freq_stripes_num=2)
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.fc2 = nn.Linear(512, 512, bias=True)
        self.fc = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):

        init_layer(self.fc)
        init_layer(self.fc2)

    def forward(self, x1,x2,x3,x4):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        
        x = x1
        x = self.spec_layer(x)
        x = 10*torch.log10(x)
        x = torch.clamp(x, min=-100.)
        x = x.transpose(1,2)
        x = x[:,None,:,:]
        #if self.training:
            #x = self.spec_augmenter(x)
        '''(batch_size, 1, times_steps, freq_bins)'''
        
        x = self.conv_block1(x, pool_size=(4, 2), pool_type='avg')
        #x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(4, 2), pool_type='avg')
        #x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        #x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        #x = F.dropout(x, p=0.2, training=self.training)
        '''(batch_size, feature_maps, time_steps, freq_bins)'''
        
        x = torch.mean(x, dim=3)        # (batch_size, feature_maps, time_stpes)
        (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc(x)
        loss = torch.softmax(x,-1)
        if self.activation == 'logsoftmax':
            output = F.log_softmax(x, dim=-1)
            
        elif self.activation == 'sigmoid':
            output = torch.sigmoid(x)
        
        return output,loss

class Cqt_Cnn(nn.Module):
    
    def __init__(self, classes_num, activation):
        super(Cqt_Cnn, self).__init__()

        self.activation = activation
        self.spec_layer = Spec.CQT(sr=config.sample_rate, hop_length=config.hop_size, fmin=220, fmax=None, n_bins=64, bins_per_octave=12, norm=1, window='hann', center=True, pad_mode='reflect')
        #self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, freq_drop_width=5, freq_stripes_num=2)
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.fc2 = nn.Linear(512, 512, bias=True)
        self.fc = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):

        init_layer(self.fc)
        init_layer(self.fc2)

    def forward(self, x1,x2,x3,x4):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        
        x = x1
        x = self.spec_layer(x)
        x = 10*torch.log10(x)
        x = torch.clamp(x, min=-100.)
        x = x.transpose(1,2)
        x = x[:,None,:,:]
        #if self.training:
            #x = self.spec_augmenter(x)
        '''(batch_size, 1, times_steps, freq_bins)'''
        
        x = self.conv_block1(x, pool_size=(4, 2), pool_type='avg')
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(4, 2), pool_type='avg')
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        # x = F.dropout(x, p=0.2, training=self.training)
        '''(batch_size, feature_maps, time_steps, freq_bins)'''
        
        x = torch.mean(x, dim=3)        # (batch_size, feature_maps, time_stpes)
        (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc(x)
        loss = torch.softmax(x,-1)
        if self.activation == 'logsoftmax':
            output = F.log_softmax(x, dim=-1)
            
        elif self.activation == 'sigmoid':
            output = torch.sigmoid(x)
        
        return output,loss

class Gamm_Cnn(nn.Module):
    
    def __init__(self, classes_num, activation):
        super(Gamm_Cnn, self).__init__()

        self.activation = activation
        #self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, freq_drop_width=5, freq_stripes_num=2)
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.fc2 = nn.Linear(512, 512, bias=True)
        self.fc = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):

        init_layer(self.fc)
        init_layer(self.fc2)

    def forward(self, x1,x2,x3,x4):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        
        x = x2
        x = x[:,None,:,:]
        #if self.training:
            #x = self.spec_augmenter(x)
        '''(batch_size, 1, times_steps, freq_bins)'''
        
        x = self.conv_block1(x, pool_size=(4, 2), pool_type='avg')
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(4, 2), pool_type='avg')
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        # x = F.dropout(x, p=0.2, training=self.training)
        '''(batch_size, feature_maps, time_steps, freq_bins)'''
        
        x = torch.mean(x, dim=3)        # (batch_size, feature_maps, time_stpes)
        (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc(x)
        loss = torch.softmax(x,-1)
        if self.activation == 'logsoftmax':
            output = F.log_softmax(x, dim=-1)
            
        elif self.activation == 'sigmoid':
            output = torch.sigmoid(x)
        
        return output,loss

class Mfcc_Cnn(nn.Module):
    
    def __init__(self, classes_num, activation):
        super(Mfcc_Cnn, self).__init__()

        self.activation = activation
        #self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, freq_drop_width=5, freq_stripes_num=2)
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.fc2 = nn.Linear(512, 512, bias=True)
        self.fc = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):

        init_layer(self.fc)
        init_layer(self.fc2)

    def forward(self, x1,x2,x3,x4):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        
        x = x3
        x = x[:,None,:,:]
        #if self.training:
            #x = self.spec_augmenter(x)
        '''(batch_size, 1, times_steps, freq_bins)'''
        
        x = self.conv_block1(x, pool_size=(4, 2), pool_type='avg')
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(4, 2), pool_type='avg')
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        # x = F.dropout(x, p=0.2, training=self.training)
        '''(batch_size, feature_maps, time_steps, freq_bins)'''
        
        x = torch.mean(x, dim=3)        # (batch_size, feature_maps, time_stpes)
        (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc(x)
        loss = torch.softmax(x,-1)
        if self.activation == 'logsoftmax':
            output = F.log_softmax(x, dim=-1)
            
        elif self.activation == 'sigmoid':
            output = torch.sigmoid(x)
        
        return output,loss

class DCMR(nn.Module):
    
    def __init__(self, classes_num, activation):
        super(DCMR, self).__init__()

        self.activation = activation
        self.logmel_cnn = Logmel_Cnn(classes_num, activation)
        self.cqt_cnn = Cqt_Cnn(classes_num, activation)
        self.gamm_cnn = Gamm_Cnn(classes_num, activation)
        self.mfcc_cnn = Mfcc_Cnn(classes_num, activation)
        self.ite = 15000 # set your pre-trained model iteration
        self.model_path = 'workspace/checkpoints/main_eva/logmel_86frames_40melbins/TAU-urban-acoustic-scenes-2020-mobile-development/holdout_fold=1/' # set your model saved path.
        self.init_weights()

    def init_weights(self):
        logmel_path = self.model_path +'Logmel_Cnn/'+str(self.ite)+'_iterations.pth'
        cqt_path = self.model_path +'Cqt_Cnn/'+str(self.ite)+'_iterations.pth'
        gamm_path = self.model_path +'Gamm_Cnn/'+str(self.ite)+'_iterations.pth'
        mfcc_path = self.model_path +'Mfcc_Cnn/'+str(self.ite)+'_iterations.pth'

        logmel_ch = torch.load(logmel_path)
        cqt_ch = torch.load(cqt_path)
        gamm_ch = torch.load(gamm_path)
        mfcc_ch = torch.load(mfcc_path)
        self.logmel_cnn.load_state_dict(logmel_ch['model'])
        self.cqt_cnn.load_state_dict(cqt_ch['model'])
        self.gamm_cnn.load_state_dict(gamm_ch['model'])
        self.mfcc_cnn.load_state_dict(mfcc_ch['model'])

    def forward(self, input1,input2,input3,input4):
        '''
        Input: (batch_size, total_frames)'''
        
        x1,loss1 = self.logmel_cnn(input1,input2,input3,input4)
        x2,loss2 = self.cqt_cnn(input1,input2,input3,input4)
        x3,loss3 = self.gamm_cnn(input1,input2,input3,input4)
        x4,loss4 = self.mfcc_cnn(input1,input2,input3,input4)
        x = x1 + x2 + x3 + x4
        loss = (loss1+loss2+loss3+loss4)/4.
        return x,loss

class Logmel_DCMF(nn.Module):
    
    def __init__(self, classes_num, activation):
        super(Logmel_DCMF, self).__init__()

        self.activation = activation
        self.spec_layer = Spec.MelSpectrogram(sr=config.sample_rate, n_fft=config.window_size, n_mels=config.mel_bins, hop_length=config.hop_size, window='hann', center=True, pad_mode='reflect', htk=False, fmin=0.0, fmax=None, norm=1, trainable_mel=False, trainable_STFT=False)
        self.conv_block1_1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block1_2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block1_3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block1_4 = ConvBlock(in_channels=256, out_channels=512)
        
        self.conv_block2_1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2_2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block2_3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block2_4 = ConvBlock(in_channels=256, out_channels=512)
                              
        self.conv_block3_1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block3_2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3_3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block3_4 = ConvBlock(in_channels=256, out_channels=512)
                              
        self.conv_block4_1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block4_2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block4_3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4_4 = ConvBlock(in_channels=256, out_channels=512)
                              
                             
        self.fc1_1 = nn.Linear(512, 512, bias=True)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1_2 = nn.Linear(512, classes_num, bias=True)
        
        self.fc2_1 = nn.Linear(512, 512, bias=True)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2_2 = nn.Linear(512, classes_num, bias=True)
        
        self.fc3_1 = nn.Linear(512, 512, bias=True)
        self.dropout3 = nn.Dropout(p=0.5)
        self.fc3_2 = nn.Linear(512, classes_num, bias=True)
        
        self.fc4_1 = nn.Linear(512, 512, bias=True)
        self.dropout4 = nn.Dropout(p=0.5)
        self.fc4_2 = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):

        init_layer(self.fc1_1)
        init_layer(self.fc1_2)
        init_layer(self.fc2_1)
        init_layer(self.fc2_2)
        init_layer(self.fc3_1)
        init_layer(self.fc3_2)
        init_layer(self.fc4_1)
        init_layer(self.fc4_2)

    def forward(self, x1,x2,x3,x4):
        '''
        Input: (batch_size, scale_num, total_frames)'''
        outputs = []
        
        x = input
        x = self.spec_layer(x1)
        x = 10*torch.log10(x)
        x = torch.clamp(x, min=-100.)
        x = x.transpose(1,2)
        x = x[:,None,:,:]
        '''(batch_size, 1, times_steps, freq_bins)'''
        temp = x.clone()
        count = 0.
        
        x = temp[:,:,:,0:16]
        x = self.conv_block1_1(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block1_2(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block1_3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block1_4(x, pool_size=(2, 2), pool_type='avg')
        '''(batch_size, feature_maps, time_steps, freq_bins)'''
        (x, _) = torch.max(x[:,:,:,0], dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc1_1(x))
        x = self.dropout1(x)
        x = self.fc1_2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x
        count += 1.
        outputs.append(x)
        
        
        x = temp[:,:,:,8:24]
        x = self.conv_block2_1(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block2_2(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block2_3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2_4(x, pool_size=(2, 2), pool_type='avg')
        (x, _) = torch.max(x[:,:,:,0], dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc2_1(x))
        x = self.dropout2(x)
        x = self.fc2_2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x + co_output
        count += 1.
        outputs.append(x)
        
        x = temp[:,:,:,16:32]
        x = self.conv_block3_1(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block3_2(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block3_3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3_4(x, pool_size=(2, 2), pool_type='avg')
        (x, _) = torch.max(x[:,:,:,0], dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc3_1(x))
        x = self.dropout3(x)
        x = self.fc3_2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x + co_output
        count += 1.
        outputs.append(x)
        
        x = temp[:,:,:,24:40]
        x = self.conv_block4_1(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block4_2(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block4_3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4_4(x, pool_size=(2, 2), pool_type='avg')
        (x, _) = torch.max(x[:,:,:,0], dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc4_1(x))
        x = self.dropout4(x)
        x = self.fc4_2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x + co_output
        count += 1.
        outputs.append(x)
        
        outputs.append(co_output/count)
        outputs = torch.cat((outputs), 1)
        loss = torch.softmax(outputs,-1)
        if self.activation == 'logsoftmax':
            output = F.log_softmax(outputs, dim=-1)
            
        elif self.activation == 'sigmoid':
            output = torch.sigmoid(outputs)
            
        return output,loss

class Mfcc_DCMF(nn.Module):
    
    def __init__(self, classes_num, activation):
        super(Mfcc_DCMF, self).__init__()

        self.activation = activation
        
        self.conv_block1_1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block1_2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block1_3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block1_4 = ConvBlock(in_channels=256, out_channels=512)
        
        self.conv_block2_1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2_2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block2_3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block2_4 = ConvBlock(in_channels=256, out_channels=512)
                              
        self.conv_block3_1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block3_2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3_3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block3_4 = ConvBlock(in_channels=256, out_channels=512)
                              
        self.conv_block4_1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block4_2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block4_3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4_4 = ConvBlock(in_channels=256, out_channels=512)
                              
                             
        self.fc1_1 = nn.Linear(512, 512, bias=True)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1_2 = nn.Linear(512, classes_num, bias=True)
        
        self.fc2_1 = nn.Linear(512, 512, bias=True)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2_2 = nn.Linear(512, classes_num, bias=True)
        
        self.fc3_1 = nn.Linear(512, 512, bias=True)
        self.dropout3 = nn.Dropout(p=0.5)
        self.fc3_2 = nn.Linear(512, classes_num, bias=True)
        
        self.fc4_1 = nn.Linear(512, 512, bias=True)
        self.dropout4 = nn.Dropout(p=0.5)
        self.fc4_2 = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):

        init_layer(self.fc1_1)
        init_layer(self.fc1_2)
        init_layer(self.fc2_1)
        init_layer(self.fc2_2)
        init_layer(self.fc3_1)
        init_layer(self.fc3_2)
        init_layer(self.fc4_1)
        init_layer(self.fc4_2)

    def forward(self, x1,x2,x3,x4):
        '''
        Input: (batch_size, scale_num, total_frames)'''
        outputs = []
        
        x = x3
        x = x[:,None,:,:]
        '''(batch_size, 1, times_steps, freq_bins)'''
        temp = x.clone()
        count = 0.
        
        x = temp[:,:,:,0:16]
        x = self.conv_block1_1(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block1_2(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block1_3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block1_4(x, pool_size=(2, 2), pool_type='avg')
        '''(batch_size, feature_maps, time_steps, freq_bins)'''
        (x, _) = torch.max(x[:,:,:,0], dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc1_1(x))
        x = self.dropout1(x)
        x = self.fc1_2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x
        count += 1.
        outputs.append(x)
        
        
        x = temp[:,:,:,8:24]
        x = self.conv_block2_1(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block2_2(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block2_3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2_4(x, pool_size=(2, 2), pool_type='avg')
        (x, _) = torch.max(x[:,:,:,0], dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc2_1(x))
        x = self.dropout2(x)
        x = self.fc2_2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x + co_output
        count += 1.
        outputs.append(x)
        
        x = temp[:,:,:,16:32]
        x = self.conv_block3_1(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block3_2(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block3_3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3_4(x, pool_size=(2, 2), pool_type='avg')
        (x, _) = torch.max(x[:,:,:,0], dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc3_1(x))
        x = self.dropout3(x)
        x = self.fc3_2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x + co_output
        count += 1.
        outputs.append(x)
        
        x = temp[:,:,:,24:40]
        x = self.conv_block4_1(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block4_2(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block4_3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4_4(x, pool_size=(2, 2), pool_type='avg')
        (x, _) = torch.max(x[:,:,:,0], dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc4_1(x))
        x = self.dropout4(x)
        x = self.fc4_2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x + co_output
        count += 1.
        outputs.append(x)
        
        outputs.append(co_output/count)
        outputs = torch.cat((outputs), 1)
        loss = torch.softmax(outputs,-1)
        if self.activation == 'logsoftmax':
            output = F.log_softmax(outputs, dim=-1)
            
        elif self.activation == 'sigmoid':
            output = torch.sigmoid(outputs)
            
        return output,loss
    
class Cqt_DCMF(nn.Module):
    
    def __init__(self, classes_num, activation):
        super(Cqt_DCMF, self).__init__()

        self.activation = activation
        self.spec_layer = Spec.CQT(sr=config.sample_rate, hop_length=config.hop_size, fmin=220, fmax=None, n_bins=64, bins_per_octave=12, norm=1, window='hann', center=True, pad_mode='reflect')
        self.conv_block1_1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block1_2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block1_3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block1_4 = ConvBlock(in_channels=256, out_channels=512)
        
        self.conv_block2_1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2_2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block2_3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block2_4 = ConvBlock(in_channels=256, out_channels=512)
                              
        self.conv_block3_1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block3_2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3_3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block3_4 = ConvBlock(in_channels=256, out_channels=512)
                              
        self.conv_block4_1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block4_2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block4_3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4_4 = ConvBlock(in_channels=256, out_channels=512)
        
        self.conv_block5_1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block5_2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block5_3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block5_4 = ConvBlock(in_channels=256, out_channels=512)
        
        self.conv_block6_1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block6_2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block6_3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block6_4 = ConvBlock(in_channels=256, out_channels=512)
        
        self.conv_block7_1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block7_2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block7_3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block7_4 = ConvBlock(in_channels=256, out_channels=512)
                              
                             
        self.fc1_1 = nn.Linear(512, 512, bias=True)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1_2 = nn.Linear(512, classes_num, bias=True)
        
        self.fc2_1 = nn.Linear(512, 512, bias=True)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2_2 = nn.Linear(512, classes_num, bias=True)
        
        self.fc3_1 = nn.Linear(512, 512, bias=True)
        self.dropout3 = nn.Dropout(p=0.5)
        self.fc3_2 = nn.Linear(512, classes_num, bias=True)
        
        self.fc4_1 = nn.Linear(512, 512, bias=True)
        self.dropout4 = nn.Dropout(p=0.5)
        self.fc4_2 = nn.Linear(512, classes_num, bias=True)
        
        self.fc5_1 = nn.Linear(512, 512, bias=True)
        self.dropout5 = nn.Dropout(p=0.5)
        self.fc5_2 = nn.Linear(512, classes_num, bias=True)
        
        self.fc6_1 = nn.Linear(512, 512, bias=True)
        self.dropout6 = nn.Dropout(p=0.5)
        self.fc6_2 = nn.Linear(512, classes_num, bias=True)
        
        self.fc7_1 = nn.Linear(512, 512, bias=True)
        self.dropout7 = nn.Dropout(p=0.5)
        self.fc7_2 = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):

        init_layer(self.fc1_1)
        init_layer(self.fc1_2)
        init_layer(self.fc2_1)
        init_layer(self.fc2_2)
        init_layer(self.fc3_1)
        init_layer(self.fc3_2)
        init_layer(self.fc4_1)
        init_layer(self.fc4_2)
        init_layer(self.fc5_1)
        init_layer(self.fc5_2)
        init_layer(self.fc6_1)
        init_layer(self.fc6_2)
        init_layer(self.fc7_1)
        init_layer(self.fc7_2)

    def forward(self, x1,x2,x3,x4):
        '''
        Input: (batch_size, scale_num, total_frames)'''
        outputs = []
        
        x = input
        x = self.spec_layer(x1)
        x = 10*torch.log10(x)
        x = torch.clamp(x, min=-100.)
        x = x.transpose(1,2)
        x = x[:,None,:,:]
        '''(batch_size, 1, times_steps, freq_bins)'''
        temp = x.clone()
        count = 0.
        
        x = temp[:,:,:,0:16]
        x = self.conv_block1_1(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block1_2(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block1_3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block1_4(x, pool_size=(2, 2), pool_type='avg')
        '''(batch_size, feature_maps, time_steps, freq_bins)'''
        (x, _) = torch.max(x[:,:,:,0], dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc1_1(x))
        x = self.dropout1(x)
        x = self.fc1_2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x
        count += 1.
        outputs.append(x)
        
        
        x = temp[:,:,:,8:24]
        x = self.conv_block2_1(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block2_2(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block2_3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2_4(x, pool_size=(2, 2), pool_type='avg')
        (x, _) = torch.max(x[:,:,:,0], dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc2_1(x))
        x = self.dropout2(x)
        x = self.fc2_2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x + co_output
        count += 1.
        outputs.append(x)
        
        x = temp[:,:,:,16:32]
        x = self.conv_block3_1(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block3_2(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block3_3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3_4(x, pool_size=(2, 2), pool_type='avg')
        (x, _) = torch.max(x[:,:,:,0], dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc3_1(x))
        x = self.dropout3(x)
        x = self.fc3_2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x + co_output
        count += 1.
        outputs.append(x)
        
        x = temp[:,:,:,24:40]
        x = self.conv_block4_1(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block4_2(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block4_3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4_4(x, pool_size=(2, 2), pool_type='avg')
        (x, _) = torch.max(x[:,:,:,0], dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc4_1(x))
        x = self.dropout4(x)
        x = self.fc4_2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x + co_output
        count += 1.
        outputs.append(x)
        
        x = temp[:,:,:,32:48]
        x = self.conv_block5_1(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block5_2(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block5_3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block5_4(x, pool_size=(2, 2), pool_type='avg')
        (x, _) = torch.max(x[:,:,:,0], dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc5_1(x))
        x = self.dropout5(x)
        x = self.fc5_2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x + co_output
        count += 1.
        outputs.append(x)
        
        x = temp[:,:,:,40:56]
        x = self.conv_block6_1(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block6_2(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block6_3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block6_4(x, pool_size=(2, 2), pool_type='avg')
        (x, _) = torch.max(x[:,:,:,0], dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc6_1(x))
        x = self.dropout6(x)
        x = self.fc6_2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x + co_output
        count += 1.
        outputs.append(x)
        
        x = temp[:,:,:,48:64]
        x = self.conv_block7_1(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block7_2(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block7_3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block7_4(x, pool_size=(2, 2), pool_type='avg')
        (x, _) = torch.max(x[:,:,:,0], dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc7_1(x))
        x = self.dropout7(x)
        x = self.fc7_2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x + co_output
        count += 1.
        outputs.append(x)
        
        outputs.append(co_output/count)
        outputs = torch.cat((outputs), 1)
        loss = torch.softmax(outputs,-1)
        if self.activation == 'logsoftmax':
            output = F.log_softmax(outputs, dim=-1)
            
        elif self.activation == 'sigmoid':
            output = torch.sigmoid(outputs)
            
        return output,loss

class Gamm_DCMF(nn.Module):
    
    def __init__(self, classes_num, activation, fixed=False):
        super(Gamm_DCMF, self).__init__()

        self.activation = activation
        self.conv_block1_1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block1_2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block1_3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block1_4 = ConvBlock(in_channels=256, out_channels=512)
        
        self.conv_block2_1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2_2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block2_3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block2_4 = ConvBlock(in_channels=256, out_channels=512)
                              
        self.conv_block3_1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block3_2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3_3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block3_4 = ConvBlock(in_channels=256, out_channels=512)
                              
        self.conv_block4_1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block4_2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block4_3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4_4 = ConvBlock(in_channels=256, out_channels=512)
        
        self.conv_block5_1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block5_2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block5_3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block5_4 = ConvBlock(in_channels=256, out_channels=512)
        
        self.conv_block6_1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block6_2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block6_3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block6_4 = ConvBlock(in_channels=256, out_channels=512)
        
        self.conv_block7_1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block7_2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block7_3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block7_4 = ConvBlock(in_channels=256, out_channels=512)
                              
                             
        self.fc1_1 = nn.Linear(512, 512, bias=True)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1_2 = nn.Linear(512, classes_num, bias=True)
        
        self.fc2_1 = nn.Linear(512, 512, bias=True)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2_2 = nn.Linear(512, classes_num, bias=True)
        
        self.fc3_1 = nn.Linear(512, 512, bias=True)
        self.dropout3 = nn.Dropout(p=0.5)
        self.fc3_2 = nn.Linear(512, classes_num, bias=True)
        
        self.fc4_1 = nn.Linear(512, 512, bias=True)
        self.dropout4 = nn.Dropout(p=0.5)
        self.fc4_2 = nn.Linear(512, classes_num, bias=True)
        
        self.fc5_1 = nn.Linear(512, 512, bias=True)
        self.dropout5 = nn.Dropout(p=0.5)
        self.fc5_2 = nn.Linear(512, classes_num, bias=True)
        
        self.fc6_1 = nn.Linear(512, 512, bias=True)
        self.dropout6 = nn.Dropout(p=0.5)
        self.fc6_2 = nn.Linear(512, classes_num, bias=True)
        
        self.fc7_1 = nn.Linear(512, 512, bias=True)
        self.dropout7 = nn.Dropout(p=0.5)
        self.fc7_2 = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):

        init_layer(self.fc1_1)
        init_layer(self.fc1_2)
        init_layer(self.fc2_1)
        init_layer(self.fc2_2)
        init_layer(self.fc3_1)
        init_layer(self.fc3_2)
        init_layer(self.fc4_1)
        init_layer(self.fc4_2)
        init_layer(self.fc5_1)
        init_layer(self.fc5_2)
        init_layer(self.fc6_1)
        init_layer(self.fc6_2)
        init_layer(self.fc7_1)
        init_layer(self.fc7_2)

    def forward(self, x1,x2,x3,x4):
        '''
        Input: (batch_size, scale_num, total_frames)'''
        outputs = []
        
        x = x2
        x = x[:,None,:,:]
        '''(batch_size, 1, times_steps, freq_bins)'''
        temp = x.clone()
        count = 0.
        
        x = temp[:,:,:,0:16]
        x = self.conv_block1_1(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block1_2(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block1_3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block1_4(x, pool_size=(2, 2), pool_type='avg')
        '''(batch_size, feature_maps, time_steps, freq_bins)'''
        (x, _) = torch.max(x[:,:,:,0], dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc1_1(x))
        x = self.dropout1(x)
        x = self.fc1_2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x
        count += 1.
        outputs.append(x)
        
        
        x = temp[:,:,:,8:24]
        x = self.conv_block2_1(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block2_2(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block2_3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2_4(x, pool_size=(2, 2), pool_type='avg')
        (x, _) = torch.max(x[:,:,:,0], dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc2_1(x))
        x = self.dropout2(x)
        x = self.fc2_2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x + co_output
        count += 1.
        outputs.append(x)
        
        x = temp[:,:,:,16:32]
        x = self.conv_block3_1(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block3_2(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block3_3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3_4(x, pool_size=(2, 2), pool_type='avg')
        (x, _) = torch.max(x[:,:,:,0], dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc3_1(x))
        x = self.dropout3(x)
        x = self.fc3_2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x + co_output
        count += 1.
        outputs.append(x)
        
        x = temp[:,:,:,24:40]
        x = self.conv_block4_1(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block4_2(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block4_3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4_4(x, pool_size=(2, 2), pool_type='avg')
        (x, _) = torch.max(x[:,:,:,0], dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc4_1(x))
        x = self.dropout4(x)
        x = self.fc4_2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x + co_output
        count += 1.
        outputs.append(x)
        
        x = temp[:,:,:,32:48]
        x = self.conv_block5_1(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block5_2(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block5_3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block5_4(x, pool_size=(2, 2), pool_type='avg')
        (x, _) = torch.max(x[:,:,:,0], dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc5_1(x))
        x = self.dropout5(x)
        x = self.fc5_2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x + co_output
        count += 1.
        outputs.append(x)
        
        x = temp[:,:,:,40:56]
        x = self.conv_block6_1(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block6_2(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block6_3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block6_4(x, pool_size=(2, 2), pool_type='avg')
        (x, _) = torch.max(x[:,:,:,0], dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc6_1(x))
        x = self.dropout6(x)
        x = self.fc6_2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x + co_output
        count += 1.
        outputs.append(x)
        
        x = temp[:,:,:,48:64]
        x = self.conv_block7_1(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block7_2(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block7_3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block7_4(x, pool_size=(2, 2), pool_type='avg')
        (x, _) = torch.max(x[:,:,:,0], dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc7_1(x))
        x = self.dropout7(x)
        x = self.fc7_2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x + co_output
        count += 1.
        outputs.append(x)
        
        outputs.append(co_output/count)
        outputs = torch.cat((outputs), 1)
        loss = torch.softmax(outputs,-1)
        if self.activation == 'logsoftmax':
            output = F.log_softmax(outputs, dim=-1)
            
        elif self.activation == 'sigmoid':
            output = torch.sigmoid(outputs)
            
        return output,loss
    
class Logmel_DCMT(nn.Module):
    
    def __init__(self, classes_num, activation):
        super(Logmel_DCMT, self).__init__()

        self.activation = activation
        self.spec_layer = Spec.MelSpectrogram(sr=config.sample_rate, n_fft=config.window_size, n_mels=config.mel_bins, hop_length=config.hop_size, window='hann', center=True, pad_mode='reflect', htk=False, fmin=0.0, fmax=None, norm=1, trainable_mel=False, trainable_STFT=False)
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.fc1 = nn.Linear(512, 512, bias=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):

        init_layer(self.fc1)
        init_layer(self.fc2)

    def forward(self, x1,x2,x3,x4):
        '''
        Input: (batch_size, scale_num, total_frames)'''
        outputs = []
        
        x = input
        x = self.spec_layer(x1)
        x = 10*torch.log10(x)
        x = torch.clamp(x, min=-100.)
        x = x.transpose(1,2)
        x = x[:,None,:,:]
        x = self.conv_block1(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')   
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg') 
        x = torch.mean(x, dim=3)
        temp = x.clone()
        
        count = 0.
        x = temp[:,:,0]
        x = F.relu_(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x
        count += 1.
        outputs.append(x)
        
        for i in range(1,temp.shape[2],1):
            x = temp[:,:,i]
            x = F.relu_(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            x = x.view(x.shape[0], 1, x.shape[1])
            co_output = x + co_output
            count += 1.
            outputs.append(x)

        outputs.append(co_output/count)
        outputs = torch.cat((outputs), 1)
        loss = torch.softmax(outputs,-1)
        if self.activation == 'logsoftmax':
            output = F.log_softmax(outputs, dim=-1)
            
        elif self.activation == 'sigmoid':
            output = torch.sigmoid(outputs)
            
        return output,loss

class Cqt_DCMT(nn.Module):
    
    def __init__(self, classes_num, activation):
        super(Cqt_DCMT, self).__init__()

        self.activation = activation
        self.spec_layer = Spec.CQT(sr=config.sample_rate, hop_length=config.hop_size, fmin=220, fmax=None, n_bins=64, bins_per_octave=12, norm=1, window='hann', center=True, pad_mode='reflect')
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.fc1 = nn.Linear(512, 512, bias=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):

        init_layer(self.fc1)
        init_layer(self.fc2)

    def forward(self, x1,x2,x3,x4):
        '''
        Input: (batch_size, scale_num, total_frames)'''
        outputs = []
        
        x = input
        x = self.spec_layer(x1)
        x = 10*torch.log10(x)
        x = torch.clamp(x, min=-100.)
        x = x.transpose(1,2)
        x = x[:,None,:,:]
        x = self.conv_block1(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')   
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg') 
        x = torch.mean(x, dim=3)
        temp = x.clone()
        
        count = 0.
        x = temp[:,:,0]
        x = F.relu_(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x
        count += 1.
        outputs.append(x)
        
        for i in range(1,temp.shape[2],1):
            x = temp[:,:,i]
            x = F.relu_(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            x = x.view(x.shape[0], 1, x.shape[1])
            co_output = x + co_output
            count += 1.
            outputs.append(x)

        outputs.append(co_output/count)
        outputs = torch.cat((outputs), 1)
        loss = torch.softmax(outputs,-1)
        if self.activation == 'logsoftmax':
            output = F.log_softmax(outputs, dim=-1)
            
        elif self.activation == 'sigmoid':
            output = torch.sigmoid(outputs)
            
        return output,loss
    
class Gamm_DCMT(nn.Module):
    
    def __init__(self, classes_num, activation):
        super(Gamm_DCMT, self).__init__()

        self.activation = activation
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.fc1 = nn.Linear(512, 512, bias=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):

        init_layer(self.fc1)
        init_layer(self.fc2)

    def forward(self, x1,x2,x3,x4):
        '''
        Input: (batch_size, scale_num, total_frames)'''
        outputs = []
        
        x = x2
        x = x[:,None,:,:]
        x = self.conv_block1(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')   
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg') 
        x = torch.mean(x, dim=3)
        temp = x.clone()
        
        count = 0.
        x = temp[:,:,0]
        x = F.relu_(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x
        count += 1.
        outputs.append(x)
        
        for i in range(1,temp.shape[2],1):
            x = temp[:,:,i]
            x = F.relu_(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            x = x.view(x.shape[0], 1, x.shape[1])
            co_output = x + co_output
            count += 1.
            outputs.append(x)

        outputs.append(co_output/count)
        outputs = torch.cat((outputs), 1)
        loss = torch.softmax(outputs,-1)
        if self.activation == 'logsoftmax':
            output = F.log_softmax(outputs, dim=-1)
            
        elif self.activation == 'sigmoid':
            output = torch.sigmoid(outputs)
            
        return output,loss
    
class Mfcc_DCMT(nn.Module):
    
    def __init__(self, classes_num, activation):
        super(Mfcc_DCMT, self).__init__()

        self.activation = activation
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.fc1 = nn.Linear(512, 512, bias=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):

        init_layer(self.fc1)
        init_layer(self.fc2)

    def forward(self, x1,x2,x3,x4):
        '''
        Input: (batch_size, scale_num, total_frames)'''
        outputs = []
        
        x = x3
        x = x[:,None,:,:]
        x = self.conv_block1(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')   
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg') 
        x = torch.mean(x, dim=3)
        temp = x.clone()
        
        count = 0.
        x = temp[:,:,0]
        x = F.relu_(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x
        count += 1.
        outputs.append(x)
        
        for i in range(1,temp.shape[2],1):
            x = temp[:,:,i]
            x = F.relu_(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            x = x.view(x.shape[0], 1, x.shape[1])
            co_output = x + co_output
            count += 1.
            outputs.append(x)

        outputs.append(co_output/count)
        outputs = torch.cat((outputs), 1)
        loss = torch.softmax(outputs,-1)
        if self.activation == 'logsoftmax':
            output = F.log_softmax(outputs, dim=-1)
            
        elif self.activation == 'sigmoid':
            output = torch.sigmoid(outputs)
            
        return output,loss

class DCMR_DCMT(nn.Module):
    
    def __init__(self, classes_num, activation):
        super(DCMR_DCMT, self).__init__()

        self.activation = activation
        self.model1 = Logmel_DCMT(classes_num, activation)
        self.model2 = Cqt_DCMT(classes_num, activation)
        self.model3 = Gamm_MultiFrames_CNN(classes_num, activation)
        self.model4 = Mfcc_MultiFrames_CNN(classes_num, activation)

        self.ite = 15000  # set your pre-trained model iteration
        self.model_path = 'workspace/checkpoints/main_eva/logmel_86frames_40melbins/TAU-urban-acoustic-scenes-2020-mobile-development/holdout_fold=1/'  # set your model saved path.
        self.init_weights()

    def init_weights(self):
        model_path1 = self.model_path +'Logmel_DCMT/'+str(self.ite)+'_iterations.pth'
        model_path2 = self.model_path +'Cqt_DCMT/'+str(self.ite)+'_iterations.pth'
        model_path3 = self.model_path +'Gamm_DCMT/'+str(self.ite)+'_iterations.pth'
        model_path4 = self.model_path +'Mfcc_DCMT/'+str(self.ite)+'_iterations.pth'

        model_ch1 = torch.load(model_path1)
        model_ch2 = torch.load(model_path2)
        model_ch3 = torch.load(model_path3)
        model_ch4 = torch.load(model_path4)
        
        self.model1.load_state_dict(model_ch1['model'])
        self.model2.load_state_dict(model_ch2['model'])
        self.model3.load_state_dict(model_ch3['model'])
        self.model4.load_state_dict(model_ch4['model'])
       

    def forward(self, input1,input2,input3,input4):
        '''
        Input: (batch_size, total_frames)'''
        
        x1,loss1 = self.model1(input1,input2,input3,input4)
        x2,loss2 = self.model2(input1,input2,input3,input4)
        x3,loss3 = self.model3(input1,input2,input3,input4)
        x4,loss4 = self.model4(input1,input2,input3,input4)
        
        x = x1[:,-1] + x2[:,-1] + x3[:,-1] + x4[:,-1]
        loss = (loss1[:,-1]+loss2[:,-1]+loss3[:,-1]+loss4[:,-1])/4.
        return x,loss

class DCMR_DCMF(nn.Module):
    
    def __init__(self, classes_num, activation):
        super(Ensemble_CNN6, self).__init__()

        self.activation = activation
        self.model1 = Logmel_DCMF(classes_num, activation)
        self.model2 = Cqt_DCMF(classes_num, activation)
        self.model3 = Gamm_DCMF(classes_num, activation)
        self.model4 = Mfcc_DCMF(classes_num, activation)
        self.ite = 15000  # set your pre-trained model iteration
        self.model_path = 'workspace/checkpoints/main_eva/logmel_86frames_40melbins/TAU-urban-acoustic-scenes-2020-mobile-development/holdout_fold=1/'  # set your model saved path.
        self.init_weights()

    def init_weights(self):
        model_path1 = self.model_path +'Logmel_DCMF/'+str(self.ite)+'_iterations.pth'
        model_path2 = self.model_path +'Cqt_DCMF/'+str(self.ite)+'_iterations.pth'
        model_path3 = self.model_path +'Gamm_DCMF/'+str(self.ite)+'_iterations.pth'
        model_path4 = self.model_path +'Mfcc_DCMF/'+str(self.ite)+'_iterations.pth'

        model_ch1 = torch.load(model_path1)
        model_ch2 = torch.load(model_path2)
        model_ch3 = torch.load(model_path3)
        model_ch4 = torch.load(model_path4)
        
        self.model1.load_state_dict(model_ch1['model'])
        self.model2.load_state_dict(model_ch2['model'])
        self.model3.load_state_dict(model_ch3['model'])
        self.model4.load_state_dict(model_ch4['model'])
       

    def forward(self, input1,input2,input3,input4):
        '''
        Input: (batch_size, total_frames)'''
        
        x1,loss1 = self.model1(input1,input2,input3,input4)
        x2,loss2 = self.model2(input1,input2,input3,input4)
        x3,loss3 = self.model3(input1,input2,input3,input4)
        x4,loss4 = self.model4(input1,input2,input3,input4)
        
        x = x1[:,-1] + x2[:,-1] + x3[:,-1] + x4[:,-1]
        loss = (loss1[:,-1]+loss2[:,-1]+loss3[:,-1]+loss4[:,-1])/4.
        return x,loss
    
class DCMR_DCMF_DCMT(nn.Module):
    
    def __init__(self, classes_num, activation):
        super(DCMR_DCMF_DCMT, self).__init__()

        self.activation = activation
        self.model1 = DCMR_DCMT(classes_num, activation)
        self.model2 = DCMR_DCMF(classes_num, activation)

    def forward(self, input1,input2,input3,input4):
        '''
        Input: (batch_size, total_frames)'''
        
        x1,loss1 = self.model1(input1,input2,input3,input4)
        x2,loss2 = self.model2(input1,input2,input3,input4)

        x = x1 + x2
        loss = (loss1+loss2)/2.

        return x,loss  
