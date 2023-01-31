# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 07:22:39 2022

@author: Mohammad
"""

import numpy as np
import torch
import pandas as pd
import argparse
from model import *
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import dataset
from utils import *
import glob
import os
from tqdm import tqdm

from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE


from sklearn.utils import resample
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    parser = argparse.ArgumentParser(description='CNN_lstm')
    parser.add_argument("--epochs",dest= 'epochs', default= 250) 
    parser.add_argument("--input_channels",dest= 'input_channels', default= 3)

    parser.add_argument("--hidden_size",dest= 'hidden_size', default= 100)
    parser.add_argument("--batch_size",dest= 'batch_size', default= 128)
    parser.add_argument("--learning_rate",dest= 'learning_rate', default= 0.00001)

    parser.add_argument("--train_path",dest= 'train_path', default= 'data/mitbih_train.csv')
    parser.add_argument("--test_path",dest= 'test_path', default= 'data/mitbih_test.csv')

    parser.add_argument("--num_classes",dest= 'num_classes', default= 5)
    parser.add_argument("--num_featrues",dest= 'num_featrues', default= 187)
    
    return parser.parse_args()

if __name__ == '__main__':
    print('Main')
    args = arg_parse()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    ###########################
    data_train = np.asarray(pd.read_csv(args.train_path))
    
    
    df_0 = data_train[data_train[:,187] == 0]
    df_1 = data_train[data_train[:,187] == 1]
    df_2 = data_train[data_train[:,187] == 2]
    df_3 = data_train[data_train[:,187] == 3]
    df_4 = data_train[data_train[:,187] == 4]
    
    
    print(df_0.shape)
    print(df_1.shape)
    print(df_2.shape)
    print(df_3.shape)
    print(df_4.shape)
    
    # df_0_upsample = resample(df_0, n_samples = 20000, replace = True, random_state = 123)
    df_1_upsample = resample(df_1, n_samples = 72470, replace = True, random_state = 123)
    df_2_upsample = resample(df_2, n_samples = 72470, replace = True, random_state = 123)
    df_3_upsample = resample(df_3, n_samples = 72470, replace = True, random_state = 123)
    df_4_upsample = resample(df_4, n_samples = 72470, replace = True, random_state = 123)
    
    
    df_1_upsample = np.asarray(df_1_upsample)
    df_2_upsample = np.asarray(df_2_upsample)
    df_3_upsample = np.asarray(df_3_upsample)
    df_4_upsample = np.asarray(df_4_upsample)
    
    data_train = np.concatenate((df_0, 
                                 df_1_upsample, 
                                 df_2_upsample, 
                                 df_3_upsample, 
                                 df_4_upsample))
    data_train[:,187] = data_train[:,187].astype('int') 
    

    data_test = np.asarray(pd.read_csv(args.test_path))
    
    net = CNN(args).to(device)
    
    
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    criterian = nn.CrossEntropyLoss()

    loss_history = []
    
    for epoch in range(args.epochs):
        Loss_epoch = [] 
        Acc = []
        
        random.shuffle(data_train)
        max_itr = int(len(data_train) / args.batch_size - 1)
        for i in tqdm(range(max_itr)):
            
            net.train()
            
            batch_data = data_train[ i*args.batch_size : (i+1)*args.batch_size, :]
            
            batch_input = batch_data[:, :-1]
            batch_label = batch_data[:, -1]
            batch_input = np.asarray(batch_input, dtype= 'float32' )
            batch_input = np.expand_dims(batch_input, 1)
            batch_input = torch.from_numpy(batch_input).to(device)
            batch_label = np.asarray(batch_label, dtype= 'int64' )
            batch_label = torch.from_numpy(batch_label).to(device)

            pred = net(batch_input)
            
            loss = criterian(pred, batch_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            Loss_epoch.append(loss.cpu().detach().numpy())
            
            
            
            winners = pred.argmax(dim=1)
            corrects = (winners == batch_label)
            accuracy = corrects.sum().float() / float( batch_label.size(0) )
            Acc.append(accuracy.cpu().detach().numpy())
        
        
        Loss_epoch = np.asarray(Loss_epoch)
        loss_history.append(Loss_epoch.mean())
        
        Acc = np.asarray(Acc)
        
        
        if epoch % 1 == 0 and epoch > 0:
            acc_test_avr = test(args, net , device)
            print("accuracy: ", Acc.mean(), acc_test_avr)
            
            
            
            
            
            
            

