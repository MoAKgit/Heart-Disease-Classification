# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 15:45:30 2022

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




def test(args, model, device):
    Pred = []
    target_label = []
    avr = 0
    data_test = np.asarray(pd.read_csv(args.test_path))
    # print(data_test.shape)
    data_test[:,187] = data_test[:,187].astype('int') 
    for i, sample in enumerate(data_test):
        batch_input = data_test[i, :-1]
        batch_label = data_test[i, -1]
        
        batch_input = np.expand_dims(batch_input, 0)
        
        batch_input = np.asarray(batch_input, dtype= 'float32' )
        batch_input = np.expand_dims(batch_input, 0)
        batch_input = torch.from_numpy(batch_input).to(device)
        batch_label = np.asarray(batch_label, dtype= 'int64' )

        
        pred = model(batch_input)
        pred = pred.cpu().detach().numpy()
        if len(Pred) == 0:
            Pred = pred
        else:
            Pred = np.concatenate(( Pred, pred), 0)
            
        
        target_label.append(batch_label)
    
    target_label = np.asarray(target_label)
    winners = np.argmax(Pred, axis=1)
    corrects = (winners == target_label)
    
    accuracy = corrects.sum() / len(target_label) 
    return accuracy



