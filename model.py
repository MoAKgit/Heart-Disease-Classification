# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 07:39:07 2022

@author: Mohammad
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import random


class CNNLSTM(nn.Module):
    def __init__(self, args):
        super().__init__()
        input_features = args.num_featrues
        num_layers = 1
        hidden_size = 100
        
        self.conv1 = nn.Conv1d(in_channels = 1, 
                               out_channels = 32, 
                               kernel_size = 3)
        
        self.lstm = nn.LSTM(input_size = 32,
                            num_layers = num_layers,
                            hidden_size = hidden_size,
                            proj_size  = 1,
                            batch_first=True)
        self.linear1 = nn.Linear(in_features= args.num_featrues, 
                                out_features= int(args.num_featrues/2))
        
        self.linear2 = nn.Linear(in_features= int(args.num_featrues/2), 
                                out_features= int(args.num_featrues/4))
        
        self.linear3 = nn.Linear(in_features= int(args.num_featrues/4), 
                                out_features= args.num_classes)
        self.drop_out1 = nn.Dropout(0.2)
        self.drop_out2 = nn.Dropout(0.15)
        self.softmax  = nn.Softmax(dim=1)
    def forward (self, x):
        # print("11111111111111", x.shape)
        x = self.conv1(x)
        x = x.permute(0,2,1)
        x,_= self.lstm(x)
        x = torch.squeeze(x, 2)
        # print("33333333333", x.shape)
        x = self.linear1(x)
        x = self.drop_out1(x)
        x = self.linear2(x)
        x = self.drop_out2(x)
        x = self.linear3(x)
        x = self.softmax(x)
        # print("44444444444", x.shape)
        return x
    
class CNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        input_features = args.num_featrues
        num_layers = 1
        hidden_size = 100
        
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels = 1, out_channels = 32, kernel_size = 7,
                                                               padding= 1,
                                                               stride = 2),
            nn.BatchNorm1d(32),
            nn.ReLU()
            )
        
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels = 32, out_channels = 64, kernel_size = 7,
                                                               padding= 1,
                                                               stride = 2),
            # nn.Dropout(0.2),
            nn.BatchNorm1d(64),
            nn.ReLU()
            )
        
        self.layer3 = nn.Sequential(
            nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 7,
                                                               padding= 1,
                                                               stride = 2),
            # nn.Dropout(0.2),
            nn.BatchNorm1d(64),
            nn.ReLU()
            )
        self.layer4 = nn.Sequential(
            nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 7,
                                                               padding= 1,
                                                               stride = 2),
            nn.Dropout(0.4),
            nn.BatchNorm1d(64),
            nn.ReLU()
            )
        
        self.linear1 = nn.Linear(in_features= 64*8, 
                                out_features= 300)
        self.linear2 = nn.Linear(in_features= 300, 
                                out_features= 100)
        self.linear3 = nn.Linear(in_features= 100, 
                                out_features= args.num_classes)
        self.relu = nn.ReLU()
        self.softmax  = nn.Softmax(dim=1)
        
    def forward (self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        s = x.shape 
        x = x.view(s[0], -1)
        
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.softmax(x)
        return x
      
    
    
    
