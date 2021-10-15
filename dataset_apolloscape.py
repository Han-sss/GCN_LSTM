from torch.utils import data
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import os
import torch
from torchvision import transforms
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle            

# dir = '/home/mount/GCN-lstm/data/Aplloscape'
# DATA_DIR = dir + '/prediction_train/'
# DATA_DIR_TEST = dir + '/prediction_test/'
# total data number is 26909

class trajectory_dataset(Dataset): 
    def __init__(self, root_dir, datatype = 'train', transform=None):
        self.root_dir = root_dir   
        self.transform = transform
        self.datatype = datatype

        if datatype == 'train':
            train_input_dir = self.root_dir+"/prediction_train/formatted/train_input_sub.npy"
            train_result_dir = self.root_dir+"/prediction_train/formatted/train_result_sub.npy"
            print(train_input_dir)
            print(train_result_dir)
            self.input = np.load(train_input_dir, allow_pickle=True)
            self.result = np.load(train_result_dir, allow_pickle=True)
            
        elif datatype == 'test': 
            train_input_dir = self.root_dir+"/prediction_train/formatted/train_input.npy"
            self.input = np.load(train_input_dir, allow_pickle=True)

        self.input_len = 6
        self.node_feature_num = 10
        self.res_len = 6
        self.class_num = 5

    def __len__(self):
        return len(self.input)
    
    def __getitem__(self,index):
        ### 
        if self.datatype == 'train':
            return self.input[index], self.result[index]
            
        elif self.datatype == 'test':
            return self.input[index] 
