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


class trajectory_dataset(Dataset): 
    def __init__(self, data_dir, datatype = 'train', transform=None):
        self.root_dir = data_dir   
        self.transform = transform 
        if datatype == 'train':
            s1 = open( self.root_dir + 'stream1_obs_data_train.pkl', 'rb')
            t1 = open( self.root_dir + 'stream1_pred_data_train.pkl', 'rb')

        elif datatype == 'test': 
        
    
    def __len__(self):
        return len(self.src1)
    
    def __getitem__(self,index):
        ### 

        return s1_input_tensor, s1_target_tensor, s2_input_tensor, s2_target_tensor, batch_agent_id, model_target
