import torch
from torchvision.datasets.cifar import CIFAR10
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
import torchvision
import numpy as np
import random
import os
import pandas as pd
import torch.nn as nn
from collections import defaultdict
import copy

import pickle
import time
import pandas as pd


import sys
sys.path.append("..")

# from tools.vgg16_custom_relu import VGG16_CIFAR10
from tools.vgg16_custom_tanh import VGG16_CIFAR10
from tools.controller import Controller

np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=torch.inf)

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")



model_dims = {
    1: {"name": "input", "dim": {"channel": 3, "out_size": 32}},   # Input image

    2: {"name": "cnn", "dim": {"channel": 64, "kernel": 3, "stride": 1, "padding":1, "out_size": 16}},   # After conv1_2 + pool
    3: {"name": "cnn", "dim": {"channel": 128, "kernel": 3, "stride": 1, "padding":1, "out_size": 8}},   # After conv2_2 + pool
    4: {"name": "cnn", "dim": {"channel": 256, "kernel": 3, "stride": 1, "padding":1, "out_size": 4}},   # After conv3_3 + pool
    5: {"name": "cnn", "dim": {"channel": 512, "kernel": 3, "stride": 1, "padding":1, "out_size": 2}},   # After conv4_3 + pool

    6: {"name": "cnn", "dim": {"channel": 512, "kernel": 3, "stride": 1, "padding":1, "out_size": 2}},   # conv5_1
    7: {"name": "cnn", "dim": {"channel": 512, "kernel": 3, "stride": 1, "padding":1, "pool":True, "out_size": 1}},   # conv5_2 + pool
    8: {"name": "cnn", "dim": {"channel": 512, "kernel": 3, "stride": 1, "padding":1, "pool":False, "out_size": 1}},   # conv5_3 

    9: {"name": "fc", "dim": {"out_size": 1024}},  # Flatten(512×1×1) → 1024
    10: {"name": "fc", "dim": {"out_size": 512}},
    11: {"name": "fc", "dim": {"out_size": 10}}
}



model_dims_small = {
    1: {"name": "cnn", "dim": {"channel": 512, "kernel": 3, "stride": 1, "padding":1, "out_size": 1}},   # conv5_2
    2: {"name": "cnn", "dim": {"channel": 512, "kernel": 3, "stride": 1, "padding":1, "pool":False, "out_size": 1}},   # conv5_3

    3: {"name": "fc", "dim": {"out_size": 1024}},  # Flatten(512×1×1) → 1024
    4: {"name": "fc", "dim": {"out_size": 512}},
    5: {"name": "fc", "dim": {"out_size": 10}}
}

selected_classes = [0,1,2,3,4,5,6,7,8,9]

model_zoo = {
    '64': [21, 64, 64, 1],
    '128': [21, 128, 128, 1]
}

if __name__ == '__main__':
    seed = 29
    
    # set random seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    
    model_name = "DDPG_64_C1.pth"
    model_path = "Lidar/models/"
    
    dims = model_zoo['64']
    net_H = Controller(dims, 2)
    # net_H = VGG16_CIFAR10(model_dims, None, device)
    net_H.load_state_dict(torch.load(model_path + model_name))
    net_H = net_H.to(device)
    
    for name, param in net_H.named_parameters():
        if param.requires_grad:
            print(f"Layer: {name} | Shape: {param.shape}")
            print(param.data)
            input()
            
    # print(net_H.conv5_3.weight.shape)  # e.g., torch.Size([512, 512, 3, 3])
    # print(net_H.conv5_3.weight)        # Full tensor

    # Convert to NumPy for analysis
    # weights = net_H.conv5_3.weight.data.cpu().numpy()
    # print(weights)
