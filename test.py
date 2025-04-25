import torch
import numpy as np
import os
import pandas as pd
from collections import defaultdict

import pickle
import time
import argparse

import sys
sys.path.append("..")

from slope import cal_slope
from avg_fraction_edge_mnist import avg_f_cal
from Lidar.draw_traj import lidar_draw
from Lidar.avg_fraction_edge import avg_statics_lidar


import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")


'''
parameters:
    @ q_NGR, q_INV, q_EXP
    @ model type/name: fc, cnn
    @ model path
    @ result path
    @ example num

'''
def parse_args():
    parse = argparse.ArgumentParser(description='Neural Data Graph')
    parse.add_argument('--image', type=int, default=1, required=False, help='If test Image')
    parse.add_argument('--lidar', type=int, default=0, required=False, help='If test LiDAR')
    parse.add_argument('--cifar', type=str, default='small', required=False, help='Small or big model for CIFAR')
    parse.add_argument('--metric', type=str, required=True, help='Definition of NDG')
    parse.add_argument('--model_type', type=str, default='fc', required=False, help='Type of test model')
    parse.add_argument('--dataset', type=str, default='mnist', required=False, help='Dataset')
    parse.add_argument('--model_name', type=str, default='ori', required=False, help='Name of test model')
    parse.add_argument('--model_path', type=str, default='pgd/models/', required=False, help='Path of test model')
    parse.add_argument('--mnist_res_path', type=str, default='./', required=False, help='Result path')
    parse.add_argument('--lidar_res_path', type=str, default='./', required=False, help='Result path')
    parse.add_argument('--mnist_data_path', type=str, required=False, help='Data path')
    parse.add_argument('--lidar_data_path', type=str, required=False, help='Data path')
    parse.add_argument('--sample_num', type=int, default=50, required=False, help='Number of test examples')
    args = parse.parse_args() 
    return args




if __name__=='__main__':
    args = parse_args()
    
    if args.image:
        # print(f'Start calculate slopes...\n')
        # cal_slope(args)
        
        print(f'Start calculate average fraction per label...\n')
        avg_f_cal(args)
        
    if args.lidar:
        # lidar_draw(args)
        avg_statics_lidar(args)
        