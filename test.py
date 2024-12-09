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
from avg_fraction import avg_f_cal


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
    parse.add_argument('--metric', type=str, required=True, help='Definition of NDG')
    parse.add_argument('--model_type', type=str, required=True, help='Type of test model')
    parse.add_argument('--model_name', type=str, required=True, help='Name of test model')
    parse.add_argument('--model_path', type=str, required=True, help='Path of test model')
    parse.add_argument('--res_path', type=str, required=True, help='Result path')
    parse.add_argument('--data_path', type=str, required=True, help='Data path')
    parse.add_argument('--sample_num', type=int, default=50, required=False, help='Number of test examples')
    args = parse.parse_args() 
    return args




if __name__=='__main__':
    args = parse_args()
    
    print(f'Start calculate slopes...\n')
    cal_slope(args)
    
    print(f'Start calculate average fraction per label...\n')
    avg_f_cal(args)
        