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

from pgd.single_ex_test import fc_main
from pgd.single_test_linear import fc_linear_main
from CNN.single_ex_test_small import cnn_main
from CNN.single_test_cifar_small import cifar_small_main
from CNN.single_test_cifar_big import cifar_big_main

from Lidar.LoadSampleScript import start_lidar

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
    parse.add_argument('--dataset', type=str, default='mnist', required=False, help='Dataset')
    parse.add_argument('--model_type', type=str, default='fc', required=False, help='Type of test model')
    parse.add_argument('--model_name', type=str, default='ori', required=False, help='Name of test model')
    parse.add_argument('--model_path', type=str, default='pgd/models/', required=False, help='Path of test model')
    parse.add_argument('--sample_num', type=int, default=50, required=False, help='Number of test examples')
    parse.add_argument('--mnist_res_path', type=str, default='./', required=False, help='Result path')
    parse.add_argument('--lidar_res_path', type=str, default='./', required=False, help='Result path')
    args = parse.parse_args() 
    return args




if __name__=='__main__':
    args = parse_args()
    
    model_type = args.model_type
    
    if args.image:
        if model_type.lower() == "fc":
            fc_main(args)
        elif model_type.lower() == "fc_linear":
            fc_linear_main(args)
        elif model_type.lower() == "cnn" and args.dataset == 'mnist':
            cnn_main(args)
        elif model_type.lower() == "cnn" and args.dataset == 'cifar':
            if args.cifar.lower() == 'small':
                cifar_small_main(args)
            elif args.cifar.lower() == 'big':
                cifar_big_main(args)
            else:
                raise Exception("Invalid CIFAR model type, model type should be {small, big}!")
        else:
            raise Exception("Invalid model type, model type should be {fc, fc_linear, cnn}!")
    
    if args.lidar:
        start_lidar(args)
    
        