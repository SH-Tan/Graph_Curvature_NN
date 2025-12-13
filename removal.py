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

from CNN.remove_edge_cnn_union_perlayer_w_combined import remove_edge_cnn_union_perlayer_w_combined
from CNN.remove_edge_cnn_union_w_combined import remove_edge_cnn_union_w_combined
from CNN.remove_edge_cifar_union_w_small import remove_edge_cifar_union_w_small
from CNN.remove_edge_cifar_union_w_small_combined import remove_edge_cifar_union_w_small_combined
from CNN.remove_edge_cifar100_union import remove_edge_cifar100_union
from CNN.remove_edge_cifar_union_perlayer_weight import remove_edge_cifar_union_perlayer_w
from CNN.remove_edge_cifar100_union_perlayer import remove_edge_cifar100_union_perlayer
from pgd.community_check_fc import community_check_fc
from CNN.community_check_cnn import community_check_cnn
from CNN.community_check_cifar import community_check_cifar
from CNN.community_check_cifar_new import community_check_cifar_new
from CNN.community_check_cifar_new_small import community_check_cifar_new_small
from CNN.community_check_cifar_vgg9 import community_check_cifar_vgg9
from CNN.remove_edge_cifar_union_perlayer_weight_small import remove_edge_cifar_union_perlayer_w_small
from CNN.remove_edge_cifar_union_perlayer_weight_small_combined import remove_edge_cifar_union_perlayer_w_small_combined
from CNN.community_check_cifar100 import community_check_cifar100
from CNN.remove_edge_cifar100_union_combined import remove_edge_cifar100_union_combined
from CNN.remove_edge_cifar100_union_perlayer_combined import remove_edge_cifar100_perlayer_w_small_combined
from Lidar.plot_trajectories import main_lidar
from Lidar.plot_trajectories_perlayer import main_lidar_perlayer
from Lidar.remove_weights_fc import remove_w_lidar
from CNN.process_and_save import process_data

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
    parse.add_argument('--alpha', type=float, default=0., required=False, help='Alpha used for distribution')
    parse.add_argument('--hops', type=int, default=1, required=False, help='Hops between nodes to calculate curvature')
    parse.add_argument('--dataset', type=str, default='mnist', required=False, help='Dataset')
    parse.add_argument('--model_type', type=str, default='fc', required=False, help='Type of test model')
    parse.add_argument('--model_name', type=str, default='ori', required=False, help='Name of test model')
    parse.add_argument('--model_path', type=str, default='pgd/models/', required=False, help='Path of test model')
    parse.add_argument('--sample_num', type=int, default=50, required=False, help='Number of test examples')
    parse.add_argument('--mnist_res_path', type=str, default='./', required=False, help='Result path')
    parse.add_argument('--lidar_res_path', type=str, default='./', required=False, help='Result path')
    parse.add_argument('--mnist_data_path', type=str, required=False, help='Data path')
    parse.add_argument('--lidar_data_path', type=str, required=False, help='Data path')
    parse.add_argument('--edge', type=int, default=0, required=False, help='If test edge')
    parse.add_argument('--node', type=int, default=0, required=False, help='If test node')
    parse.add_argument('--community', type=int, default=0, required=False, help='If test community')
    parse.add_argument('--activation', type=str, default="relu", required=False, help='Activation function')
    args = parse.parse_args() 
    return args




if __name__=='__main__':
    args = parse_args()
    
    model_type = args.model_type
    
    if args.image:
        if model_type.lower() == "cnn":
            if args.edge and args.dataset.lower() == "mnist":
                # remove_edge_cnn_union_perlayer_w_combined(args)
                remove_edge_cnn_union_w_combined(args)
            if args.community and args.dataset.lower() == "mnist":
                community_check_cnn(args)
            if args.edge and args.dataset.lower() == "cifar":
                # process_data(args)
                # remove_edge_cifar_union_perlayer_w(args)
                remove_edge_cifar_union_w_small_combined(args)
                # remove_edge_cifar_union_perlayer_w_small_combined(args)
                # remove_w_cifar100(args)
                # remove_edge_cifar100_perlayer_w_small_combined(args)
                # remove_edge_cifar100_union_combined(args)
                # remove_wadan_cifar(args)
                # remove_edge_cifar_union_w_vgg11(args)
            if args.community and args.dataset.lower() == "cifar":
                # community_check_cifar_new_small(args)
                community_check_cifar_vgg9(args)
                # community_check_cifar100(args)
                # iterative_edge_removal_and_curvature(args)
        else:
            raise Exception("Invalid model type, model type should be {fc, fc_linear, cnn}!")
    
    if args.lidar:
        remove_w_lidar(args)
    
        