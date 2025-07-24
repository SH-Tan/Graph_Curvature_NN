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

from pgd.remove_edge_fc import remove_edge_fc
from pgd.remove_edge_fc_perE import remove_edge_fc_perE
from pgd.remove_edge_fc_perE_vec import remove_edge_fc_perE_vec
from pgd.remove_edge_fc_perE_alllabel import remove_edge_fc_perE_alllabels
from pgd.remove_edge_fc_union import remove_edge_fc_union
from pgd.remove_edge_fc_union_pgd import remove_edge_fc_union_pgd
from pgd.remove_edge_union_fcold import remove_edge_fc_union_old
from pgd.remove_weights_fc import remove_w_fc
from pgd.remove_edge_fc_union_perlayer import remove_edge_fc_union_perlayer
from pgd.remove_edge_fc_perlayer_old import remove_edge_fc_union_perlayer_old
from CNN.remove_edge_cnn_union import remove_edge_cnn_union
from CNN.remove_weights_cnn import remove_w_cnn
from CNN.remove_edge_cnn_union_perlayer import remove_edge_cnn_union_perlayer
from CNN.remove_edge_cifar_union import remove_edge_cifar_union
from CNN.remove_edge_cifar_union_perlayer import remove_edge_cifar_union_perlayer
from CNN.remove_weights_cifar import remove_w_cifar
from pgd.community_check_fc import community_check_fc
from CNN.community_check_cnn import community_check_cnn
from CNN.community_check_cifar import community_check_cifar
from Lidar.plot_trajectories import main_lidar
from Lidar.plot_trajectories_perlayer import main_lidar_perlayer
from Lidar.remove_weights_fc import remove_w_lidar

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
        if model_type.lower() == "fc":
            if args.edge:
                remove_w_fc(args)
            if args.community:
                community_check_fc(args)
        elif model_type.lower() == "cnn":
            if args.edge and args.dataset.lower() == "mnist":
                remove_w_cnn(args)
            if args.community and args.dataset.lower() == "mnist":
                community_check_cnn(args)
            if args.edge and args.dataset.lower() == "cifar":
                # remove_edge_cifar_union(args)
                remove_w_cifar(args)
            if args.community and args.dataset.lower() == "cifar":
                community_check_cifar(args)
        else:
            raise Exception("Invalid model type, model type should be {fc, fc_linear, cnn}!")
    
    if args.lidar:
        main_lidar(args)
    
        