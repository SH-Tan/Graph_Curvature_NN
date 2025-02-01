import torch
from torchvision.datasets.cifar import CIFAR10
import torchvision.transforms as transforms
import torchvision

import numpy as np
import random
import os
import pandas as pd
import torch.nn as nn
from collections import defaultdict
import time

# from GraphRicciCurvature.OllivierRicci import OllivierRicci
import networkx as nx
import torch.nn.functional as F
import pickle

import sys
sys.path.append("..")

import tools.utils as utils
from RicciCurvature.OllivierRicci import OllivierRicci
from tools.LeNet5_custom import LeNet_custom
from tools.graph_curvature import graph_curvature_main_torch


np.set_printoptions(threshold=np.inf)

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")


transform_train = torchvision.transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(size=32, padding=4),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = torchvision.transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

data_train = CIFAR10('./data/cifar10', train=True, download=True, transform=transform_train)
data_test = CIFAR10('./data/cifar10', train=False, download=True, transform=transform_test)


model_dims = {
    1: {"name": "input", "dim": {"channel": 3, "out_size": 32}},
    2: {"name": "cnn", "dim": {"channel": 6, "kernel": 6, "stride": 2, "out_size": 14}},
    3: {"name": "cnn", "dim": {"channel": 16, "kernel": 6, "stride": 2, "out_size": 5}},
    4: {"name": "fc", "dim": {"out_size": 120}},
    5: {"name": "fc", "dim": {"out_size": 84}},
    6: {"name": "fc", "dim": {"out_size": 10}}
}


selected_classes = [0,1,2,3,4,5,6,7,8,9]


def standard_PGD(model, images, labels, device, eps=11/255, alpha=2/255, iters=40):
    images = images.to(device)
    labels = labels.to(device)
    loss = nn.CrossEntropyLoss()
        
    ori_images = images.data
        
    for i in range(iters) :    
        images = images.clone().detach().requires_grad_(True)
        outputs = model(images)

        model.zero_grad()
        cost = loss(outputs, labels).to(device)
        cost.backward()

        adv_images = images + alpha*images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
            
    return images


def test(n, loader, eps, alpha, iters, device):    
    n.eval()
    robust_pair = defaultdict(list)
    succ_pair = defaultdict(list)
    
    for l in selected_classes:
        for i, (images, labels) in enumerate(loader[l]):
            images = images.to(device)
            labels = labels.to(device)
            output = n(images)
            pred = output.detach().max(1)[1]
            
            adv_img = standard_PGD(n, images, labels, device, eps, alpha, iters)
            adv_out = n(adv_img)
            adv_pred = adv_out.detach().max(1)[1]
            
            robust_l = pred.eq(labels.view_as(pred)) & adv_pred.eq(pred)
            
            succ_l = pred.eq(labels.view_as(pred)) & ~adv_pred.eq(labels.view_as(pred))
            
            succ_pair[l].append((images[succ_l].cpu(), adv_img[succ_l].cpu()))
            robust_pair[l].append((images[robust_l].cpu(), adv_img[robust_l].cpu()))
            
        print(f'Finish label {l}....')
    return succ_pair, robust_pair


def get_fraction(curvature, b):
    c = []
    neg = []
    total_e = []
    for i in range(b):
        curr = np.array(curvature[i])
        neg.append(len(curr[curr<0]))
        total_e.append(len(curr))
    return np.array(neg), np.array(total_e), curr


def cal_dims(model_dims):
    dims = []
    layer_num = len(model_dims)
    
    for i in range(1, layer_num + 1):
        cur_name = model_dims[i]["name"]
        cur_dim = model_dims[i]["dim"]
        cur_size = cur_dim['out_size']

        cur_channel = 1 if (cur_name == "fc") else cur_dim['channel']

        if cur_name == "input":
            cur_nodes = cur_channel * cur_size**2
        else:
            cur_nodes = cur_size if (cur_name == "fc") else cur_dim['channel']*(cur_size**2)
            
        dims.append(cur_nodes)
    
    return dims



'''
parameters:
    @ q_NGR, q_INV, q_EXP     --G_Def
    @ model type/name: fc, cnn   --model_type
    @ model path   --model_path
    @ result path  --res_path
    @ example num    --sample_num

'''
def cifar_small_main(args):
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

    selected_classes = [0,1,2,3,4,5,6,7,8,9]
    train_loader, test_loader, valid_loader, valid_dataset, test_dataset = utils.get_new_data(selected_classes, data_train, data_test, valid_num = 2000, test_bs=1)

    # sep_dataloader = utils.sep_label(valid_dataset, selected_classes, bs=1)
    sep_dataloader = utils.sep_label(test_dataset, selected_classes, bs=2000)
    
    dims = cal_dims(model_dims)
    
    model_type = args.model_type
    model_pre_name = args.model_name
    res_path = args.mnist_res_path
    model_path = args.model_path
    metric = args.metric
    sample_size = args.sample_num
    
    model_full_n = model_type.lower() + model_pre_name.lower()
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    
    # build model
    model_name= "best_cifar.pth"
    if model_pre_name.lower() == 'ori':
        model_name= "best_cifar.pth"
    elif model_pre_name.lower() == 'adv':
        model_name= "best_cifar_adv.pth"
        
    print(model_name)
    
    net_H = LeNet_custom(model_dims, device, input_c=3)
    net_H.load_state_dict(torch.load(model_path + model_name))
    net_H = net_H.to(device)
    
    eps = [1,2,3,5]
    Q = [1]
    
    for q in Q:
        with open(res_path + "cifar10.txt", "w+") as f:
            f.write(f'dims = {dims}\n\n')
            for e in eps:
                f.write(f'eps = {e}....\n\n')
                
                succ_pair, robust_pair = test(net_H, sep_dataloader, eps=e/255, alpha=2/255, iters=40, device = device)
                    
                sample_size = 50
                
                robust_c = defaultdict(list)
                nonrobust_c = defaultdict(list)
                non_fraction = defaultdict(list)
                rob_fraction = defaultdict(list)
                    
            
                for l in selected_classes:
                    print(f'For label {l}....\n')
                    f.write(f'Label {l}....\n')

                    count = 0
                    for (ori_im, adv_im) in succ_pair[l]:
                        for im in ori_im:
                            img = im.to(device)
                            edge_array, nodes_ori, output = net_H.NN_info_batch(img.unsqueeze(0))
                            
                            if metric.lower() == "q_ngr" or metric.lower() == "q_inv":
                                weights = output.detach().clone().to(device)                   
                                weights[edge_array == 0] = 0.
                                
                            elif metric.lower() == "q_exp":
                                weights = edge_array.detach().clone().to(device)  
                            
                            if metric.lower() == "q_ngr":
                                _, weights_inv = net_H.normalization_weight_w1(nodes_ori, weights, dims, model_dims)
                                
                            elif metric.lower() == "q_inv":
                                _, weights_inv = net_H.normalization_weight_w2(nodes_ori, weights, dims, model_dims)
                
                            elif metric.lower() == "q_exp":
                                weights_inv = net_H.normalization_weight_w6(nodes_ori, weights, dims, model_dims, q)
                            else:
                                raise Exception("Invalid graph metric, metric should be {q_ngr, q_inv, q_exp}!")
                            
                            weights_inv = weights_inv.detach()

                            ricci_curvature = graph_curvature_main_torch(dims, weights_inv, model_dims=model_dims, device=device)
        
                            neg_num, total_edge, c = get_fraction(ricci_curvature, weights_inv.shape[0])
                        
                            nonrobust_c[l].append(c)
                            non_fraction[l].append(neg_num/total_edge)
                            
                            count += 1
                            if (count % 10 == 0):
                                print(f'Finish {count} graphs....')
                                
                            if (count >= sample_size):
                                break
                            
                    count = 0
                    for (ori_im, adv_im) in robust_pair[l]:
                        for im in ori_im:
                            img = im.to(device)
                            edge_array, nodes_ori, output = net_H.NN_info_batch(img.unsqueeze(0))
                            
                            if metric.lower() == "q_ngr" or metric.lower() == "q_inv":
                                weights = output.detach().clone().to(device)                   
                                weights[edge_array == 0] = 0.
                                
                            elif metric.lower() == "q_exp":
                                weights = edge_array.detach().clone().to(device)  
                            
                            if metric.lower() == "q_ngr":
                                _, weights_inv = net_H.normalization_weight_w1(nodes_ori, weights, dims, model_dims)
                                
                            elif metric.lower() == "q_inv":
                                _, weights_inv = net_H.normalization_weight_w2(nodes_ori, weights, dims, model_dims)
                
                            elif metric.lower() == "q_exp":
                                weights_inv = net_H.normalization_weight_w6(nodes_ori, weights, dims, model_dims, q)
                            else:
                                raise Exception("Invalid graph metric, metric should be {q_ngr, q_inv, q_exp}!")
                            
                            weights_inv = weights_inv.detach()
                            
                            ricci_curvature = graph_curvature_main_torch(dims, weights_inv, model_dims=model_dims, device=device)
        
                            neg_num, total_edge, c = get_fraction(ricci_curvature, weights_inv.shape[0])
           
                            robust_c[l].append(c)
                            rob_fraction[l].append(neg_num/total_edge)
                            
                            count += 1
                            if (count % 10 == 0):
                                print(f'Finish {count} graphs....')
                                
                            if (count >= sample_size):
                                break
                    
                with open(res_path + model_full_n + str(e) + metric + str(q) + "frac_robust_cifar.pkl", 'wb') as file:
                    pickle.dump(rob_fraction, file)
                with open(res_path + model_full_n + str(e) + metric + str(q) +  "frac_norobust_cifar.pkl", 'wb') as file:
                    pickle.dump(non_fraction, file)
                    
                with open(res_path + model_full_n + str(e) + metric + str(q) + "curv_robust_cifar.pkl", 'wb') as file:
                    pickle.dump(robust_c, file)
                with open(res_path + model_full_n + str(e) + metric + str(q) + "curv_norobust_cifar.pkl", 'wb') as file:
                    pickle.dump(nonrobust_c, file)
