import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
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
import networkx as nx

import sys
sys.path.append("..")

import tools.utils as utils

from tools.graph_curvature import graph_curvature_main_torch

np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=torch.inf)

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

data_train = MNIST('./data/mnist',
                  train=True,
                  download=True,
                  transform=transforms.Compose([
                      # transforms.Resize((32, 32)),
                      transforms.ToTensor()]))

data_test = MNIST('./data/mnist',
                  train=False,
                  download=True,
                  transform=transforms.Compose([
                      # transforms.Resize((32, 32)),
                      transforms.ToTensor()]))



selected_classes = [0,1,2,3,4,5,6,7,8,9]


nodes_num = 2118

model_dims = {
    1: {"name": "input", "dim": {"channel": 1, "out_size": 28}},
    2: {"name": "cnn", "dim": {"channel": 6, "kernel": 6, "stride": 2, "out_size": 12}},
    3: {"name": "cnn", "dim": {"channel": 16, "kernel": 6, "stride": 2, "out_size": 4}},
    4: {"name": "fc", "dim": {"out_size": 120}},
    5: {"name": "fc", "dim": {"out_size": 84}},
    6: {"name": "fc", "dim": {"out_size": 10}}
}



def load_dataset_from_disk(path, batch_size=128, shuffle=True):
    data_path = f"{path}/data.pt"
    images, labels = torch.load(data_path)
    dataset = TensorDataset(images, labels)
    return dataset



def standard_PGD(model, images, labels, device, eps=11/255, alpha=2/255, iters=40):
    images = images.to(device)
    labels = labels.to(device)
    loss = nn.CrossEntropyLoss()
        
    ori_images = images.data
        
    for i in range(iters) :    
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        cost = loss(outputs, labels).to(device)
        cost.backward()

        adv_images = images + alpha*images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
            
    return images




def test_clean(n, loader, device = 'cuda'):
    n.eval()
    total_correct = 0.
    
    for i, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        output = n(images)
        pred = output.detach().max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()

    # print(f'Test Accuracy for label {l}: {(float(total_correct) / len(loader.dataset)):.3f}')
    
    acc = float(total_correct) / len(loader.dataset)
    return acc



def test_adversarial(net, loader, eps=.1, alpha=.1, iters=100, device = 'cuda'):
    # prepare model for testing (only important for dropout, batch norm, etc.)
    net.eval()
    correct = 0.

    for data, target in loader:

        data = standard_PGD(net, data, target, device, eps=eps, alpha=alpha, iters=iters)
        data, target = data.to(device), target.to(device)

        output = net(data)
        pred = output.detach().max(1)[1]
        correct += pred.eq(target.view_as(pred)).sum()
    
    return float(correct) / len(loader.dataset)



def test(n, loader, device):    
    n.eval()
    robust_pair = defaultdict(list)
    succ_pair = defaultdict(list)
 
    for l in selected_classes:
        for i, (images, labels) in enumerate(loader[l]):
            images = images.to(device)
            labels = labels.to(device)
            output = n(images)
            pred = output.detach().max(1)[1]
            
            robust_l = pred.eq(labels.view_as(pred))
            succ_l = ~pred.eq(labels.view_as(pred))

            succ_pair[l].append((images[succ_l].cpu(), pred[succ_l].cpu()))
            robust_pair[l].append((images[robust_l].cpu(), pred[robust_l].cpu()))

    return succ_pair, robust_pair



def get_top_c(curvature, b, prefix_dims, threshold = -50):
    c = []
    neg_e = set()  # Negative curvature edges
    pos_e = set()  # Positive curvature edges
    
    for batch in range(b):
        ricci_curv = np.array(curvature[batch])
        for (i, j, curr) in ricci_curv:
            if curr > 1:
                continue

            i_layer = np.searchsorted(prefix_dims, i, side='right') - 1
            if i_layer >= 3:
                c.append((i,j,curr))

    c.sort(key=lambda x: x[2])
    
    for (i,j,curr) in c:
        i1 = (int)(i)
        j1 = (int)(j)
        if curr < 0:
            neg_e.add((i1,j1))
        else:
            pos_e.add((i1,j1))
        
    return c, neg_e, pos_e


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




def community_check_cnn(args):
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

    
    # train_loader, test_loader, valid_loader, valid_dataset, test_dataset = utils.get_new_data(selected_classes, data_train, data_test, test_bs=2000, valid_num=5000)

    val_set = load_dataset_from_disk("./data/MNIST_val", batch_size=64, shuffle=False)
    sep_dataloader = utils.sep_label(val_set, selected_classes, bs=1000)
    
    eps = [0.03, 0.07, 0.1, 0.2]
    dims = cal_dims(model_dims)
    
    model_type = args.model_type
    model_pre_name = args.model_name
    res_path = args.mnist_res_path
    model_path = args.model_path
    metric = args.metric
    dataset = args.dataset
    alpha = args.alpha
    hops = args.hops
    sample_size = args.sample_num
    activation = args.activation
    
    if activation.lower() == "relu":
        from tools.LeNet5_custom_small import LeNet_custom_v2
    elif activation.lower() == "tanh":
        from tools.LeNet5_custom_small_tanh import LeNet_custom_v2
    
    model_full_n = model_type.lower() + model_pre_name.lower()

    dims = cal_dims(model_dims)
    prefix_dims = np.cumsum([0] + dims).tolist()
    
    if not os.path.exists(res_path):
        os.makedirs(res_path)
        
    # build model
    if model_pre_name == 'ori':
        model_name = "cnn_ori_"
    elif model_pre_name == 'adv':
        model_name = "cnn_adv_"
    elif model_pre_name == 'wd':
        model_name = "cnn_wd_"
        
    model_name = model_name + activation + ".pth"

    net_H = LeNet_custom_v2(model_dims, None, device)
    net_H.load_state_dict(torch.load(model_path + model_name))
    net_H = net_H.to(device)

    net_full = copy.deepcopy(net_H)

    print(model_name)    
            
    succ_pair, robust_pair = test(net_H, sep_dataloader, device=device)
    
    res_l = defaultdict(list)
    # res_l_non = defaultdict(list)

    for l in selected_classes:
        # for (images, labels) in succ_pair[l]:
        #     for idx in range(images.shape[0]):
        #         if (idx >= sample_size):
        #             print(f'Finish {idx} examples....')
        #             break
                    
        #         img = images[idx].to(device)
        #         edge_array, nodes_ori, output = net_full.NN_info_batch(img.unsqueeze(0))

        #         if metric.lower() == "w1":
        #             weights = output.detach().clone().to(device)                   
        #             # weights[edge_array == 0] = 0.
                
        #         elif metric.lower() == "w3":
        #             weights = output.detach().clone().to(device)                   
        #             # weights[edge_array == 0] = 0.
                    
        #         if metric.lower() == "w1":
        #             weights_inv1, weights_inv2 = net_full.normalization_weight_w1(nodes_ori, weights, dims, model_dims)
        #             weights_inv = weights_inv1.detach()
        #             weights_inv2 = weights_inv2.detach()
        #             ricci_curvature, sp_dict = graph_curvature_main_torch(dims, weights_inv, device=device, model_dims=model_dims, probability_w=weights_inv2, alpha=alpha)
                        
        #         elif metric.lower() == "w3":
        #             weights_inv1, weights_inv2 = net_full.normalization_weight_w3(nodes_ori, weights, dims, model_dims)
        #             weights_inv = weights_inv1.detach()
        #             weights_inv2 = weights_inv2.detach()
        #             ricci_curvature, sp_dict = graph_curvature_main_torch(dims, weights_inv, device=device, model_dims=model_dims, probability_w=weights_inv2, alpha=alpha)
        #         else:
        #             raise Exception("Invalid graph metric, metric should be {q_ngr, q_inv, q_exp}!")
                
        #         res_l_non[l].append((ricci_curvature, weights_inv.shape[0], dims, nodes_ori.cpu()))
                
    
        for (images, labels) in robust_pair[l]:
            for idx in range(images.shape[0]):
                if (idx >= sample_size):
                    print(f'Finish {idx} examples....')
                    break
                    
                img = images[idx].to(device)
                edge_array, nodes_ori, output = net_full.NN_info_batch(img.unsqueeze(0))

                if metric.lower() == "w1":
                    weights = output.detach().clone().to(device)                   
                    # weights[edge_array == 0] = 0.
                
                elif metric.lower() == "w3":
                    weights = output.detach().clone().to(device)                   
                    # weights[edge_array == 0] = 0.
                    
                if metric.lower() == "w1":
                    weights_inv1, weights_inv2 = net_full.normalization_weight_w1(nodes_ori, weights, dims, model_dims)
                    weights_inv = weights_inv1.detach()
                    weights_inv2 = weights_inv2.detach()
                    ricci_curvature = graph_curvature_main_torch(dims, weights_inv, device=device, model_dims=model_dims, probability_w=weights_inv2, alpha=alpha)
                        
                elif metric.lower() == "w3":
                    weights_inv1, weights_inv2 = net_full.normalization_weight_w3(nodes_ori, weights, dims, model_dims)
                    weights_inv = weights_inv1.detach()
                    weights_inv2 = weights_inv2.detach()
                    ricci_curvature = graph_curvature_main_torch(dims, weights_inv, device=device, model_dims=model_dims, probability_w=weights_inv2, alpha=alpha, layers_to_process=[0,1,2,3,4])
                else:
                    raise Exception("Invalid graph metric, metric should be {q_ngr, q_inv, q_exp}!")
                
                res_l[l].append((ricci_curvature, weights_inv.shape[0], dims, nodes_ori.cpu()))
                
        print(f'Finished label {l}.')
                    
    with open(res_path + model_full_n + metric + '_' + dataset + "_res_correct.pkl", 'wb') as file:
        pickle.dump(res_l, file)
    # with open(res_path + model_full_n + metric + '_' + dataset + "_res_misclassified.pkl", 'wb') as file:
    #     pickle.dump(res_l_non, file)

