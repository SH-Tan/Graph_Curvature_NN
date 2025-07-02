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
from tools.graph_curvature_dynamic import graph_curvature_main_torch

np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=torch.inf)

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")


layers = [2, 4, 5, 6, 7]

model_zoo = {
    2: [784, 20, 15, 10],
    21: [784, 200, 150, 10],
    4: [784, 15, 25, 20, 15, 10],
    5: [784, 20, 30, 30, 20, 15, 10],
    6: [784, 20, 30, 30, 35, 20, 15, 10],
    7: [784, 30, 30, 40, 50, 30, 25, 20, 10]
}

selected_classes = [0,1,2,3,4,5,6,7,8,9]



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




def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def community_check_fc(args):
    seed = 59
    set_seed(seed)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # train_loader, test_loader, valid_loader, valid_dataset, test_dataset = utils.get_new_data(selected_classes, data_train, data_test, test_bs=2000, valid_num=5000)
    val_set = load_dataset_from_disk("./data/MNIST_val", batch_size=64, shuffle=False)
    sep_dataloader = utils.sep_label(val_set, selected_classes, bs=1000)
    
    eps = [0.03, 0.07, 0.1, 0.2]
    
    model_type = args.model_type
    model_pre_name = args.model_name
    res_path = args.mnist_res_path
    model_path = args.model_path
    metric = args.metric
    dataset = args.dataset
    alpha = args.alpha
    sample_size = args.sample_num
    activation = args.activation
    
    if activation.lower() == "relu":
        from tools.small_model_relu import FC_MD
    elif activation.lower() == "tanh":
        from tools.small_model_tanh import FC_MD
    
    model_full_n = model_type.lower() + model_pre_name.lower()
    
    if not os.path.exists(res_path):
        os.makedirs(res_path)
        
    layers = [2,4]
    if 'big' in model_pre_name.lower():
        layers = [2]

    # remove_frac = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1]
    # remove_num = [5,20,50,100,150,300,500,700,1000,1500,2000,2500,3000,3500,5000,10000]
    # remove_num = [5,20,50,100,150,300,500,700,1000,1500]
    
    # build model
    for layer_num in layers:
        dims = model_zoo[layer_num]
        
        if model_pre_name.lower() == "ori" or model_pre_name.lower() == "decay":
            model_name = "best_ori_10l_" + str(layer_num) + ".pth"
        elif model_pre_name.lower() == "adv":
            model_name = "pgdtrain_" + str(layer_num) + ".pth"
        elif model_pre_name.lower() == 'big_adv':
            model_name = "big_adv_"
            dims = model_zoo[21]
        elif model_pre_name.lower() == 'big_ori':
            model_name = "big_ori_"
            dims = model_zoo[21]
        elif model_pre_name.lower() == 'big_wd':
            model_name = "big_wd_"
            dims = model_zoo[21]
        else:
            raise Exception("Invalid model name, model name should be {ori, decay, adv}!")
        
        model_name = model_name + activation + ".pth"
        
        prefix_dims = np.cumsum([0] + dims).tolist()
        
        print(f'Now for model {model_path + model_name}....\n')

        net_H = FC_MD(dims, layer_num)

        net_H.load_state_dict(torch.load(model_path + model_name))
        net_H = net_H.to(device)

        net_full = copy.deepcopy(net_H)
        
        neural_list = []
        nodes_num = 0
        edges_num = 0
        i = 0
        for p in net_H.parameters():
            if i == 0:
                nodes_num += p.shape[1]
            if i%2 == 0:
                nodes_num += p.shape[0]
                edges_num += (p.shape[0] * p.shape[1])
                neural_list.append(p.shape[0])
            i += 1
   
        # test_cleanacc = test_clean(net_full, test_loader)
            
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
            #         edge_array, nodes_ori, output, all_node = net_full.NN_info_batch(img.unsqueeze(0))
                    
            #         if metric.lower() == "w1":
            #             weights = output.detach().clone().to(device)                   
            #             # weights[edge_array == 0] = 0.
                    
            #         elif metric.lower() == "w3":
            #             weights = output.detach().clone().to(device)                   
            #             # weights[edge_array == 0] = 0.
                        
            #         if metric.lower() == "w1":
            #             weights_inv1, weights_inv2 = net_full.normalization_weight_w1(nodes_ori, weights, dims)
            #             weights_inv = weights_inv1.detach()
            #             weights_inv2 = weights_inv2.detach()
            #             ricci_curvature, sp_dict = graph_curvature_main_torch(dims, weights_inv, device=device, probability_w=weights_inv2, alpha=alpha)
                            
            #         elif metric.lower() == "w3":
            #             weights_inv1, weights_inv2 = net_full.normalization_weight_w3(nodes_ori, weights, dims)
            #             weights_inv = weights_inv1.detach()
            #             weights_inv2 = weights_inv2.detach()
            #             ricci_curvature, sp_dict = graph_curvature_main_torch(dims, weights_inv, device=device, probability_w=weights_inv2, alpha=alpha)
            #         else:
            #             raise Exception("Invalid graph metric, metric should be {q_ngr, q_inv, q_exp}!")
                    
            #         res_l_non[l].append((ricci_curvature, weights_inv.shape[0], dims, nodes_ori.cpu()))

                               
            for (images, labels) in robust_pair[l]:
                for idx in range(images.shape[0]):
                    if (idx >= sample_size):
                        print(f'Finish {idx} examples....')
                        break

                    img = images[idx].to(device)
                    edge_array, nodes_ori, output, all_node = net_full.NN_info_batch(img.unsqueeze(0))

                    if metric.lower() == "w1":
                        weights = output.detach().clone().to(device)                   
                        # weights[edge_array == 0] = 0.
                    
                    elif metric.lower() == "w3":
                        weights = output.detach().clone().to(device)                   
                        weights[edge_array == 0] = 0.
                        
                    if metric.lower() == "w1":
                        weights_inv1, weights_inv2 = net_full.normalization_weight_w1(nodes_ori, weights, dims)
                        weights_inv = weights_inv1.detach()
                        weights_inv2 = weights_inv2.detach()
                        ricci_curvature= graph_curvature_main_torch(dims, weights_inv, device=device, probability_w=weights_inv2, alpha=alpha)
                            
                    elif metric.lower() == "w3":
                        weights_inv1, weights_inv2 = net_full.normalization_weight_w3(nodes_ori, weights, dims)
                        weights_inv = weights_inv1.detach()
                        weights_inv2 = weights_inv2.detach()
                        ricci_curvature = graph_curvature_main_torch(dims, weights_inv, device=device, probability_w=weights_inv2, alpha=alpha)
                    else:
                        raise Exception("Invalid graph metric, metric should be {q_ngr, q_inv, q_exp}!")
                    
                    res_l[l].append((ricci_curvature, weights_inv.shape[0], dims, nodes_ori.cpu()))
                    
            print(f'Finished label {l}.')
                    
        with open(res_path + model_full_n + metric + '_' + str(layer_num) + dataset + "_res_correct.pkl", 'wb') as file:
            pickle.dump(res_l, file)
        # with open(res_path + model_full_n + metric + '_' + str(layer_num) + dataset + "_res_misclassified.pkl", 'wb') as file:
        #     pickle.dump(res_l_non, file)


           