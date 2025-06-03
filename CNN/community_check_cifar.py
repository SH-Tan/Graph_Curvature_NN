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
import copy

import pickle
import time
import pandas as pd

import sys
sys.path.append("..")

import tools.utils as utils
from tools.small_model_relu import FC_MD
from tools.FC_linear import FC_Linear
from tools.LeNet5_custom import LeNet_custom
from tools.graph_curvature import graph_curvature_main_torch
from tools.get_c import get_c
from tools.get_community import multi_community_from_output, negative_edge_communities, community_split_by_community_louvain, find_all_backward_communities, write_graph_info_to_excel


np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=torch.inf)

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

# 4852 + 10

selected_classes = [0,1,2,3,4,5,6,7,8,9]



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
    neg_e_second = set()  # Negative edges in second layer
    neg_e_other = set()   # Negative edges in other layers 
    pos_e = set()
    
    for batch in range(b):
        ricci_curv = np.array(curvature[batch])
        for (i, j, curr) in ricci_curv:
            if curr > 1:
                continue
            c.append((i,j,curr))

    c.sort(key=lambda x: x[2])
    
    for (i,j,curr) in c:
        i_layer = np.searchsorted(prefix_dims, i, side='right') - 1
        i1 = (int)(i)
        j1 = (int)(j)
        if curr < 0:
            if i_layer == 1:  # Second layer (index 1)
                neg_e_second.add((i1,j1))
            else:
                neg_e_other.add((i1,j1))
        elif curr >= 0:
            pos_e.add((i1,j1))
        
    return c, neg_e_second, neg_e_other, pos_e


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




def community_check_cifar(args):
    seed = 29
    
    # set random seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    
    train_loader, test_loader, valid_loader, valid_dataset, test_dataset = utils.get_new_data(selected_classes, data_train, data_test, test_bs=2000, valid_num=5000)

    sep_dataloader = utils.sep_label(test_dataset, selected_classes, bs=5000)
    
    eps = [1,2,3,5]
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
    
    model_full_n = model_type.lower() + model_pre_name.lower()

    dims = cal_dims(model_dims)
    prefix_dims = np.cumsum([0] + dims).tolist()
    
    if not os.path.exists(res_path):
        os.makedirs(res_path)
        
    # build model
    model_name= "cnn_cifar_ori.pth"
    if model_pre_name.lower() == 'ori':
        model_name= "cnn_cifar_ori.pth"
    elif model_pre_name.lower() == 'adv':
        model_name= "cnn_cifar_adv.pth"
    
    net_H = LeNet_custom(model_dims, device, input_c=3)
    net_H.load_state_dict(torch.load(model_path + model_name))
    net_H = net_H.to(device)

    net_full = copy.deepcopy(net_H)

    print(model_name)
    # remove_frac = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1]
    remove_num = [5,20,50,100,150,300,500,700,1000,1500,2000,2500,3000,3500,5000]
      
    test_cleanacc = test_clean(net_full, test_loader)
    succ_pair, robust_pair = test(net_H, sep_dataloader, device=device)
         
    # succ_pair, robust_pair = test(net_H, sep_dataloader, eps=e, alpha=2/255, iters=40, device=device)

    for l in selected_classes:
        with open(res_path + "community_cifar_" + str(l) + ".txt", "w+") as ff:
            ff.write(f'For model {model_name}: \n')
            ff.write(f'The clean accuracy for original model is {test_cleanacc}\n\n')
            print(f'Current label {l}: \n')
            ff.write(f'Current label {l}: \n')

            edge_list_mis = []
            node_list_mis= [] 
            edge_list_ground = []
            node_list_ground = [] 
            for (images, labels) in succ_pair[l]:
                for idx in range(images.shape[0]):
                    if (idx >= sample_size):
                        print(f'Finish {idx} examples....')
                        break
                    
                    ff.write(f'\nFor misclassified example {idx}: True label {l}, Acctual output {labels[idx]} \n')
                    img = images[idx].to(device)
                    edge_array, nodes_ori, output = net_full.NN_info_batch(img.unsqueeze(0))

                    weights = output.detach().clone().to(device)                   
                    weights[edge_array == 0] = 0.
                    weights_inv = net_full.normalization_weight_w2(nodes_ori, weights, dims, model_dims)
                    weights_inv = weights_inv.detach()
                    
                    ricci_curvature, sp_dict = graph_curvature_main_torch(dims, weights_inv, device=device, model_dims=model_dims, alpha=alpha)

                    summary, node_communities, graph_info = find_all_backward_communities(
                        ricci_curvature, 1, prefix_dims, threshold=-2
                    )

                    # Get node indices for true and predicted labels
                    true_output_node = prefix_dims[-2] + l
                    pred_output_node = prefix_dims[-2] + labels[idx].item()
                    
                    # Get community IDs that contain each output node
                    true_comms = node_communities.get(true_output_node, set())
                    pred_comms = node_communities.get(pred_output_node, set())

                    # Extract stats for the true label community (only one community expected)
                    if true_comms:
                        true_cid = next(iter(true_comms))  # Take the first (should usually be only one)
                        true_nodes = set(summary[true_cid]["nodes"])
                        true_edges = set(map(tuple, summary[true_cid]["edges"]))

                        ff.write(f"\n--- Ground Truth Community ---\n")
                        ff.write(f"  Community ID: {true_cid}\n")
                        ff.write(f"  Node count: {len(true_nodes)}\n")
                        ff.write(f"  Edge count: {len(true_edges)}\n")

                        node_list_ground.append(len(true_nodes))
                        edge_list_ground.append(len(true_edges))
                    else:
                        ff.write(f"\n--- Ground Truth Community ---\n")
                        ff.write(f"  Not found for node {true_output_node}\n")
                        node_list_ground.append(0)
                        edge_list_ground.append(0)

                    # Extract stats for the predicted label community (only if different)
                    if pred_output_node != true_output_node and pred_comms:
                        pred_cid = next(iter(pred_comms))
                        pred_nodes = set(summary[pred_cid]["nodes"])
                        pred_edges = set(map(tuple, summary[pred_cid]["edges"]))

                        ff.write(f"\n--- Predicted Output Community ---\n")
                        ff.write(f"  Community ID: {pred_cid}\n")
                        ff.write(f"  Node count: {len(pred_nodes)}\n")
                        ff.write(f"  Edge count: {len(pred_edges)}\n")

                        node_list_mis.append(len(pred_nodes))
                        edge_list_mis.append(len(pred_edges))
                    else:
                        ff.write(f"\n--- Predicted Output Community ---\n")
                        ff.write(f"  Not found for node {pred_output_node} (or same as ground truth)\n")
                        node_list_mis.append(0)
                        edge_list_mis.append(0)
            
            edge_list_correct = []
            node_list_correct = []        
            for (images, labels) in robust_pair[l]:
                for idx in range(images.shape[0]):
                    if (idx >= sample_size):
                        print(f'Finish {idx} examples....')
                        break
                    
                    ff.write(f'\nFor correct classified example {idx}: True label {l}, Acctual output {labels[idx]} \n')
                    img = images[idx].to(device)
                    edge_array, nodes_ori, output = net_full.NN_info_batch(img.unsqueeze(0))

                    weights = output.detach().clone().to(device)                   
                    weights[edge_array == 0] = 0.
                    weights_inv = net_full.normalization_weight_w2(nodes_ori, weights, dims, model_dims)
                    weights_inv = weights_inv.detach()
                    
                    ricci_curvature, sp_dict = graph_curvature_main_torch(dims, weights_inv, device=device, model_dims=model_dims, alpha=alpha)

                    summary, node_communities, graph_info = find_all_backward_communities(
                        ricci_curvature, 1, prefix_dims, threshold=-2
                    )

                    # Get node indices for true and predicted labels
                    true_output_node = prefix_dims[-2] + l

                    # Get community IDs that contain each output node
                    true_comms = node_communities.get(true_output_node, set())

                    # Extract stats for the true label community (only one community expected)
                    if true_comms:
                        true_cid = next(iter(true_comms))  # Take the first (should usually be only one)
                        true_nodes = set(summary[true_cid]["nodes"])
                        true_edges = set(map(tuple, summary[true_cid]["edges"]))

                        ff.write(f"\n--- Ground Truth Community ---\n")
                        ff.write(f"  Community ID: {true_cid}\n")
                        ff.write(f"  Node count: {len(true_nodes)}\n")
                        ff.write(f"  Edge count: {len(true_edges)}\n")

                        node_list_correct.append(len(true_nodes))
                        edge_list_correct.append(len(true_edges))
                    else:
                        ff.write(f"\n--- Ground Truth Community ---\n")
                        ff.write(f"  Not found for node {true_output_node}\n")
                        node_list_correct.append(0)
                        edge_list_correct.append(0)

            node_list_mis = np.array(node_list_mis)
            node_list_ground = np.array(node_list_ground)
            node_list_correct = np.array(node_list_correct)
            edge_list_mis = np.array(edge_list_mis)
            edge_list_ground = np.array(edge_list_ground)
            edge_list_correct = np.array(edge_list_correct)
                        
            ff.write(f'\n For the misclassified examples: \n') 
            ff.write(f'There are total {len(node_list_mis)} comminties, {len(node_list_mis[node_list_mis == 0])} are zero (not exist); {len(node_list_ground)} ground truth communities, {len(node_list_ground[node_list_ground == 0])} are zero (not exist). \n')
            ff.write(f'The average node number (non-zero) for misclassified community is {np.mean(node_list_mis[node_list_mis != 0])}, edge number is {np.mean(edge_list_mis[edge_list_mis != 0])}\n')     
            ff.write(f'The average node number (non-zero) for ground truth community is {np.mean(node_list_ground[node_list_ground!=0])}, edge number is {np.mean(edge_list_ground[edge_list_ground!=0])}\n')
            ff.write(f'\n For the correct classified examples: \n')     
            ff.write(f'There are total {len(node_list_correct)} comminties, {len(node_list_correct[node_list_correct == 0])} are zero (not exist). \n') 
            ff.write(f'The average node number (non-zero) for correct classified examples is {np.mean(node_list_correct[node_list_correct != 0])}, edge number is {np.mean(edge_list_correct[edge_list_correct!=0])}\n')        