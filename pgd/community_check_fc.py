import torch
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
import numpy as np
import random
import os
import pandas as pd
import torch.nn as nn
from collections import defaultdict
import seaborn as sns
import copy

# from GraphRicciCurvature.OllivierRicci import OllivierRicci
import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.integrate import simps
import pickle
import time
import pandas as pd
import networkx as nx

import sys
sys.path.append("..")

import tools.utils as utils
from tools.small_model import FC_MD
from tools.graph_curvature import graph_curvature_main_torch
from tools.get_community import multi_community_from_output, negative_edge_communities, community_split_by_community_louvain, find_all_backward_communities, write_graph_info_to_excel
from tools.get_node import get_key_nodes

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
            if i_layer == 3:  # Second layer (index 1)
                neg_e_second.add((i1,j1))
            else:
                neg_e_other.add((i1,j1))
        elif curr >= 0:
            pos_e.add((i1,j1))
        
    return c, neg_e_second, neg_e_other, pos_e



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
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    train_loader, test_loader, valid_loader, valid_dataset, test_dataset = utils.get_new_data(selected_classes, data_train, data_test, test_bs=2000, valid_num=5000)

    sep_dataloader = utils.sep_label(test_dataset, selected_classes, bs=5000)
    
    eps = [0.03, 0.07, 0.1, 0.2]
    
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
        elif model_pre_name.lower() == 'big_adv_01':
            model_name = "fc_big_adv.pth"
            dims = model_zoo[21]
        elif model_pre_name.lower() == 'big_adv':
            model_name = "best_21_adv.pth"
            dims = model_zoo[21]
        elif model_pre_name.lower() == 'big_ori':
            model_name = "fc_big_ori_new.pth"
            dims = model_zoo[21]
        elif model_pre_name.lower() == 'big_wd':
            model_name = "fc_big_wd.pth"
            dims = model_zoo[21]
        elif model_pre_name.lower() == 'big_wd_05':
            model_name = "fc_big_wd_05.pth"
            dims = model_zoo[21]
        elif model_pre_name.lower() == 'big_wd_0001':
            model_name = "fc_big_wd_0001.pth"
            dims = model_zoo[21]
        else:
            raise Exception("Invalid model name, model name should be {ori, decay, adv}!")
        
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
   
        test_cleanacc = test_clean(net_full, test_loader)
            
        succ_pair, robust_pair = test(net_H, sep_dataloader, device=device)

        for l in selected_classes:      
            with open(res_path + "community_fc_" + str(l) + str(layer_num) + ".txt", "w+") as ff:
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
                        edge_array, nodes_ori, output, all_node = net_full.NN_info_batch(img.unsqueeze(0))
    
                        weights = output.detach().clone().to(device)                   
                        weights[edge_array == 0] = 0.
                        weights_inv = net_full.normalization_weight_w2(nodes_ori, weights, dims)
                        weights_inv = weights_inv.detach()

                        ricci_curvature, sp_dict = graph_curvature_main_torch(dims, weights_inv, device=device, alpha=alpha)

                        # community_sizes, node_communities, graph_info = multi_community_from_output(ricci_curvature, 1, prefix_dims)
                        summary, node_communities, graph_info = find_all_backward_communities(
                            ricci_curvature, 1, prefix_dims, threshold=0.0
                        )
                        
                        ff.write("=== Graph Info ===\n")
                        for key, info in graph_info.items():
                            ff.write(f"{key}:\n")
                            if isinstance(info, dict):
                                for k, v in info.items():
                                    ff.write(f"  {k}: {v}\n")
                            else:
                                ff.write(f"  {info}\n")
                        
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
                        if pred_comms:
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
                edge_list_other = []
                node_list_other= []                  
                for (images, labels) in robust_pair[l]:
                    for idx in range(images.shape[0]):
                        if (idx >= sample_size):
                            print(f'Finish {idx} examples....')
                            break
                    
                        ff.write(f'\nFor correct classified example {idx}: True label {l}, Acctual output {labels[idx]} \n')
                        img = images[idx].to(device)
                        edge_array, nodes_ori, output, all_node = net_full.NN_info_batch(img.unsqueeze(0))
    
                        weights = output.detach().clone().to(device)                   
                        weights[edge_array == 0] = 0.
                        weights_inv = net_full.normalization_weight_w2(nodes_ori, weights, dims)
                        weights_inv = weights_inv.detach()

                        ricci_curvature, sp_dict = graph_curvature_main_torch(dims, weights_inv, device=device, alpha=alpha)

                        # community_sizes, node_communities, graph_info = multi_community_from_output(ricci_curvature, 1, prefix_dims)
                        summary, node_communities, graph_info = find_all_backward_communities(
                            ricci_curvature, 1, prefix_dims, threshold=0.0
                        )
                        
                        ff.write("=== Graph Info ===\n")
                        for key, info in graph_info.items():
                            ff.write(f"{key}:\n")
                            if isinstance(info, dict):
                                for k, v in info.items():
                                    ff.write(f"  {k}: {v}\n")
                            else:
                                ff.write(f"  {info}\n")

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
                            
                            # Find the largest community not equal to the ground-truth one
                            largest_cid = None
                            largest_size = -1

                            for cid, comm in summary.items():
                                if cid == true_cid:
                                    continue
                                size = len(comm["nodes"])  # Or use len(comm["edges"]) to switch to edge-based size
                                if size > largest_size:
                                    largest_cid = cid
                                    largest_size = size

                            if largest_cid is not None:
                                largest_nodes = set(summary[largest_cid]["nodes"])
                                largest_edges = set(map(tuple, summary[largest_cid]["edges"]))

                                node_list_other.append(len(largest_nodes))
                                edge_list_other.append(len(largest_edges))

                                ff.write(f"\n--- Largest Non-GT Community ---\n")
                                ff.write(f"  Community ID: {largest_cid}\n")
                                ff.write(f"  Node count: {len(largest_nodes)}\n")
                                ff.write(f"  Edge count: {len(largest_edges)}\n")
                            else:
                                node_list_other.append(0)
                                edge_list_other.append(0)
                                ff.write(f"\n--- Largest Non-GT Community ---\n")
                                ff.write(f"  Not found (only GT community exists)\n")
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
                node_list_other = np.array(node_list_other)
                edge_list_other = np.array(edge_list_other)
                            
                ff.write(f'\n For the misclassified examples: \n') 
                ff.write(f'There are total {len(node_list_mis)} comminties, {len(node_list_mis[node_list_mis == 0])} are zero (not exist); {len(node_list_ground)} ground truth communities, {len(node_list_ground[node_list_ground == 0])} are zero (not exist). \n')
                ff.write(f'The average node number (non-zero) for misclassified community is {np.mean(node_list_mis[node_list_mis != 0])}, edge number is {np.mean(edge_list_mis[edge_list_mis != 0])}\n')     
                ff.write(f'The average node number (non-zero) for ground truth community is {np.mean(node_list_ground[node_list_ground!=0])}, edge number is {np.mean(edge_list_ground[edge_list_ground!=0])}\n')
                ff.write(f'\n For the correct classified examples: \n')     
                ff.write(f'There are total {len(node_list_correct)} comminties, {len(node_list_correct[node_list_correct == 0])} are zero (not exist). \n') 
                ff.write(f'The average node number (non-zero) for correct classified examples is {np.mean(node_list_correct[node_list_correct != 0])}, edge number is {np.mean(edge_list_correct[edge_list_correct!=0])}\n')      
                ff.write(f'The average node number (non-zero) for largest non-GT communities is {np.mean(node_list_other[node_list_other != 0])}, edge number is {np.mean(edge_list_other[edge_list_other != 0])}\n')  