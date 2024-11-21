import torch
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
import numpy as np
import random
import os
import pandas as pd

from GraphRicciCurvature.OllivierRicci import OllivierRicci
import networkx as nx

import standard_pgd_test
import copy
import pickle
import community as community_louvain

import sys
sys.path.append("..")

from tools.edge_remove import Edge_Remove
import tools.utils as utils
from tools.small_model import FC_MD


os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")



data_train = MNIST('../data/mnist',
                  train=True,
                  download=True,
                  transform=transforms.Compose([
                      # transforms.Resize((32, 32)),
                      transforms.ToTensor()]))

data_test = MNIST('../data/mnist',
                  train=False,
                  download=True,
                  transform=transforms.Compose([
                      # transforms.Resize((32, 32)),
                      transforms.ToTensor()]))


file_path = "edge_v/"
model_path = "models/"
res_path = "res/"
graph_file = "sep_avg_graph/"

layers = [2, 4, 5, 6, 7]

model_zoo = {
    2: [784, 20, 15, 10],
    4: [784, 15, 25, 20, 15, 10],
    5: [784, 20, 30, 30, 20, 15, 10],
    6: [784, 20, 30, 30, 35, 20, 15, 10],
    7: [784, 30, 30, 40, 50, 30, 25, 20, 10]
}




# for directed graph
def cal_partition(partition, f, layer_num):
    model_dims = model_zoo[layer_num]
    communities = set(partition)
    n = len(communities)
    f.write(f'Total {n} partitions...\n')

    for com in communities:
        if (len(com) > 1):
            f.write(f'This partition has {len(com)} nodes...\n')
            max_partition = np.array(list(com))
            cur_l = 0
            pre_n = 0
            
            while cur_l < len(model_dims):
                cur_nodes = model_dims[cur_l]
                
                cur_l_n = max_partition[max_partition >= pre_n]
                cur_l_n = cur_l_n[cur_l_n < pre_n+cur_nodes]
                
                f.write(f'The current layer is {cur_l}, it has {len(cur_l_n)} nodes...\n')
                cur_l += 1
                pre_n += cur_nodes
            
            f.write('\n')
          


def test(n, loader):
    n.eval()
    total_correct = 0
    
    for i, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        output = n(images)
        pred = output.detach().max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()

    # print(f'Test Accuracy for label {l}: {(float(total_correct) / len(loader.dataset)):.3f}')
    
    acc = float(total_correct) / len(loader.dataset)
    return acc
  
            
if __name__ == '__main__':
    seed = 59
    
    # set random seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    selected_classes = [0,1,2,3,4,5,6,7,8,9]
    train_loader, test_loader, valid_loader, valid_dataset, test_dataset = utils.get_new_data(selected_classes, data_train, data_test)

    valid_dataloader = utils.sep_label(valid_dataset, selected_classes)
    sep_dataloader = utils.sep_label(test_dataset, selected_classes)
    
    
    # build model
    for layer_num in layers:
        model_name = "best_ori_10l_" + str(layer_num) + ".pth"
        
        # model_name = "pgdtrain_" + str(layer_num) + ".pth"

        dims = model_zoo[layer_num]
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
            
        print(f'Total nodes are {nodes_num}, total edges are {edges_num}.')

        with open(res_path + "sepnet_dir_partition.txt", "a+") as f:    
            f.write(f'For model {dims}.... \n')   

            for l in selected_classes:
                acc = test(net_H, valid_dataloader[l])
                f.write(f'The acc for label {l} is {acc:.3f}...\n')
                
                # net_H.load_state_dict(torch.load(res_path + model_name))
                adjacent_m = utils.build_adjm(valid_dataloader[l], net_H, nodes_num, dims, device)   

                # Create network object
                G = nx.from_numpy_array(adjacent_m, create_using=nx.DiGraph)

                orf = OllivierRicci(G, alpha=0., verbose="ERROR")
                # orf.compute_ricci_flow(iterations=100)
                orf.compute_ricci_curvature()

                G1 = orf.G.copy()
                f.write(f'For label {l} : {G1}.... \n')
                print(G1)
                
                partition = nx.community.greedy_modularity_communities(G1)
                cal_partition(partition, f, layer_num)
                
                with open(graph_file + 'graph_' + str(layer_num) + '_' + str(l) + '.pkl', 'wb') as file:
                    pickle.dump(G1, file)
                
                f.write('\n')
            f.write('\n')