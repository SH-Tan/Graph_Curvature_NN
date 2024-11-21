import torch
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
import numpy as np
import random
import os
import pandas as pd
import matplotlib.pyplot as plt

# from GraphRicciCurvature.OllivierRicci import OllivierRicci

import networkx as nx

import standard_pgd_test
import copy
import pickle
import community as community_louvain

import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append("..")

from tools.edge_remove import Edge_Remove
import tools.utils as utils
from tools.small_model import FC_MD
from RicciCurvature.OllivierRicci_newW import OllivierRicci

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
graph_file = "graph/"

layers = [2, 4, 5, 6, 7]

model_zoo = {
    2: [784, 20, 15, 10],
    4: [784, 15, 25, 20, 15, 10],
    5: [784, 20, 30, 30, 20, 15, 10],
    6: [784, 20, 30, 30, 35, 20, 15, 10],
    7: [784, 30, 30, 40, 50, 30, 25, 20, 10]
}

# dims = [784, 20, 15, 10]
# layer_num = 2

# model_name = "best_ori_10l_2.pth"

# net_H = FC_MD(dims, layer_num)

# net_H.load_state_dict(torch.load(model_path + model_name))
# net_H = net_H.to(device)


def show_results(G, curvature="ricciCurvature"):
    weights = []
    sorted_edge = sorted(G.edges(data=True), key=lambda edge: edge[2].get(curvature, 0), reverse=True) # ascending:False
    curvatures = []
    # edge = list(np.array(sorted_edge)[:, 0:2])
    # edge_set = {(n1,n2) for (n1,n2) in edge}
    neg_e = 0
    edge_set = []
    remain_edges = set()
    
    for (n1,n2,c) in sorted_edge:
        weights.append(c["weight"])
        curvatures.append(c[curvature])
        # edge_set.add((n1, n2))
        if (c[curvature] > 0):
            neg_e += 1
            edge_set.append((n1, n2))
            
    # print(len(edge_set))
            
    top_10_percent_count = int(len(edge_set) * 0.01)
    edge_set = edge_set[:top_10_percent_count]
    
    weights_np = np.array(weights)
    np_curvature = np.array(curvatures)

    # fig, ax = plt.subplots()
    # ax.plot(weights_np[weights_np > 0.2], np_curvature[weights_np > 0.2])
    # plt.savefig("oldw_c.png")
    # plt.close()
    # input()

    # avg_w = weights/neg_e if neg_e > 0 else 0.
    # avg_c = curvatures/neg_e if neg_e > 0 else 0.

    return edge_set


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


# def cal_partition(partition, f):
#     communities = set(partition.values())
#     n = len(communities)
#     f.write(f'Total {n} partitions...\n')

#     nodes_per_c_n = np.zeros((n), dtype=np.int32)

#     nodes_per_c_dict = {}
#     for i in range(n):
#         nodes_per_c_dict[i] = []
        
#     for k, v in partition.items():
#         nodes_per_c_n[v] += 1
#         nodes_per_c_dict[v].append(k)
        
#     for i in range(n):
#         if nodes_per_c_n[i] > 1:
#             f.write(f'Partition {i} has {nodes_per_c_n[i]} nodes.\n')
#             f.write(f'Partition {i} contains nodes {nodes_per_c_dict[i]}.\n\n')
#     f.write('\n')


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
    for layer_num in [2]:
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
        
        net_H.eval()
        
        ori_acc = test(net_H, test_loader)
        
        real_acc = []
        adv_acc = []
        sub_real_acc = []
        sub_adv_acc = []
        model_size = []
        avg_weight = []
        avg_cur = []

        with open(res_path + "net_compress_dir_1over_newW.txt", "w+") as f:    
            f.write(f'For model {dims}.... \n')   
            f.write(f'The origin acc is {ori_acc:.3f}, has edge num of {edges_num}...')
            f.write('\n')
            flag = 1
            index = 0
            
            acc_full, succ_attack, sub_real, sub_adversary = standard_pgd_test.main(sep_dataloader, dims, net_full, net_full, f, mask = "Original full", eps=11/255, alpha=2/255, iters=40)

            real_acc.append(acc_full)
            adv_acc.append(succ_attack)
            sub_real_acc.append(sub_real)
            sub_adv_acc.append(sub_adversary)
            model_size.append(round(1, 0))
            step = 0

            # net_H.load_state_dict(torch.load(res_path + model_name))
            adjacent_m = utils.build_adjm(valid_loader, net_H, nodes_num, dims, device)   
            total_edge = -1
            cur_edge = -1

            while(flag):            
                
                # Create network object
                G = nx.from_numpy_array(adjacent_m, create_using=nx.DiGraph)

                orf = OllivierRicci(G, alpha=0., method='OTD')
                # orf.compute_ricci_flow(iterations=100)
                orf.compute_ricci_curvature()

                G1 = orf.G.copy()
                f.write(f'Start with {G1}.... \n')
                print(G1)
                
                
                # partition = nx.community.greedy_modularity_communities(G1)
                # cal_partition(partition, f, layer_num)
                
                # with open(res_path + graph_file + 'graph_' + str(layer_num) + '_' + str(step) + '.pkl', 'wb') as file:
                #     pickle.dump(G1, file)
                
                if total_edge == -1:
                    total_edge = len(G1.edges())
                    print("total edges are : ", total_edge)
                
                cur_edge = len(G1.edges())
                
                edge_set = show_results(G1, "ricciCurvature")
                
                removed_e = edge_set
                    
                cur_n = "full" + str(layer_num)
                
                if index == 0:
                    model_n = model_name
                else:
                    model_n = cur_n + ".pth"
            
                    
                net_H.load_state_dict(torch.load(model_path + model_n))
                edge_r = Edge_Remove(net_H, dims, 28, G1, "ricciCurvature", model_path)
                edge_r.e_remove(removed_e, cur_n + ".pth")
                
                # test acc
                net_new = FC_MD(dims, layer_num)

                net_new.load_state_dict(torch.load(model_path + cur_n + ".pth"))
                net_new = net_new.to(device)
                
                acc = test(net_new, test_loader)
                
                f.write(f'Test Accuracy after remove {len(removed_e)} edges: {acc:.3f}...\n')
                f.write("\n") 
                
                # or (len(sub_adv_acc)> 0 and sub_adversary < sub_adv_acc[-1]) 
                
                print(f'Remove {len(removed_e)} edges...\n')
                
                if len(removed_e)  == 0 or (acc < 0.8):
                    flag = 0
                else:
                    # torch.save(net_new.state_dict(), res_path + cur_n + ".pth")
                    adjacent_m = utils.build_adjm(valid_loader, net_new, nodes_num, dims, device)
                    
                # f.write('\n')
                # f.write('*'*50 + 'PGD test' + '*'*50)
                # f.write('\n')
                # f.write('\n')
                    
                acc_full, succ_attack, sub_real, sub_adversary = standard_pgd_test.main(sep_dataloader, dims, net_new, net_full, f, mask = model_n, eps=11/255, alpha=2/255, iters=40)

                real_acc.append(acc_full)
                adv_acc.append(succ_attack)
                sub_real_acc.append(sub_real)
                sub_adv_acc.append(sub_adversary)
                model_size.append(round(((cur_edge - len(removed_e))/total_edge), 3))
                print(f'Current remain edges is: {(cur_edge - len(removed_e))}...\n')
                
                index += 1
            
            f.write('\n')
            f.write('\n')
            f.write(f'{model_size}')
            
            f.write('\n')
            f.write('='*100)
        utils.plot_acc(real_acc, adv_acc, str(layer_num) + "newW__w0_eps11_255", res_path, sub_real_acc, sub_adv_acc, model_size)
                    
                            
                        