import torch
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
import numpy as np
import random
import os
import pandas as pd
import torch.nn as nn
from collections import defaultdict

# from GraphRicciCurvature.OllivierRicci import OllivierRicci
import networkx as nx
import torch.nn.functional as F
import matplotlib.pyplot as plt
import community as community_louvain
import statsmodels.api as sm
from scipy.integrate import simps

import sys
sys.path.append("..")

import tools.utils as utils
import tools.cnn_adj_matrix as build_cnn_adj
from tools.LeNet5 import LeNet as LeNet
from tools.LeNet5_custom_v2 import LeNet_custom_v2 as LeNet_custom_v2
from RicciCurvature.OllivierRicci import OllivierRicci


os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")



data_train = MNIST('../data/mnist',
                  train=True,
                  download=True,
                  transform=transforms.Compose([
                      transforms.Resize((32, 32)),
                      transforms.ToTensor()]))

data_test = MNIST('../data/mnist',
                  train=False,
                  download=True,
                  transform=transforms.Compose([
                      transforms.Resize((32, 32)),
                      transforms.ToTensor()]))



file_path = "edge_v/"
model_path = "models/"
res_path = "w_c/"

# model_path = "advtrain/models/"
# res_path = "advtrain/"


nodes_num = 9118

model_dims = {
    1: {"name": "input", "dim": {"channel": 1, "out_size": 32}},
    2: {"name": "cnn", "dim": {"channel": 6, "kernel": 5, "stride": 1, "out_size": 28}},
    3: {"name": "pooling", "dim": {"channel": 6, "kernel": 2, "stride": 2, "out_size": 14}},
    4: {"name": "cnn", "dim": {"channel": 16, "kernel": 5, "stride": 1, "out_size": 10}},
    5: {"name": "pooling", "dim": {"channel": 16, "kernel": 2, "stride": 2, "out_size":5}},
    6: {"name": "fc", "dim": {"out_size": 120}},
    7: {"name": "fc", "dim": {"out_size": 84}},
    8: {"name": "fc", "dim": {"out_size": 10}}
}



def show_results(G, curvature="ricciCurvature"):
    weights = []
    sorted_edge = sorted(G.edges(data=True), key=lambda edge: edge[2].get(curvature, 0), reverse=True) # ascending
    curvatures = []
    # edge = list(np.array(sorted_edge)[:, 0:2])
    # edge_set = {(n1,n2) for (n1,n2) in edge}
    neg_e = 0
    edge_set = set()
    remain_edges = set()
    
    for (n1,n2,c) in sorted_edge:
        weights.append(c["weight"])
        curvatures.append(c[curvature])
        # edge_set.add((n1, n2))
        if (c[curvature] > 0):
            neg_e += 1
            
            edge_set.add((n1, n2))
        else:
            remain_edges.add((n1, n2))
            
    print(f'The minimun curvature is {np.min(curvatures)}, maximun is {np.max(curvatures)}...')

    # avg_w = weights/neg_e if neg_e > 0 else 0.
    # avg_c = curvatures/neg_e if neg_e > 0 else 0.
    
    # weights_np = np.array(weights)
    # np_curvature = np.array(curvatures)

    # fig, ax = plt.subplots()
    # ax.plot(weights_np[weights_np > 0.2], np_curvature[weights_np > 0.2])
    # plt.savefig("w_c/" + name + "neww_c.png")
    # plt.close()

    return edge_set, remain_edges, weights, curvatures


def standard_PGD(model, images, labels, eps=11/255, alpha=2/255, iters=40):
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


def test(n, loader, eps, alpha, iters):    
    n.eval()
    robust_pair = defaultdict(list)
    succ_pair = defaultdict(list)
    sample = 20
    finish = set()
    
    for l in selected_classes:
        for i, (images, labels) in enumerate(loader[l]):
            images = images.to(device)
            labels = labels.to(device)
            output = n(images)
            pred = output.detach().max(1)[1]
            
            adv_img = standard_PGD(n, images, labels, eps, alpha, iters)
            adv_out = n(adv_img)
            adv_pred = adv_out.detach().max(1)[1]
            
            robust_l = pred.eq(labels.view_as(pred)) & adv_pred.eq(pred)
            
            succ_l = pred.eq(labels.view_as(pred)) & ~adv_pred.eq(pred)
            
            succ_pair[l].append((images[succ_l].cpu(), adv_img[succ_l].cpu()))
            robust_pair[l].append((images[robust_l].cpu(), adv_img[robust_l].cpu()))
            
            # if (l == pred.cpu().item()):
            #     if len(succ_pair[labels.cpu().item()]) < sample and pred.cpu().item() != adv_pred.cpu().item():
            #         # print(f'Correct label is {labels.cpu().item()}, pred label is {pred.cpu().item()}, adv label is {adv_pred.cpu().item()}....')
            #         succ_pair[l].append((images.cpu(), adv_img.cpu()))
            #     elif len(robust_pair[labels.cpu().item()]) < sample and pred.cpu().item() == adv_pred.cpu().item():
            #         robust_pair[l].append((images.cpu(), adv_img.cpu()))
            #     elif len(succ_pair[labels.cpu().item()]) >= sample and len(robust_pair[labels.cpu().item()]) >= sample:
            #         break
                    
            # if len(finish) == 10:
            #     break
        print(f'Finish label {l}....')

    # print(f'Test Accuracy for label {l}: {(float(total_correct) / len(loader.dataset)):.3f}')
    # acc = float(total_correct) / len(loader.dataset)
    return succ_pair, robust_pair


def single_test(n, im, l, eps, alpha, iters):
    n.eval()
    
    im = im.to(device)
    output = n(im)
    labels = output.detach().max(1)[1]
    
    adv_img = standard_PGD(n, im, labels, eps, alpha, iters)
    adv_out = n(adv_img)
    adv_pred = adv_out.detach().max(1)[1]
    
    print(f'Correct label is {l}, predition label is {labels.cpu().item()}, attacked img prediction is {adv_pred.cpu().item()}....\n')
    return adv_pred.cpu().item()


def cal_edge_v(net, im, device):     
    net.eval()

    d = im.to(device)
    
    edge_array, nodes = net.edge_w_batch(d)
    edge_array = edge_array.cpu().detach().numpy() 
    output = net.get_weights(d)
    output = output.cpu().detach().numpy() 
    
    output[edge_array == 0] = 0.
    w_avg = np.mean(output, axis=0)
    
    return w_avg, nodes
    

def build_cnn_adj(nodes_num, model_dims, net, im, device, file_n = None):
    
    # raw value: edge weights
    if (file_n != None):
        edge_weights = pd.read_csv(file_n, index_col=0)
        w_avg = edge_weights.values.squeeze()
    else:
        w_avg, nodes = cal_edge_v(net, im, device)
    # w_avg = np.abs(e_weights) 
    
    # build adjacent matrix
    adjacent_m = np.zeros((nodes_num, nodes_num), dtype=np.float32)

    current_l = 1
    start_col = 0
    cur_s_col = 0
    nodes_total = 0

    layer_num = len(model_dims)

    for i in range(1, layer_num):
        current_l = i
        next_l= i + 1

        cur_name = model_dims[current_l]["name"]
        cur_dim = model_dims[current_l]["dim"]
        cur_size = cur_dim['out_size']

        cur_channel = 1 if (cur_name == "fc") else cur_dim['channel']

        if cur_name == "input":
            cur_nodes = cur_channel * cur_size**2
        else:
            cur_nodes = cur_size if (cur_name == "fc") else cur_dim['channel']*(cur_size**2)

        nxt_name = model_dims[next_l]["name"]
        nxt_dim = model_dims[next_l]["dim"]
        out_size = nxt_dim['out_size']

        # cnn layer
        if (nxt_name == "cnn" or nxt_name == "pooling"):
            k = nxt_dim['kernel']
            s = nxt_dim['stride']
            c = nxt_dim['channel']
            
            step = k**2      
            n = 0
            tensor_2d = torch.arange(cur_nodes).reshape(1,cur_channel,cur_size,cur_size).float()
            
            if (nxt_name == "cnn"):
                indices = F.unfold(tensor_2d, (k,k), stride = s).transpose(1,2).int()
                end_col = start_col + step*cur_channel
                for ur_c in range(c): 
                    for l in range(indices.shape[1]):
                        cur_idx = indices[0,l] + nodes_total 
                        cur_idx = cur_idx.tolist()
                        assert(len(cur_idx) == step*cur_channel)
                        adjacent_m[cur_idx, nodes_total+cur_nodes+n] = w_avg[start_col : end_col]
            
                        start_col = end_col
                        end_col = start_col + step*cur_channel
                        n += 1
            else:
                indices = F.unfold(tensor_2d, (k,k), stride = s)
                i_unf = indices.view(1, cur_channel, k*k, -1).transpose(2,3)
                indices = i_unf.reshape(i_unf.shape[0], i_unf.shape[1]*i_unf.shape[2], i_unf.shape[3]).int()
                
                for l in range(indices.shape[1]):
                    end_col = start_col + step
                    cur_idx = indices[0,l] + nodes_total 
                    cur_idx = cur_idx.tolist()
                    assert(len(cur_idx) == step)
                    adjacent_m[cur_idx, nodes_total+cur_nodes+n] = w_avg[start_col : end_col]

                    start_col = end_col
                    end_col = start_col + step
                    n += 1
                    
            nodes_total += cur_nodes
            # print(start_col)  

        # fc layer
        elif (nxt_name == "fc"):  
            cur_s_col = nodes_total + cur_nodes
            cur_e_col = nodes_total + cur_nodes + out_size

            end_col = start_col + out_size

            for node in range(nodes_total, nodes_total + cur_nodes, 1):
                # print(f'i : {node}, start col : {cur_s_col}, end_col : {cur_e_col}, from {start_col} to {end_col}')
                adjacent_m[node, cur_s_col : cur_e_col] = w_avg[start_col : end_col]        
                
                start_col = end_col
                end_col = end_col + out_size            
                    
            nodes_total += cur_nodes
        # print(nodes_total)
        
    return adjacent_m, nodes


def cal_partition(G1, f):
    partition = community_louvain.best_partition(G1)
    communities = set(partition.values())
    n = len(communities)
    f.write(f'Total {n} partitions...\n')

    nodes_per_c_n = np.zeros((n), dtype=np.int32)

    nodes_per_c_dict = defaultdict(list)
    
    big_p = defaultdict(list)
    
    for k, v in partition.items():
        nodes_per_c_n[v] += 1
        nodes_per_c_dict[v].append(k)
        
        if nodes_per_c_n[v] > 5:
            big_p[v] = nodes_per_c_dict[v]


    for k, v in big_p.items():
        f.write(f'This partition has {nodes_per_c_n[k]} nodes: {v}...\n')
        max_partition = np.array(v)
        cur_l = 1
        pre_n = 0
        
        while cur_l <= len(model_dims):
            cur_name = model_dims[cur_l]["name"]
            cur_dim = model_dims[cur_l]["dim"]
            cur_size = cur_dim['out_size']

            cur_channel = 1 if (cur_name == "fc") else cur_dim['channel']

            if cur_name == "input":
                cur_nodes = cur_channel * cur_size**2
            else:
                cur_nodes = cur_size if (cur_name == "fc") else cur_dim['channel']*(cur_size**2)
            
            cur_l_n = max_partition[max_partition >= pre_n]
            cur_l_n = cur_l_n[cur_l_n < pre_n+cur_nodes]
            
            f.write(f'The current layer is {cur_l}, it has {len(cur_l_n)} nodes...\n')
            cur_l += 1
            pre_n += cur_nodes
        
        f.write('\n')



def draw_cdf(robust, nonrobust, l, mark = ''):    
    robust_area = []
    norobust_area = []
    
    for l in selected_classes:
        fig, ax = plt.subplots()
        area = 0.
        count = 0
        for c in robust[l]:
            d = c[c > -50]
            d = np.sort(d)
            
            ecdf = sm.distributions.ECDF(d)
            x = np.linspace(-50, 0.01, num=1000)
            y = ecdf(x)
            
            area += simps(y, x, dx=0.001)
            count += 1
            
            ax.plot(x, y, color = 'red', linewidth = 0.5)
            # ax.hist(d, bins=100, alpha=0.8, label=str(l), color = 'red')
        area = area/count if count > 0 else 0.
        robust_area.append(area)
        
        area = 0.
        count = 0
        for c in nonrobust[l]:
            d = c[c > -50]
            d = np.sort(d)
            
            ecdf = sm.distributions.ECDF(d)
            x = np.linspace(-50, 0.01, num=1000)
            y = ecdf(x)
            
            area += simps(y, x, dx=0.001)
            count += 1
            ax.plot(x, y, color = 'blue', linewidth = 0.5)
            # ax.hist(d, bins=100, alpha=0.5, label=str(l), color = 'skyblue')
        area = area/count if count > 0 else 0.
        robust_area.append(area)
    
        plt.savefig(res_path + str(l) + mark +  "_2new_wc.png")
        plt.close()
        
        print(f'Finish label {l}..')
    
    fig1, ax1 = plt.subplots()
    ax1.plot(range(len(robust_area)), robust_area, color = 'red', linewidth = 0.5)
    ax1.plot(range(len(norobust_area)), norobust_area, color = 'skyblue', linewidth = 0.5)
    plt.savefig(res_path + mark +  "_new_area.png")
    plt.close()
    

if __name__ == '__main__':
    seed = 29
    
    # set random seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    selected_classes = [0,1,2,3,4,5,6,7,8,9]
    train_loader, test_loader, valid_loader, valid_dataset, test_dataset = utils.get_new_data(selected_classes, data_train, data_test, valid_num = 2000, test_bs=1)

    # sep_dataloader = utils.sep_label(valid_dataset, selected_classes, bs=1)
    sep_dataloader = utils.sep_label(test_dataset, selected_classes, bs=2000)
    
    
    # build model
    # origin model
    model_name= "mnist_relu.pth"
    # model_name= "mnist_cnn.pth"
    # model_name= "pgdtrain_lenet.pth"
    # model = LeNet()
    # model.load_state_dict(torch.load(model_path + model_name))
    # model = model.to(device)
    
    net_H = LeNet_custom_v2(model_dims, None, device)
    net_H.load_state_dict(torch.load(model_path + model_name))
    net_H = net_H.to(device)
        
    succ_pair, robust_pair = test(net_H, sep_dataloader, eps=0.2, alpha=0.1, iters=100)
        
    sample_size = 10
    e1 = []
    e2 = []
    e3 = []
    e4 = []
    c1 = []
    c2 = []
    c3 = []
    c4 = []
    
    robust_c = defaultdict(list)
    nonrobust_c = defaultdict(list)
    with open(res_path + "partitions_compabs.txt", "w+") as f:
        for l in selected_classes:

            f.write(f'Label {l}....\n')
            var_w = []
            var_w1 = []
            avg_c = []
            avg_c1 = []
            total = 0.
            i = 0.
            # nonrobust images
            f.write(f'\n NonRobust img pair ... \n')
            for (ori_im, adv_im) in succ_pair[l]:
                for im in ori_im:
                    adj_m_ori, nodes_ori = build_cnn_adj(nodes_num, model_dims, net_H, im[np.newaxis,:], device)
                    
                    G = nx.from_numpy_array(adj_m_ori, create_using=nx.DiGraph)
                    orf = OllivierRicci(G, alpha=0., method='OTD')
                    orf.recal_graph_weight(nodes_ori)
                    orf.compute_ricci_curvature()
                    G1 = orf.G.copy()
                    
                    f.write(f'origina graph: {G1}\n')
                    edge_set, remain_edges, w, c = show_results(G1, "ricciCurvature")
                    
                    e1.append(len(edge_set))
                    f.write(f'Has {len(edge_set)} positive edges...\n')
                    c1.append(np.sum(c))
                    f.write(f'The total positive curvature is {np.sum(c)}, average is {np.mean(c)}, variance is {np.var(c)}....\n')
                    f.write(f'The total positive edge weight is {np.sum(w)}, average is {np.mean(w)}, variance is {np.var(w)}....\n')
                    nonrobust_c[l].append(np.array(c))
                    
                    i += 1
                    if (i % 10 == 0):
                        print(f'Finish {i} graphs....')
                                
                    if (i >= sample_size):
                        break
                # avg_c.append(np.mean(c))
                # var_w.append(np.var(w))
                # cal_partition(G1, f)
                
                # f.write(f'\nADV graph\n')
                # adj_m_adv = build_cnn_adj(nodes_num, model_dims, net_H, adv_im, device)
                # adj_m_adv[adj_m_adv < 0] = 0.
                
                # G = nx.from_numpy_array(adj_m_adv)
                # orf = OllivierRicci(G, alpha=0.5, verbose="TRACE")
                # orf.compute_ricci_curvature()
                # G1 = orf.G.copy()
                # f.write(f'adv graph: {G1}\n') 
                # # cal_partition(G1, f)
                # edge_set, remain_edges, w, c = show_results(G1, "ricciCurvature")
                # e2.append(len(edge_set))
                # f.write(f'Has {len(edge_set)} positive edges...\n')
                # c2.append(np.sum(c))
                # f.write(f'The total positive curvature is {np.sum(c)}, average is {np.mean(c)}, variance is {np.var(c)}....\n')
                # f.write(f'The total positive edge weight is {np.sum(w)}, average is {np.mean(w)}, variance is {np.var(w)}....\n')

            f.write(f'\nRobust img pair ... \n')
            f.write(f'Label {l}....\n')
            i = 0
            for (ori_im, adv_im) in robust_pair[l]:
                for im in ori_im:
                    adj_m_ori, nodes_ori = build_cnn_adj(nodes_num, model_dims, net_H, im[np.newaxis,:], device)
                    
                    G = nx.from_numpy_array(adj_m_ori, create_using=nx.DiGraph)
                    orf = OllivierRicci(G, alpha=0., method = 'OTD')
                    orf.recal_graph_weight(nodes_ori)
                    orf.compute_ricci_curvature()
                    G1 = orf.G.copy()
                    
                    f.write(f'origina graph: {G1}\n')
                    edge_set, remain_edges, w, c = show_results(G1, "ricciCurvature")
                    
                    e3.append(len(edge_set))
                    f.write(f'Has {len(edge_set)} positive edges...\n')
                    c3.append(np.sum(c))
                    f.write(f'The total positive curvature is {np.sum(c)}, average is {np.mean(c)}, variance is {np.var(c)}....\n')
                    f.write(f'The total positive edge weight is {np.sum(w)}, average is {np.mean(w)}, variance is {np.var(w)}....\n')
                    robust_c[l].append(np.array(c))
                    
                    i += 1
                    
                    if (i % 10 == 0):
                        print(f'Finish {i} graphs....')
                                
                    if (i >= sample_size):
                        break
                # avg_c1.append(np.mean(c))
                # var_w1.append(np.var(w))
                # cal_partition(G1, f)

                # f.write(f'\nADV graph\n')
                # adj_m_adv = build_cnn_adj(nodes_num, model_dims, net_H, adv_im, device)
                # adj_m_adv[adj_m_adv < 0] = 0.
                
                # G = nx.from_numpy_array(adj_m_adv)
                # orf = OllivierRicci(G, alpha=0.5, verbose="TRACE")
                # orf.compute_ricci_curvature()
                # G1 = orf.G.copy()
                # f.write(f'adv graph: {G1}\n') 
                # edge_set, remain_edges, w, c = show_results(G1, "ricciCurvature")
                # e4.append(len(edge_set))
                # f.write(f'Has {len(edge_set)} positive edges...\n')
                # c4.append(np.sum(c))
                # f.write(f'The total positive curvature is {np.sum(c)}, average is {np.mean(c)}, variance is {np.var(c)}....\n')
                # f.write(f'The total positive edge weight is {np.sum(w)}, average is {np.mean(w)}, variance is {np.var(w)}....\n')
                # # cal_partition(G1, f)
                                
            # f.write('\n')
            # f.write('='*50)
            # f.write('\n\n')
        draw_cdf(robust_c, nonrobust_c, l, mark='1overcdf')
    # utils.plot_acc(e1, e2, 'lenet5_compare_edge_abs', res_path, acc_3 = e3, acc_4 = e4)
    # utils.plot_acc(c1, c2, 'lenet5_compare_curvature_abs', res_path, acc_3 = c3, acc_4 = c4)