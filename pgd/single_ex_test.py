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
import community as community_louvain
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.integrate import simps
import pickle
import time

import sys
sys.path.append("..")

import tools.utils as utils
from tools.small_model import FC_MD
from RicciCurvature.OllivierRicci import OllivierRicci
from tools.FC_linear import FC_Linear

os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


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
    4: [784, 15, 25, 20, 15, 10],
    5: [784, 20, 30, 30, 20, 15, 10],
    6: [784, 20, 30, 30, 35, 20, 15, 10],
    7: [784, 30, 30, 40, 50, 30, 25, 20, 10]
}

selected_classes = [0,1,2,3,4,5,6,7,8,9]



def show_results(G, curvature="ricciCurvature", name = ""):
    weights = []
    sorted_edge = sorted(G.edges(data=True), key=lambda edge: edge[2].get(curvature, 0), reverse=False) # ascending
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
        if (c[curvature] < 0):
            neg_e += 1
            edge_set.add((n1, n2))
        else:
            remain_edges.add((n1, n2))

    return edge_set, remain_edges, weights, curvatures


def standard_PGD(model, images, labels, eps=11/255, alpha=2/255, iters=40):
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


def test(n, loader, eps, alpha, iters):    
    n.eval()
    robust_pair = defaultdict(list)
    succ_pair = defaultdict(list)
 
    
    for l in selected_classes:
        for i, (images, labels) in enumerate(loader[l]):
            images = images.to(device)
            labels = labels.to(device)
            output = n(images)
            pred = output.detach().max(1)[1]
            
            adv_img = standard_PGD(n, images, labels, eps, alpha, iters)
            adv_out = n(adv_img)
            adv_pred = adv_out.detach().max(1)[1]
 
            robust_l = pred.eq(labels.view_as(pred)) & adv_pred.eq(labels.view_as(pred))
            
            succ_l = pred.eq(labels.view_as(pred)) & ~adv_pred.eq(labels.view_as(pred))
   
            succ_pair[l].append((images[succ_l].cpu(), adv_img[succ_l].cpu()))
            robust_pair[l].append((images[robust_l].cpu(), adv_img[robust_l].cpu()))
            
        print(f'Finish label {l}....')

    return succ_pair, robust_pair



def build_adjm(img, net, nodes_num, dims, device, metric):
    img = img.to(device)
    edge_array, nodes, _ = net.edge_w_batch(img)
    edge_array = edge_array.cpu().detach().numpy() 
    
    if metric.lower() == "q_ngr" or metric.lower() == "q_inv":
        output = net.get_weights(img)
        output = output.cpu().detach().numpy() 
        
        output[edge_array == 0] = 0.
    elif metric.lower() == "q_exp":
        output = edge_array
    else:
        raise Exception("Invalid graph metric, metric should be {q_ngr, q_inv, q_exp}!")
    
    w_avg = np.mean(output, axis=0) # (edge num,)
    
    # build adjacent matrix
    adjacent_m = np.zeros((nodes_num, nodes_num), dtype=np.float32)

    d_num = len(dims)
    cur_s_col = dims[0]
    cur_e_col = dims[0] + dims[1]

    cur_layer = 1
    start_col = 0
    end_col = dims[1]

    for i in range(nodes_num - dims[d_num-1]):
        # print(f'i : {i}, start col : {cur_s_col}, end_col : {cur_e_col}, from {start_col} to {end_col}')
        adjacent_m[i, cur_s_col : cur_e_col] = w_avg[start_col : end_col]
        
        if (cur_layer < d_num-1 and i == cur_s_col - 1):
            cur_layer += 1
            start_col = end_col
            end_col = end_col + dims[cur_layer]
            cur_s_col = cur_e_col
            cur_e_col = cur_e_col + dims[cur_layer]
        else:
            start_col = end_col
            end_col = end_col + dims[cur_layer]
            
    return adjacent_m, nodes




'''
parameters:
    @ q_NGR, q_INV, q_EXP     --G_Def
    @ model type/name: fc, cnn   --model_type
    @ model path   --model_path
    @ result path  --res_path
    @ example num    --sample_num

'''
def fc_main(args):
    seed = 59
    
    # set random seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    train_loader, test_loader, valid_loader, valid_dataset, test_dataset = utils.get_new_data(selected_classes, data_train, data_test, test_bs=1, valid_num=5000)

    sep_dataloader = utils.sep_label(test_dataset, selected_classes, bs=2000)
    
    eps = [0.03, 0.07, 0.1, 0.2]
    Q = [1]
    
    model_type = args.model_type
    model_pre_name = args.model_name
    res_path = args.mnist_res_path
    model_path = args.model_path
    metric = args.metric
    
    model_full_n = model_type.lower() + model_pre_name.lower()
    
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    
    # build model
    for layer_num in [2,4]:
        for q in Q:
            if model_pre_name.lower() == "ori" or model_pre_name.lower() == "decay":
                model_name = "best_ori_10l_" + str(layer_num) + ".pth"
            elif model_pre_name.lower() == "adv":
                model_name = "pgdtrain_" + str(layer_num) + ".pth"
            else:
                raise Exception("Invalid model name, model name should be {ori, decay, adv}!")
            print(f'Now for model {model_name}....\n')
            
            dims = model_zoo[layer_num]
            net_H = FC_MD(dims, layer_num)

            net_H.load_state_dict(torch.load(model_path + model_name))
            net_H = net_H.to(device)
            
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
        
            for e in eps:
                succ_pair, robust_pair = test(net_H, sep_dataloader, eps=e, alpha=2/255, iters=40)
                
                sample_size = args.sample_num
                
                robust_c = defaultdict(list)
                nonrobust_c = defaultdict(list)
                non_fraction = defaultdict(list)
                rob_fraction = defaultdict(list)
                    
                for l in selected_classes:
                    print(f'For label {l}....\n')
                    # non robust images
                    i = 0
                    for (ori_im, adv_im) in succ_pair[l]:
                        for im in ori_im:
                            adj_m_ori, nodes_ori = build_adjm(im, net_H, nodes_num, dims, device, metric)
                     
                            # Create network object
                            G = nx.from_numpy_array(adj_m_ori, create_using=nx.DiGraph)
            
                            orf = OllivierRicci(G, alpha=0., method="OTD")
                            
                            if metric.lower() == "q_ngr":
                                orf.recal_graph_weight(nodes_ori)
                                
                            elif metric.lower() == "q_inv":
                                orf.recal_graph_weight_w2(nodes_ori)
                
                            elif metric.lower() == "q_exp":
                                orf.recal_qexp(q, nodes_ori)
                            else:
                                raise Exception("Invalid graph metric, metric should be {q_ngr, q_inv, q_exp}!")
    
                            orf.compute_ricci_curvature()
                            G1 = orf.G.copy()
        
                            edge_set, remain_edges, w, c = show_results(G1, "ricciCurvature", name = str(l) + str(i) + '_nonrobust')

                            nonrobust_c[l].append(np.array(c))
                            non_fraction[l].append(len(edge_set)/(len(remain_edges)+len(edge_set)))
                      
                            i += 1
                            
                            if (i % 10 == 0):
                                print(f'Finish {i} graphs....')
                
                            if (i >= sample_size):
                                break
  
                
                    # robust images
                    i = 0
                    for (ori_im, adv_im) in robust_pair[l]:
                        for im in ori_im:
                            adj_m_ori, nodes_ori = build_adjm(im, net_H, nodes_num, dims, device, metric)
                    
                            G = nx.from_numpy_array(adj_m_ori, create_using=nx.DiGraph)
                            orf = OllivierRicci(G, alpha=0., method="OTD")

                            if metric.lower() == "q_ngr":
                                orf.recal_graph_weight(nodes_ori)
                                
                            elif metric.lower() == "q_inv":
                                orf.recal_graph_weight_w2(nodes_ori)
                
                            elif metric.lower() == "q_exp":
                                orf.recal_qexp(q, nodes_ori)
                            else:
                                raise Exception("Invalid graph metric, metric should be {q_ngr, q_inv, q_exp}!")
                
                            orf.compute_ricci_curvature()
                            G1 = orf.G.copy()
                            edge_set, remain_edges, w, c = show_results(G1, "ricciCurvature", name = str(l) + str(i) + '_robust')
            
                            robust_c[l].append(np.array(c))
                            rob_fraction[l].append(len(edge_set)/(len(remain_edges)+len(edge_set)))
                   
                            i += 1
                            
                            if (i % 10 == 0):
                                print(f'Finish {i} graphs....')
                                
                            if (i >= sample_size):
                                break
                    print(f'Finish {i} graphs....')
                            
                    
                with open(res_path + model_full_n + str(e) + metric + str(q) + '_' + str(layer_num) + "frac_robust.pkl", 'wb') as file:
                    pickle.dump(rob_fraction, file)
                with open(res_path + model_full_n + str(e) + metric + str(q) + '_' + str(layer_num) + "frac_norobust.pkl", 'wb') as file:
                    pickle.dump(non_fraction, file)
                    
                with open(res_path + model_full_n + str(e) + metric + str(q) + '_' + str(layer_num) + "curv_robust.pkl", 'wb') as file:
                    pickle.dump(robust_c, file)
                with open(res_path + model_full_n + str(e) + metric + str(q) + '_' + str(layer_num) + "curv_norobust.pkl", 'wb') as file:
                    pickle.dump(nonrobust_c, file)
                        