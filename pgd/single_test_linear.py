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
# from RicciCurvature.q_exponential import q_exponential

os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")


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


layers = [2, 4, 5, 6, 7]

model_zoo = {
    2: [784, 20, 15, 10],
    4: [784, 15, 25, 20, 15, 10],
    5: [784, 20, 30, 30, 20, 15, 10],
    6: [784, 20, 30, 30, 35, 20, 15, 10],
    7: [784, 30, 30, 40, 50, 30, 25, 20, 10]
}

selected_classes = [0,1,2,3,4,5,6,7,8,9]
    
file_path = "edge_v/"
model_path = "models/"
res_path = "res/"


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
            
            # adv_img1 = standard_PGD(n, images, labels, 0.05, alpha, iters)
            # adv_out1 = n(adv_img1)
            # adv_pred1 = adv_out1.detach().max(1)[1]
 
            robust_l = pred.eq(labels.view_as(pred)) & adv_pred.eq(labels.view_as(pred))
            
            succ_l = pred.eq(labels.view_as(pred)) & ~adv_pred.eq(labels.view_as(pred))
   
            succ_pair[l].append((images[succ_l].cpu(), adv_img[succ_l].cpu()))
            robust_pair[l].append((images[robust_l].cpu(), adv_img[robust_l].cpu()))

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



def build_adjm(img, net, nodes_num, dims, device):
    img = img.to(device)
    edge_array, nodes = net.edge_w_batch(img)
    edge_array = edge_array.cpu().detach().numpy() 
    
    output = net.get_weights(img)
    output = output.cpu().detach().numpy() 
    
    output[edge_array == 0] = 0.
    
    w_avg = np.mean(output, axis=0) # (edge num,)
    # print(f'the max is {np.max(w_avg)}, min is {np.min(w_avg)}')
    # w_avg = np.abs(w_avg) # absolate edge value for one image
    # w_avg[w_avg != 0] = 1/w_avg[w_avg != 0]
    
    # build adjacent matrix
    adjacent_m = np.zeros((nodes_num, nodes_num), dtype=np.float32)
    # adjacent_m *= -10000

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



# change -10000 to 0 
def q_adjust(adj, nodes, q_exp, dims):
    node_v = nodes[0].cpu()
    ind = torch.where(node_v <= 0)[0]
    # ind = np.where(adj == 0)[0]
    new_adj = np.zeros_like(adj)
    adj[:, ind[ind >= dims[0]]] *= -1 
    new_adj = 1.0/abs(q_exp.q_exponential_series(adj))
    new_adj[adj == -10000] = 0
    return new_adj
    



def draw_cdf(robust, nonrobust, layer, mark = ''):    
    robust_area = []
    norobust_area = []
    ro_pkl = defaultdict(list)
    no_pkl = defaultdict(list)
    
    for l in selected_classes:
        fig, ax = plt.subplots()
        area = 0.
        count = 0
        for c in robust[l]:
            d = c
            d = np.sort(d)
            
            ecdf = sm.distributions.ECDF(d)
            x = np.linspace(-50, 1, num=1000)
            y = ecdf(x)
            
            ro_pkl[l].append((x,y))
            
            area += simps(y, x, dx=0.001)
            count += 1
            
            # ax.plot(x, y, color = 'red', linewidth = 1)
            # ax.hist(d, bins=100, alpha=0.8, label=str(l), color = 'red')
        area = area/count if count > 0 else 0.
        robust_area.append(area)
        
        area = 0.
        count = 0
        for c in nonrobust[l]:
            d = c
            d = np.sort(d)
            
            ecdf = sm.distributions.ECDF(d)
            x = np.linspace(-50, 1, num=1000)
            y = ecdf(x)
            
            no_pkl[l].append((x,y))
            
            area += simps(y, x, dx=0.001)
            count += 1
            # ax.plot(x, y, color = 'blue', linewidth = 1)
            # ax.hist(d, bins=100, alpha=0.5, label=str(l), color = 'skyblue')
        
        area = area/count if count > 0 else 0.
        norobust_area.append(area)
    
        # plt.savefig(res_path + str(l) + mark +  "_4new_wc.png")
        # plt.close()
        
        print(f'Finish label {l}..')
        
    with open(res_path + mark + "_robust.pkl", 'wb') as file:
        pickle.dump(ro_pkl, file)
    with open(res_path + mark + "_norobust.pkl", 'wb') as file:
        pickle.dump(no_pkl, file)
    
    # fig1, ax1 = plt.subplots()
    # ax1.plot(range(len(robust_area)), robust_area, color = 'red', linewidth = 1)
    # ax1.plot(range(len(norobust_area)), norobust_area, color = 'skyblue', linewidth = 1)
    # plt.savefig(res_path + mark + "_2_area.png")
    # plt.close()
    

    

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
    
    train_loader, test_loader, valid_loader, valid_dataset, test_dataset = utils.get_new_data(selected_classes, data_train, data_test, test_bs=1, valid_num=5000)

    sep_valloader = utils.sep_label(valid_dataset, selected_classes, bs=2000)
    sep_dataloader = utils.sep_label(test_dataset, selected_classes, bs=2000)
    
    eps = [0.03, 0.05, 0.07, 0.1, 0.15, 0.2]
    Q = [1]
    
    # build model
    for layer_num in [2]:
        for q in Q:
            # q_exp = q_exponential(q)
            with open(res_path + "q=" + str(q) + "_7_" + str(layer_num) + "_linear.txt", "w+") as f:
                # # model_name = "best_ori_10l_" + str(layer_num) + ".pth"
                model_name = "best_ori_" + str(layer_num) + "_linear.pth"
                # model_name = "best_adv_" + str(layer_num) + "_linear.pth"
                
                # model_name = "pgdtrain_" + str(layer_num) + ".pth"
                print(f'Now for model {model_name}....\n')
                
                dims = model_zoo[layer_num]
                net_H = FC_Linear(dims, layer_num)

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
                
                # start1 = time.perf_counter()
                # total_t = 0.
                for e in eps:
                    f.write(f'eps = {e}....\n\n')
                    succ_pair, robust_pair = test(net_H, sep_dataloader, eps=e, alpha=2/255, iters=40)
                    
                    sample_size = 50
                    
                    robust_c = defaultdict(list)
                    nonrobust_c = defaultdict(list)
                    adv_robust_c = defaultdict(list)
                    adv_nonrobust_c = defaultdict(list)
                    non_fraction = defaultdict(list)
                    rob_fraction = defaultdict(list)
                    
                    robust_e = defaultdict(list)
                    nonrobust_e = defaultdict(list)
                        
                    for l in selected_classes:
                        print(f'For label {l}....\n')
                        f.write(f'Label {l}....\n')
                        
                        var_w = []
                        var_w1 = []
                        avg_c = []
                        avg_c1 = []

                
                        f.write(f'\nNonRobust img ... \n')
                        # non robust images
                        i = 0
                        for (ori_im, adv_im) in succ_pair[l]:
                            for im in ori_im:
                                # start = time.perf_counter()
                                adj_m_ori, nodes_ori = build_adjm(im, net_H, nodes_num, dims, device)
                                # adj = q_adjust(adj_m_ori, nodes_ori, q_exp, dims)
                                # adj_m_adv, nodes_adv = build_adjm(adv_im, net_H, nodes_num, dims, device)
                                # adjacent_m_gpu = torch.tensor(adj_m_ori, dtype=torch.float32).to(device)
                                # get_new_edge_v = net_H.get_new_edge_v(nodes_ori, adjacent_m_gpu).cpu()
                                # adj_m, nodes_diff = build_adjm((adv_im-ori_im), net_H, nodes_num, dims, device)

                                # Create network object
                                G = nx.from_numpy_array(adj_m_ori, create_using=nx.DiGraph)
                                # print(G)
                
                                orf = OllivierRicci(G, alpha=0., method="OTD")
                                orf.recal_graph_weight_w2(nodes_ori)
                                orf.compute_ricci_curvature()
                                G1 = orf.G.copy()
            
                                f.write(f'origina graph: {G1}\n')
                                edge_set, remain_edges, w, c = show_results(G1, "ricciCurvature", name = str(l) + str(i) + '_nonrobust')

                                f.write(f'Has {len(edge_set)} negative edges..., the fraction is {len(edge_set)/(len(remain_edges)+len(edge_set)):.3f}....\n\n')
                                # # f.write(f'The total negative curvature is {np.sum(c)}, average is {np.mean(c)}, variance is {np.var(c)}....\n')
                                # # f.write(f'The total negative edge weight is {np.sum(w)}, average is {np.mean(w)}, variance is {np.var(w)}....\n\n')
                                # # avg_c.append(np.mean(c))
                                # # var_w.append(np.var(w))
                                nonrobust_c[l].append(np.array(c))
                                non_fraction[l].append(len(edge_set)/(len(remain_edges)+len(edge_set)))
                                # nonrobust_e[l].append(np.array(w))
                                
                                i += 1
                                
                                if (i % 10 == 0):
                                    print(f'Finish {i} graphs....')
                                
                                # end = time.perf_counter()
                                # total_t += (end - start)
                                # f.write(f'Time cost for each image: {end - start : .3f} s\n')
                                
                                if (i >= sample_size):
                                    break
                                
                # end1 = time.perf_counter()
                # f.write(f'Time cost: {end1 - start1: .3f} s \n')
                # f.write(f'The averge time for each image: {total_t / (sample_size*len(selected_classes)) : .3f} s\n\n')
                            # partition = nx.community.greedy_modularity_communities(G1)
                            # cal_partition(partition, f, layer_num)
                            # i = 0
                            # for im in adv_im:
                            #     f.write(f'\nADV graph\n')
                            #     adj_m_adv, nodes_ori = build_adjm(im, net_H, nodes_num, dims, device)
                                
                            #     G = nx.from_numpy_array(adj_m_adv, create_using=nx.DiGraph)
                            #     orf = OllivierRicci(G, alpha=0.,  method="custom", nodes_v = nodes_ori)
                            #     orf.compute_ricci_curvature()
                            #     G1 = orf.G.copy()
                                
                            #     f.write(f'origina graph: {G1}\n')
                            #     edge_set, remain_edges, w, c = show_results(G1, "ricciCurvature", name = str(l) + str(i) + '_nonrobust')

                            #     f.write(f'Has {len(edge_set)} negative edges..., the fraction is {len(edge_set)/(len(remain_edges)+len(edge_set)):.3f}....\n\n')
                            #     # # f.write(f'The total negative curvature is {np.sum(c)}, average is {np.mean(c)}, variance is {np.var(c)}....\n')
                            #     # # f.write(f'The total negative edge weight is {np.sum(w)}, average is {np.mean(w)}, variance is {np.var(w)}....\n\n')
                            #     # # avg_c.append(np.mean(c))
                            #     # # var_w.append(np.var(w))
                            #     adv_nonrobust_c[l].append(np.array(c))
                                
                            #     i += 1
                            #     if (i % 10 == 0):
                            #         print(f'Finish {i} graphs....')
                                    
                            #     if (i >= sample_size):
                            #         break
                            # partition = nx.community.greedy_modularity_communities(G1)
                            # cal_partition(partition, f, layer_num)
                            
                            # print(f'The total edge num is {len(G1.edges())}, has {len(edge_set)} negative curvature edges...')
                        
                    
                        # robust images
                        f.write(f'\nRobust img ... \n')
                        i = 0
                        for (ori_im, adv_im) in robust_pair[l]:
                            for im in ori_im:
                                adj_m_ori, nodes_ori = build_adjm(im, net_H, nodes_num, dims, device)
                                # adj = q_adjust(adj_m_ori, nodes_ori, q_exp, dims)
                                # adj_m_adv, nodes_adv = build_adjm(adv_im, net_H, nodes_num, dims, device)
                                # adjacent_m_gpu = torch.tensor(adj_m_ori, dtype=torch.float32).to(device)
                                # get_new_edge_v = net_H.get_new_edge_v(nodes_ori, adjacent_m_gpu).cpu()
                                # adj_m, nodes_diff = build_adjm((adv_im-ori_im), net_H, nodes_num, dims, device)
                                
                                G = nx.from_numpy_array(adj_m_ori, create_using=nx.DiGraph)
                                orf = OllivierRicci(G, alpha=0., method="OTD")
                                orf.recal_graph_weight_w2(nodes_ori)
                                orf.compute_ricci_curvature()
                                G1 = orf.G.copy()
                                
                                f.write(f'origina graph: {G1}\n')
                                edge_set, remain_edges, w, c = show_results(G1, "ricciCurvature", name = str(l) + str(i) + '_robust')
                
                                f.write(f'Has {len(edge_set)} negative edges..., the fraction is {len(edge_set)/(len(remain_edges)+len(edge_set)):.3f}....\n\n')
                                # f.write(f'The total negative curvature is {np.sum(c)}, average is {np.mean(c)}, variance is {np.var(c)}....\n')
                                # f.write(f'The total negative edge weight is {np.sum(w)}, average is {np.mean(w)}, variance is {np.var(w)}....\n\n')
                                # avg_c1.append(np.mean(c))
                                # var_w1.append(np.var(w))
                                robust_c[l].append(np.array(c))
                                rob_fraction[l].append(len(edge_set)/(len(remain_edges)+len(edge_set)))
                                # robust_e[l].append(np.array(w))
                                
                                i += 1
                                
                                if (i % 10 == 0):
                                    print(f'Finish {i} graphs....')
                                    
                                if (i >= sample_size):
                                    break
                                

                            # partition = nx.community.greedy_modularity_communities(G1)
                            # cal_partition(partition, f, layer_num)
                        # i = 0
                        # for im in adv_im:  
                        #     f.write(f'\nADV graph\n')
                        #     adj_m_adv, nodes_ori = build_adjm(im, net_H, nodes_num, dims, device)
                            
                        #     G = nx.from_numpy_array(adj_m_adv, create_using=nx.DiGraph)
                        #     orf = OllivierRicci(G, alpha=0.,  method="custom", nodes_v = nodes_ori)
                        #     orf.compute_ricci_curvature()
                        #     G1 = orf.G.copy()
                            
                        #     f.write(f'origina graph: {G1}\n')
                        #     edge_set, remain_edges, w, c = show_results(G1, "ricciCurvature", name = str(l) + str(i) + '_nonrobust')

                        #     f.write(f'Has {len(edge_set)} negative edges..., the fraction is {len(edge_set)/(len(remain_edges)+len(edge_set)):.3f}....\n\n')
                        #     # # f.write(f'The total negative curvature is {np.sum(c)}, average is {np.mean(c)}, variance is {np.var(c)}....\n')
                        #     # # f.write(f'The total negative edge weight is {np.sum(w)}, average is {np.mean(w)}, variance is {np.var(w)}....\n\n')
                        #     # # avg_c.append(np.mean(c))
                        #     # # var_w.append(np.var(w))
                        #     adv_robust_c[l].append(np.array(c))
                            
                        #     i += 1
                        #     if (i % 10 == 0):
                        #         print(f'Finish {i} graphs....')
                                
                        #     if (i >= sample_size):
                        #         break
                            
                        # f.write('\n')
                        # f.write('='*50)
                        # f.write('\n\n')
                        
                    with open(res_path + str(e) + "_w7_" + str(q) + '_' + str(layer_num) + "frac_robust_linear.pkl", 'wb') as file:
                        pickle.dump(rob_fraction, file)
                    with open(res_path + str(e) + "_w7_" + str(q) + '_' + str(layer_num) + "frac_norobust_linear.pkl", 'wb') as file:
                        pickle.dump(non_fraction, file)
                        
                    with open(res_path + "_w7_" + str(q) + "_ori_c_" + str(e) + '_' + str(layer_num) + "_robust_linear.pkl", 'wb') as file:
                        pickle.dump(robust_c, file)
                    with open(res_path + "_w7_" + str(q) + "_ori_c_" + str(e) + '_' + str(layer_num) + "_norobust_linear.pkl", 'wb') as file:
                        pickle.dump(nonrobust_c, file)
                        
