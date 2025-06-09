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

import sys
sys.path.append("..")

import tools.utils as utils
from tools.small_model_tanh import FC_MD
from tools.FC_linear import FC_Linear
from tools.graph_curvature import graph_curvature_main_torch
from tools.draw_net import DrawNN
from tools.edge_remove import Edge_Remove
from tools.get_c import get_c

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


def test_clean_perlabel(n, loader, device = 'cuda'):
    n.eval()
    correct_per_label = defaultdict(int)
    total_per_label = defaultdict(int)
    
    for l in selected_classes:
        for i, (images, labels) in enumerate(loader[l]):
            images = images.to(device)
            labels = labels.to(device)
            output = n(images)
            pred = output.detach().max(1)[1]
    
            correct_per_label[l] += pred.eq(labels).sum().item()
            total_per_label[l] += labels.size(0)  

    # Calculate accuracy per label
    accuracies = {}
    for label in selected_classes:
        if total_per_label[label] > 0:
            accuracies[label] = correct_per_label[label] / total_per_label[label]
        else:
            accuracies[label] = 0.0
            
    return accuracies


def test_adversarial_perlabel_perE(n, loader, eps, alpha, iters, device = 'cuda'):
    n.eval()
    correct_per_label = defaultdict(int)
    total_per_label = defaultdict(int)
    
    for l in selected_classes:
        for i, (images, labels) in enumerate(loader[l]):
            images = images.to(device)
            labels = labels.to(device)

            adv_img = standard_PGD(n, images, labels, device, eps, alpha, iters)
            adv_out = n(adv_img)
            adv_pred = adv_out.detach().max(1)[1]
            
            correct_per_label[l] += adv_pred.eq(labels).sum().item()
            total_per_label[l] += labels.size(0)  

    accuracies = {}
    for label in selected_classes:
        if total_per_label[label] > 0:
            accuracies[label] = correct_per_label[label] / total_per_label[label]
        else:
            accuracies[label] = 0.0
            
    return accuracies


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
 
            robust_l = pred.eq(labels.view_as(pred)) & adv_pred.eq(labels.view_as(adv_pred))
            succ_l = pred.eq(labels.view_as(pred)) & ~adv_pred.eq(labels.view_as(adv_pred))

            succ_pair[l].append(images[succ_l].cpu())
            robust_pair[l].append(images[robust_l].cpu())

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
    
    # print(len(c))
    
    for (i,j,curr) in c:
        i_layer = np.searchsorted(prefix_dims, i, side='right') - 1
        i1 = (int)(i)
        j1 = (int)(j)
        if curr < 0:
            # if i_layer == 3:  # Second layer (index 1)
            #     neg_e_second.add((i1,j1))
            # else:
            neg_e_other.add((i1,j1))
        elif curr >= 0:
            pos_e.add((i1,j1))
            
    # print(len(neg_e_other))
    # print(len(pos_e))
    # input()
        
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


def remove_edge_fc_perE(args):
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
    # remove_num = [1000,1500,2000,2500,3500,5000,6000,7000,8000,9000,10000,12000,16000]
    remove_num = []
    
    # build model
    for layer_num in layers:
        dims = model_zoo[layer_num]
        
        if model_pre_name.lower() == "ori" or model_pre_name.lower() == "decay":
            model_name = "best_ori_10l_" + str(layer_num) + ".pth"
        elif model_pre_name.lower() == "adv":
            model_name = "pgdtrain_" + str(layer_num) + ".pth"
        elif model_pre_name.lower() == 'big_adv':
            model_name = "fc_big_adv.pth"
            dims = model_zoo[21]
        elif model_pre_name.lower() == 'big_adv_01':
            model_name = "fc_big_adv01.pth"
            dims = model_zoo[21]
        elif model_pre_name.lower() == 'big_ori':
            model_name = "fc_big_ori.pth"
            dims = model_zoo[21]
        elif model_pre_name.lower() == 'big_wd':
            model_name = "fc_big_wd.pth"
            dims = model_zoo[21]
        elif model_pre_name.lower() == 'big_wd_05':
            model_name = "fc_big_wd_05.pth"
            dims = model_zoo[21]
        elif model_pre_name.lower() == 'big_wd_001':
            model_name = "fc_big_wd001.pth"
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
   
        test_cleanacc = test_clean_perlabel(net_full, sep_dataloader)
            
        # succ_pair, robust_pair = test(net_H, sep_dataloader, eps=e, alpha=2/255, iters=40, device=device)

        for l in selected_classes:
            with open(res_path + "edge_fc_tanh_" + str(l) + str(layer_num) + ".txt", "w+") as ff:
                ff.write(f'For model {model_name}: \n')

                ff.write(f'The clean accuracy for original model:\n')
                for label, acc in test_cleanacc.items():
                    ff.write(f'Label {label}: {acc:.3f}\n')
                ff.write('\n')

                neg_acc_adv = []
                pos_acc_adv = []
                neg_acc_clean = []
                pos_acc_clean = []

                for (images, labels) in sep_dataloader[l]:
                    for idx in range(images.shape[0]):
                        if (idx >= sample_size):
                            break
                        img = images[idx].to(device)
                        edge_array, nodes_ori, output, all_node = net_full.NN_info_batch(img.unsqueeze(0))
    
                        weights = output.detach().clone().to(device)     
                        weights_inv1, weights_inv2 = net_full.normalization_weight_w3(nodes_ori, weights, dims)
                        weights_inv = weights_inv1.detach()
                        weights_inv2 = weights_inv2.detach()

                        ricci_curvature, sp_dict = graph_curvature_main_torch(dims, weights_inv, device=device, probability_w=weights_inv2, alpha=alpha)

                        c, neg_e_second, neg_e_other, pos_e = get_top_c(ricci_curvature, 1, prefix_dims)
                        reversed_pos_e = list(pos_e)
                        reversed_pos_e.reverse()

                        ff.write(f'It has {len(neg_e_other)} negative curvature edges, {len(pos_e)} positive curvature egdes .. \n')
                        remove_num = [0, (int)(len(neg_e_other)*0.5), len(neg_e_other), (int)(len(pos_e)*0.1), (int)(len(pos_e)*0.2), (int)(len(pos_e)*0.3), (int)(len(pos_e)*0.4), (int)(len(pos_e)*0.5), (int)(len(pos_e)*0.7), (int)(len(pos_e)*0.9), (int)(len(pos_e))]
                        
                        # start remove
                        for index, rem_f in enumerate(remove_num):
                            ff.write(f'Remove edge number {rem_f}: \n')
                            cur_n = model_full_n + '_' + str(layer_num) + '_' + str(rem_f) + '_' + str(l)

                            # remove negative curvature edges
                            net_H.load_state_dict(torch.load(model_path + model_name))
                            edge_r = Edge_Remove(net_H, dims, min(rem_f, len(neg_e_other)), res_path)
                            edge_r.e_remove(neg_e_other, cur_n + "other_neg.pth")
                            
                            # test acc
                            net_neg = FC_MD(dims, layer_num)

                            net_neg.load_state_dict(torch.load(res_path + cur_n + "other_neg.pth"))
                            net_neg = net_neg.to(device)
                            os.remove(res_path + cur_n + "other_neg.pth")

                            acc_clean_neg_other = test_clean_perlabel(net_neg, sep_dataloader)

                            # remove positive curvature edges
                            net_H.load_state_dict(torch.load(model_path + model_name))
                            edge_r = Edge_Remove(net_H, dims, min(rem_f, len(reversed_pos_e)), res_path)
                            edge_r.e_remove(reversed_pos_e, cur_n + "pos.pth")
                            
                            # test acc
                            net_pos = FC_MD(dims, layer_num)

                            net_pos.load_state_dict(torch.load(res_path + cur_n + "pos.pth"))
                            net_pos = net_pos.to(device)
                            os.remove(res_path + cur_n + "pos.pth")

                            acc_clean_pos = test_clean_perlabel(net_pos, sep_dataloader)

                            for e in eps:
                                print(f'Current eps {e}: ')
                                ff.write(f'Current eps {e}: \n')

                                test_advacc = test_adversarial_perlabel_perE(net_full, sep_dataloader, eps=e, alpha=2/255, iters=40)
                                ff.write(f'The adversary accuracy eps = {e} for original model:\n')
                                for label, acc in test_advacc.items():
                                    ff.write(f'Label {label}: {acc:.3f}\n')
                                ff.write('\n')

                                acc_adv_neg_other = test_adversarial_perlabel_perE(net_neg, sep_dataloader, eps=e, alpha=2/255, iters=40)

                                neg_acc_adv.append(acc_adv_neg_other)
                                neg_acc_clean.append(acc_clean_neg_other)
                            
                                ff.write(f'Test Accuracy after remove {(int)(min(len(neg_e_other), rem_f))} neg_e_other edges:\n')
                                ff.write(f'The clean accuracy for remove negative edges:\n')
                                for label, acc in acc_clean_neg_other.items():
                                    ff.write(f'Label {label}: {acc:.3f}\n')
                                ff.write('\n')
                                ff.write(f'The adversary accuracy for remove negative edges:\n')
                                for label, acc in acc_adv_neg_other.items():
                                    ff.write(f'Label {label}: {acc:.3f}\n')
                                ff.write('\n')


                                acc_adv_pos = test_adversarial_perlabel_perE(net_pos, sep_dataloader, eps=e, alpha=2/255, iters=40)
                            
                                pos_acc_adv.append(acc_adv_pos)
                                pos_acc_clean.append(acc_clean_pos)

                                ff.write(f'Test Accuracy after remove {(int)(min(len(reversed_pos_e), rem_f))} reversed_pos_e1 edges:\n')
                                ff.write(f'The clean accuracy for remove positive edges:\n')
                                for label, acc in acc_clean_pos.items():
                                    ff.write(f'Label {label}: {acc:.3f}\n')
                                ff.write('\n')
                                ff.write(f'The adversary accuracy for remove positive edges:\n')
                                for label, acc in acc_adv_pos.items():
                                    ff.write(f'Label {label}: {acc:.3f}\n')
                                ff.write('\n')
                                
                                ff.write("\n\n")

                                excel_path = res_path + f'accuracies_layer{layer_num}_eps{e}_label{l}.xlsx'
                                
                                # Save accuracies to Excel after each fraction
                                row_data = {
                                    'ID': [idx],
                                    'Remove Number': [rem_f],
                                    'Label': [l]
                                }
                                
                                # Group all metrics for each label together
                                for label in selected_classes:                                    
                                    # Negative edge removal accuracies
                                    row_data[f'Negative Edge Clean Acc Label {label}'] = [neg_acc_clean[-1][label]]
                                    row_data[f'Negative Edge Adv Acc Label {label}'] = [neg_acc_adv[-1][label]]
                                    
                                    # Positive edge removal accuracies
                                    row_data[f'Positive Edge Clean Acc Label {label}'] = [pos_acc_clean[-1][label]]
                                    row_data[f'Positive Edge Adv Acc Label {label}'] = [pos_acc_adv[-1][label]]
                                
                                df = pd.DataFrame(row_data)

                                # If file exists, append to it, otherwise create new
                                if os.path.exists(excel_path):
                                    existing_df = pd.read_excel(excel_path)
                                    df = pd.concat([existing_df, df], ignore_index=True)
                                    
                                df.to_excel(excel_path, index=False)                    
