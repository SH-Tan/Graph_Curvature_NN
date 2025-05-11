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
from tools.small_model import FC_MD
from tools.FC_linear import FC_Linear
from tools.graph_curvature import graph_curvature_main_torch
from tools.draw_net import DrawNN
from tools.edge_remove import Edge_Remove


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
    neg_e = set()
    pos_e = set()
    
    for batch in range(b):
        ricci_curv = np.array(curvature[batch])
        for (i, j, curr) in ricci_curv:
            c.append((i,j,curr))

    c.sort(key=lambda x: x[2])
    
    for (i,j,curr) in c:
        i_layer = np.searchsorted(prefix_dims, i, side='right') - 1
        i1 = (int)(i)
        j1 = (int)(j)
        if (curr < 0):
            neg_e.add((i1,j1))
        elif (curr >= 0):
            pos_e.add((i1,j1))
        
    return c, neg_e, pos_e




def remove_start_fc(args):
    seed = 59
    
    # set random seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
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

    remove_frac = [0.08, 0.2, 0.5, 0.8, 1]
    
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
            model_name = "fc_big_ori.pth"
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


        with open(res_path + "edge_remove_acc_neg_pos_" + str(layer_num) + ".txt", "w+") as ff:
            ff.write(f'For model {model_name}: \n')
                
            test_cleanacc = test_clean(net_full, test_loader)
            
            ff.write(f'The clean accuracy for original model is {test_cleanacc}\n')
            
            for e in eps:
                test_advacc = test_adversarial(net_full, test_loader, eps=e, alpha=2/255, iters=40)
                ff.write(f'The adversary accuracy eps = {e} for original model is {test_advacc}\n\n')
                
                print(f'Current eps {e}: ')
                ff.write(f'Current eps {e}: \n')

                succ_pair, robust_pair = test(net_H, sep_dataloader, eps=e, alpha=2/255, iters=40, device=device)

                for l in selected_classes:
                    print(f'Current label {l}: \n')
                    ff.write(f'Current label {l}: \n')

                    neg_acc_adv = []
                    pos_acc_adv = []

                    neg_acc_clean = []
                    pos_acc_clean = []

                    count = 0
                    data = None
                    for ori_im in robust_pair[l]:
                        if (count >= sample_size):
                            break
                        for im in ori_im:
                            img = im.to(device)
                            edge_array, nodes_ori, output, all_node = net_H.NN_info_batch(img.unsqueeze(0))
        
                            weights = output.detach().clone().to(device)                   
                            weights[edge_array == 0] = 0.
                            weights_inv = net_H.normalization_weight_w2(nodes_ori, weights, dims)
                            weights_inv = weights_inv.detach()

                            if (count == 0):
                                data = weights_inv
                            else:
                                data = torch.cat((data, weights_inv), dim=0)

                            count += 1
                            if (count % 10 == 0):
                                print(f'Finish {count} examples....')
                                
                            if (count >= sample_size):
                                break
                    
                    if data == None:
                        continue

                    w_avg = torch.mean(data, dim=0).unsqueeze(0)
                    ricci_curvature, sp_dict = graph_curvature_main_torch(dims, w_avg, device=device, alpha=alpha)

                    c, neg_e, pos_e = get_top_c(ricci_curvature, 1, prefix_dims, threshold = -30)
                    reversed_pos_e1 = list(pos_e)[::-1]
                    
                    ff.write(f'It has {len(neg_e)} negative curvature edges, {len(pos_e)} positive curvature egdes .. \n')

                    # start remove
                    for index, rem_f in enumerate(remove_frac):
                        ff.write(f'Remove edge fraction {rem_f}: \n')
                        cur_n = model_full_n + '_' + str(layer_num) + '_' + str(rem_f) + '_' + str(e)  + '_' + str(l)
                        
                        # remove negative curvature edges
                        net_H.load_state_dict(torch.load(model_path + model_name))
                        edge_r = Edge_Remove(net_H, dims, rem_f, res_path)
                        edge_r.e_remove(neg_e, cur_n + "neg.pth")
                        
                        # test acc
                        net_new = FC_MD(dims, layer_num)

                        net_new.load_state_dict(torch.load(res_path + cur_n + "neg.pth"))
                        net_new = net_new.to(device)
                        
                        acc_clean = test_clean(net_new, test_loader)
                        acc_adv = test_adversarial(net_new, test_loader, eps=e, alpha=2/255, iters=40)

                        neg_acc_adv.append(acc_adv)
                        neg_acc_clean.append(acc_clean)
                        
                        ff.write(f'Test Accuracy af ter remove {(int)(len(neg_e)*rem_f)} neg_e edges: clean acc {acc_clean}, adv acc {acc_adv:.3f}...\n')
                        os.remove(res_path + cur_n + "neg.pth")

                        # remove positive curvature edges
                        net_H.load_state_dict(torch.load(model_path + model_name))
                        edge_r = Edge_Remove(net_H, dims, rem_f, res_path)
                        edge_r.e_remove(reversed_pos_e1, cur_n + "pos.pth")
                        
                        # test acc
                        net_new = FC_MD(dims, layer_num)

                        net_new.load_state_dict(torch.load(res_path + cur_n + "pos.pth"))
                        net_new = net_new.to(device)
                        
                        acc_clean = test_clean(net_new, test_loader)
                        acc_adv = test_adversarial(net_new, test_loader, eps=e, alpha=2/255, iters=40)
                        
                        pos_acc_adv.append(acc_adv)
                        pos_acc_clean.append(acc_clean)

                        ff.write(f'Test Accuracy after remove {(int)(len(reversed_pos_e1)*rem_f)} reversed_pos_e1 edges: clean acc {acc_clean}, adv acc {acc_adv:.3f}...\n')
                        os.remove(res_path + cur_n + "pos.pth")
                        ff.write("\n\n")

                        # Save accuracies to Excel after each fraction
                        df = pd.DataFrame({
                            'Epsilon': [e],
                            'Remove Fraction': [rem_f],
                            'Negative Edge Clean Acc': [neg_acc_clean[-1]], 
                            'Negative Edge Adv Acc': [neg_acc_adv[-1]],
                            'Positive Edge Clean Acc': [pos_acc_clean[-1]],
                            'Positive Edge Adv Acc': [pos_acc_adv[-1]]
                        })
                        
                        excel_path = res_path + f'accuracies_layer{layer_num}_label{l}.xlsx'
                        
                        # If file exists, append to it, otherwise create new
                        if os.path.exists(excel_path):
                            existing_df = pd.read_excel(excel_path)
                            df = pd.concat([existing_df, df], ignore_index=True)
                            
                        df.to_excel(excel_path, index=False)
                    