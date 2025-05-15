import torch
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

import sys
sys.path.append("..")

import tools.utils as utils
from tools.small_model import FC_MD
from tools.FC_linear import FC_Linear
from tools.LeNet5_custom_small import LeNet_custom_v2
from tools.graph_curvature import graph_curvature_main_torch
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



selected_classes = [0,1,2,3,4,5,6,7,8,9]


nodes_num = 2118

model_dims = {
    1: {"name": "input", "dim": {"channel": 1, "out_size": 28}},
    2: {"name": "cnn", "dim": {"channel": 6, "kernel": 6, "stride": 2, "out_size": 12}},
    3: {"name": "cnn", "dim": {"channel": 16, "kernel": 6, "stride": 2, "out_size": 4}},
    4: {"name": "fc", "dim": {"out_size": 120}},
    5: {"name": "fc", "dim": {"out_size": 84}},
    6: {"name": "fc", "dim": {"out_size": 10}}
}



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
    neg_e = set()  # Negative curvature edges
    pos_e = set()  # Positive curvature edges
    
    for batch in range(b):
        ricci_curv = np.array(curvature[batch])
        for (i, j, curr) in ricci_curv:
            if curr > 1:
                continue

            i_layer = np.searchsorted(prefix_dims, i, side='right') - 1
            if i_layer >= 2:
                c.append((i,j,curr))

    c.sort(key=lambda x: x[2])
    
    for (i,j,curr) in c:
        i1 = (int)(i)
        j1 = (int)(j)
        if curr < 0:
            neg_e.add((i1,j1))
        else:
            pos_e.add((i1,j1))
        
    return c, neg_e, pos_e


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




def remove_edge_cnn(args):
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
    
    eps = [0.03, 0.07, 0.1, 0.2]
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
    if model_pre_name == 'ori':
        model_name= "cnn_ori.pth"
    elif model_pre_name == 'adv':
        model_name= "cnn_adv.pth"

    net_H = LeNet_custom_v2(model_dims, None, device)
    net_H.load_state_dict(torch.load(model_path + model_name))
    net_H = net_H.to(device)

    net_full = copy.deepcopy(net_H)

    print(model_name)
    # remove_frac = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1]
    remove_num = [5,20,50,100,150,300,500,700,1000,1500,2000,2500,3000,3500]
    
            
    test_cleanacc = test_clean(net_full, test_loader)
    # succ_pair, robust_pair = test(net_H, sep_dataloader, eps=e, alpha=2/255, iters=40, device=device)

    for l in selected_classes:
        with open(res_path + "edge_cnn_" + str(l) + ".txt", "w+") as ff:
            ff.write(f'For model {model_name}: \n')
            ff.write(f'The clean accuracy for original model is {test_cleanacc}\n\n')
            print(f'Current label {l}: \n')
            ff.write(f'Current label {l}: \n')

            neg_acc_adv = []
            pos_acc_adv = []

            neg_acc_clean = []
            pos_acc_clean = []

            count = 0
            running_sum = None
            for (images, labels) in sep_dataloader[l]:
                if (count >= sample_size):
                    break
                for idx in range(images.shape[0]):
                    img = images[idx].to(device)
                    edge_array, nodes_ori, output = net_full.NN_info_batch(img.unsqueeze(0))

                    weights = output.detach().clone().to(device)                   
                    weights[edge_array == 0] = 0.
                    weights_inv = net_full.normalization_weight_w2(nodes_ori, weights, dims, model_dims)
                    weights_inv = weights_inv.detach()

                    if running_sum is None:
                        running_sum = weights_inv
                    else:
                        running_sum = torch.cat((running_sum, weights_inv), dim=0)

                    count += 1
                    if (count % 10 == 0):
                        print(f'Finish {count} examples....')
                        
                    if (count >= sample_size):
                        break

                    # Free memory
                    del weights, weights_inv
                    torch.cuda.empty_cache()
            
            if running_sum is None:
                continue

            w_avg = torch.mean(running_sum, dim=0).unsqueeze(0)
            del running_sum
            torch.cuda.empty_cache()
            ricci_curvature, sp_dict = graph_curvature_main_torch(dims, w_avg, device=device, model_dims=model_dims, alpha=alpha)

            # all_edges, single_hop_paths, label_paths, edge_curvatures = get_c(ricci_curvature, 1, prefix_dims, l)
                
            # ff.write(f'Total number of edges in the graph after normalization: {len(all_edges)}\n')
            # ff.write(f'Found {len(single_hop_paths)} single-hop paths and {len(label_paths)} negative paths reaching label neurons:\n')
            
            # ff.write("Single-hop paths:\n")
            # for path in single_hop_paths:
            #     # Each path has only one edge
            #     i, j, curr = path[0]
            #     ff.write(f"({i},{j}): {curr:.3f}\n")

            # ff.write("\nNegative paths reaching label neurons:\n")
            # for path, total_curv in label_paths:
            #     path_str = " -> ".join([f"({i},{j}): {curr:.3f}" for i, j, curr in path])
            #     ff.write(f"{path_str} | Total curvature: {total_curv:.3f}\n")

            # ff.write("\n\n\n")
            c, neg_e, pos_e = get_top_c(ricci_curvature, 1, prefix_dims, threshold = -50)
            reversed_pos_e = list(pos_e)[::-1]
            
            ff.write(f'It has {len(neg_e)} negative curvature edges, {len(pos_e)} positive curvature egdes .. \n')
                
            # start remove
            for index, rem_f in enumerate(remove_num):
                ff.write(f'Remove edge number {rem_f}: \n')

                # remove second layer negative curvature edges
                net_neg = copy.deepcopy(net_H)
                net_neg.__build_remove_mask__(neg_e, rem_f)
                # test acc
                acc_clean_neg = test_clean(net_neg, test_loader)

                # remove positive curvature edges
                net_pos = copy.deepcopy(net_H)
                net_pos.__build_remove_mask__(reversed_pos_e, rem_f)
                # test acc
                acc_clean_pos = test_clean(net_pos, test_loader)

                for e in eps:
                    print(f'Current eps {e}: ')
                    ff.write(f'Current eps {e}: \n')

                    test_advacc = test_adversarial(net_full, test_loader, eps=e, alpha=2/255, iters=40)
                    ff.write(f'The adversary accuracy eps = {e} for original model is {test_advacc}\n\n')

                    acc_adv_neg = test_adversarial(net_neg, test_loader, eps=e, alpha=2/255, iters=40)

                    neg_acc_adv.append(acc_adv_neg)

                    neg_acc_clean.append(acc_clean_neg)
                
                    ff.write(f'Test Accuracy after remove {(int)(min(len(neg_e), rem_f))} neg_e edges: clean acc {acc_clean_neg}, eps = {e}: adv acc {acc_adv_neg:.3f}...\n')
                    
                    
                    acc_adv_pos = test_adversarial(net_pos, test_loader, eps=e, alpha=2/255, iters=40)
                
                    pos_acc_adv.append(acc_adv_pos)
                    pos_acc_clean.append(acc_clean_pos)

                    ff.write(f'Test Accuracy after remove {(int)(min(len(reversed_pos_e), rem_f))} reversed_pos_e1 edges: clean acc {acc_clean_pos}, eps = {e}: adv acc {acc_adv_pos:.3f}...\n')
                    
                    ff.write("\n\n")

                    excel_path = res_path + f'accuracies_eps{e}.xlsx'
                    
                    # Create DataFrame for this fraction
                    df = pd.DataFrame({
                        'Label': [l],
                        'Remove Number': [rem_f],
                        'Negative Edge Clean Acc': [neg_acc_clean[-1]], 
                        'Negative Edge Adv Acc': [neg_acc_adv[-1]],
                        'Positive Edge Clean Acc': [pos_acc_clean[-1]],
                        'Positive Edge Adv Acc': [pos_acc_adv[-1]]
                    })
                    
                    # If file exists, append to it, otherwise create new
                    if os.path.exists(excel_path):
                        existing_df = pd.read_excel(excel_path)
                        df = pd.concat([existing_df, df], ignore_index=True)
                        
                    df.to_excel(excel_path, index=False)                    
