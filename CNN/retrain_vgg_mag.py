import torch
from torchvision.datasets.cifar import CIFAR10
import torchvision.transforms as transforms
import torchvision
import numpy as np
import random
import os
import torch.nn as nn
from collections import defaultdict
import copy
from torch.utils.data import TensorDataset,DataLoader,Dataset

import pickle
import time
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import gc
import re

from .e2w_utils_new import *

import sys
sys.path.append("..")

import tools.utils as utils

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
    1: {"name": "input", "dim": {"channel": 3, "out_size": 32}},   # Input image

    2: {"name": "cnn", "dim": {"channel": 64, "kernel": 2, "stride": 2, "padding":0, "out_size": 16}}, 
    3: {"name": "cnn", "dim": {"channel": 128, "kernel": 3, "stride": 1, "padding":0, "out_size": 14}},   # After conv1_2 
    
    4: {"name": "cnn", "dim": {"channel": 128, "kernel": 3, "stride": 1, "padding":0, "out_size": 12}},  
    5: {"name": "cnn", "dim": {"channel": 128, "kernel": 3, "stride": 2, "padding":0, "out_size": 5}},   # After conv2_2 
    
    6: {"name": "cnn", "dim": {"channel": 256, "kernel": 3, "stride": 1, "padding":0, "out_size": 3}},
    7: {"name": "cnn", "dim": {"channel": 256, "kernel": 3, "stride": 1, "padding":0, "out_size": 1}},

    8: {"name": "fc", "dim": {"out_size": 512}},  # Flatten(512×2×2) → 1024
    9: {"name": "fc", "dim": {"out_size": 128}},
    10: {"name": "fc", "dim": {"out_size": 10}}
}


model_dims_small = model_dims
selected_classes = [0,1,2,3,4,5,6,7,8,9]

class TransformedTensorDataset(Dataset):
    def __init__(self, tensors, transform=None):
        self.images, self.labels = tensors
        self.transform = transform

    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.labels)

def load_dataset_from_disk(path, batch_size=128, shuffle=True, transform=None):
    data_path = f"{path}/data.pt"
    images, labels = torch.load(data_path)
    dataset = TransformedTensorDataset((images, labels), transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


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


def train_adversarial(net, loader, optimizer, eps=.1, alpha=.1, iters=100, device = 'cuda'):
    # prepare model for training (only important for dropout, batch norm, etc.)
    net.train()
    
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        #print(data.size())

        data = standard_PGD(net, data, target, device, eps=eps, alpha=alpha, iters=iters).to(device)
        target = target.to(device)
        optimizer.zero_grad()
        
        output = net(data)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum()
        
        loss = loss_fn(output, target)
        total_loss += loss

        # compute gradients and make updates
        loss.backward()
        optimizer.step()
        
    # print('Adversary training set: Avg. Accuracy: {}/{} ({:.2f}%)'.format(
    # correct, len(loader.dataset),
    # (100. * correct / len(loader.dataset))))

    return total_loss/len(loader), correct / len(loader.dataset)



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


def cal_edges(model_dims):
    edges = []
    layer_num = len(model_dims)
    
    for i in range(2, layer_num + 1):
        cur_name = model_dims[i]["name"]
        cur_dim = model_dims[i]["dim"]
        cur_size = cur_dim['out_size']

        pre_name = model_dims[i-1]["name"]
        pre_dim = model_dims[i-1]["dim"]
        pre_size = pre_dim['out_size']

        if cur_name == "cnn":
            k = cur_dim['kernel']
            pool = cur_dim.get('pool', False)
            if pool:
                cur_size *= 2
            pre_channel = 1 if (pre_name == "fc") else pre_dim['channel']
            cur_edges = pre_channel * k**2 * cur_size**2 * cur_dim['channel']
        else:
            pre_nodes = pre_size if (pre_name == "fc") else pre_dim['channel']*(pre_size**2)
            cur_edges = cur_size * pre_nodes
            
        edges.append(cur_edges)
    
    return edges


def cal_parameters(model_dims):
    edges = []
    layer_num = len(model_dims)
    
    for i in range(2, layer_num + 1):
        cur_name = model_dims[i]["name"]
        cur_dim = model_dims[i]["dim"]
        cur_size = cur_dim['out_size']

        pre_name = model_dims[i-1]["name"]
        pre_dim = model_dims[i-1]["dim"]
        pre_size = pre_dim['out_size']

        if cur_name == "cnn":
            k = cur_dim['kernel']
            pool = cur_dim.get('pool', False)
            if pool:
                cur_size *= 2
            pre_channel = 1 if (pre_name == "fc") else pre_dim['channel']
            cur_edges = pre_channel * k**2 * cur_dim['channel']
        else:
            pre_nodes = pre_size if (pre_name == "fc") else pre_dim['channel']*(pre_size**2)
            cur_edges = cur_size * pre_nodes
            
        edges.append(cur_edges)
    
    return edges


def retrain_one_round(
    net,
    train_loader,
    valid_loader,
    test_loader,
    ratio,
    epochs,
    lr=0.002
):
    # rebuild mask from CURRENT weights
    net.build_global_magnitude_remove_mask(
        ratio=ratio,
        freeze_smallest=False
    )

    for p in net.parameters():
        if hasattr(p, "_backward_hooks") and p._backward_hooks is not None:
            p._backward_hooks.clear()


    net.register_freeze_grad()

    optimizer = torch.optim.Adam(
        [p for p in net.parameters() if p.requires_grad],
        lr=lr
    )

    best_val = -1.0
    best_state = None

    for e in range(epochs):
        train_adversarial(
            net, train_loader, optimizer,
            eps=2/255, alpha=2/255, iters=20
        )

        val_acc = test_clean(net, valid_loader)

        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(net.state_dict())

    # restore best validation model
    net.load_state_dict(best_state)

    test_acc = test_clean(net, test_loader)
    return best_val, test_acc, best_state



def retrain_vgg_mag(args):
    seed = 29
    
    # set random seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    _, test_loader, _, valid_dataset, test_dataset = utils.get_new_data(selected_classes, data_train, data_test, test_bs=200, valid_num=5000)

    transform_train1 = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        # transforms.ToTensor(),
        # transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010)),
    ])
    
    train_loader = load_dataset_from_disk("./data/CIFAR10_train", batch_size=256, transform=transform_train1)
    valid_loader = load_dataset_from_disk("./data/CIFAR10_val", batch_size=2000, shuffle=True)

    eps = [1,2,3,5,8]
    dims = cal_dims(model_dims)
    
    model_type = args.model_type
    model_pre_name = args.model_name
    res_path = args.mnist_res_path
    model_path = args.model_path
    metric = args.metric
    dataset = args.dataset
    alpha = args.alpha
    sample_size = args.sample_num
    activation = args.activation
    data_path = args.mnist_data_path
    
    if activation.lower() == "relu":
        from tools.vgg9_custom_relu_mag import VGG9_CIFAR10
    elif activation.lower() == "tanh":
        from tools.vgg9_custom_tanh import VGG9_CIFAR10
    
    model_full_n = model_type.lower() + model_pre_name.lower()

    dims = cal_dims(model_dims)
    prefix_dims = np.cumsum([0] + dims).tolist()
    
    if not os.path.exists(res_path):
        os.makedirs(res_path)
        
    # build model
    if model_pre_name == 'ori':
        model_name = "vgg9_10_ori_"
    elif model_pre_name == 'adv':
        model_name = "vgg9_10_adv_"
    elif model_pre_name == 'wd':
        model_name = "vgg9_10_wd_"
        
    model_name = model_name + activation + "_s2.pth"
    
    print(model_name)
    
    net_H = VGG9_CIFAR10(model_dims, None, device, prefix_dims)
    net_H.load_state_dict(torch.load(model_path + model_name))
    net_H = net_H.to(device)

    net_full = copy.deepcopy(net_H)
    
    edge_dims_small = cal_edges(model_dims_small) 
    para_dims = cal_parameters(model_dims_small)

    total_para = sum(para_dims) # - sum(para_dims[0:2]) # - sum(para_dims[:11]) # - sum(para_dims[-3:])
    
    print(para_dims)
    print(total_para)
    
    total = sample_size * len(selected_classes)

    test_cleanacc = test_clean(net_full, test_loader)

    split_points = [0, 0.3, 0.5, 0.7]
    num_rounds = 6
    epochs_per_round = 25

    with open(res_path + "retrain_ori_acc_useless_mag.txt", "w+") as f:
        f.write(f'The clean acc for the full model is {test_cleanacc}...\n')
        for ep in eps:
            adv_acc = test_adversarial(
                net_full, test_loader,
                eps=ep/255, alpha=2/255, iters=20
            )
            f.write(
                f"Clean: adv acc @ eps={ep}: {adv_acc}\n"
            )
                    
        for spilt_p in split_points:
            f.write(f"\n====== Freeze ratio = {spilt_p} ======\n")

            prev_best_state = None  # ← THIS is the key

            for r in range(num_rounds):
                f.write(f"\n--- Mask / Retrain Round {r+1} ---\n")
                
                print(f'spilt_p = {spilt_p}, round = {r}')

                # ----------------------------------
                # load model for THIS round
                # ----------------------------------
                net_iter = copy.deepcopy(net_H)

                if prev_best_state is not None:
                    net_iter.load_state_dict(prev_best_state)

                # ----------------------------------
                # retrain (best-val tracked)
                # ----------------------------------
                val_acc, test_acc, best_state = retrain_one_round(
                    net=net_iter,
                    train_loader=train_loader,
                    valid_loader=valid_loader,
                    test_loader=test_loader,
                    ratio=spilt_p,
                    epochs=epochs_per_round,
                )

                f.write(
                    f"Round {r+1}: best val acc = {val_acc}, test acc = {test_acc}\n"
                )

                # ----------------------------------
                # adversarial evaluation
                # ----------------------------------
                for ep in eps:
                    adv_acc = test_adversarial(
                        net_iter, test_loader,
                        eps=ep/255, alpha=2/255, iters=20
                    )
                    f.write(
                        f"Round {r+1}: adv acc @ eps={ep}: {adv_acc}\n"
                    )

                # ----------------------------------
                # save & carry forward BEST model
                # ----------------------------------
                # ckpt_path = res_path + f"vgg9_iter{r+1}_ratio{spilt_p}.pth"
                # torch.save(best_state, ckpt_path)

                prev_best_state = best_state  # ← THIS enables chaining

            f.write("\n\n")

