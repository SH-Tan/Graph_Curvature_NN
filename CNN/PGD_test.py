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

import sys
sys.path.append("..")

import tools.utils as utils
import tools.cnn_adj_matrix as build_cnn_adj
from tools.LeNet5_small import LeNet as LeNet
from tools.LeNet5_custom_small import LeNet_custom_v2 as LeNet_custom_v2
from RicciCurvature.OllivierRicci import OllivierRicci

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

file_path = "edge_v/"
model_path = "models/"
res_path = "res/"

# model_path = "advtrain/models/"
# res_path = "advtrain/"
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
    correct = 0.
    adv_acc = 0.
    total = 0.
    
    for l in selected_classes:
        for i, (images, labels) in enumerate(loader[l]):
            total += len(loader[l].dataset)
            images = images.to(device)
            labels = labels.to(device)
            output = n(images)
            pred = output.detach().max(1)[1]
            
            adv_img = standard_PGD(n, images, labels, eps, alpha, iters)
            adv_out = n(adv_img)
            adv_pred = adv_out.detach().max(1)[1]
            
            correct += (pred.eq(labels.data.view_as(pred)).sum().item())
            
            adv_acc += (adv_pred.eq(labels.data.view_as(adv_pred)).sum().item())

        print(f'Finish label {l}....')

    return correct/total, adv_acc/total





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
    
    model_name= "mnist_relu_small.pth"
    # model_name= "mnist_cnn.pth"
    # model_name= "pgdtrain_lenet.pth"
    # model = LeNet()
    # model.load_state_dict(torch.load(model_path + model_name))
    # model = model.to(device)
    
    train_loader, test_loader, valid_loader, valid_dataset, test_dataset = utils.get_new_data(selected_classes, data_train, data_test, valid_num = 2000, test_bs=1)

    # sep_dataloader = utils.sep_label(valid_dataset, selected_classes, bs=1)
    sep_dataloader = utils.sep_label(test_dataset, selected_classes, bs=2000)
    
    
    net_H = LeNet_custom_v2(model_dims, None, device)
    net_H.load_state_dict(torch.load(model_path + model_name))
    net_H = net_H.to(device)
    eps = [0.03, 0.05, 0.07, 0.1, 0.15, 0.2]
        
    with open(res_path + "adv_acc.txt", "a+") as f:
        # model_name= "mnist_relu_small.pth"
        
        for e in eps:
            acc, adv_acc = test(net_H, sep_dataloader, eps=e, alpha=2/255, iters=40)
            f.write(f'For eps = {e}, the original acc is {acc:.3f}, adv acc is {adv_acc:.3f}...\n')
        f.write('\n')