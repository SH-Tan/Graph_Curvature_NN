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
from tools.small_model import FC_MD
from tools.FC_linear import FC_Linear
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
res_path = "./"


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
    
    train_loader, test_loader, valid_loader, valid_dataset, test_dataset = utils.get_new_data(selected_classes, data_train, data_test, test_bs=1, valid_num=5000)

    sep_valloader = utils.sep_label(valid_dataset, selected_classes, bs=1)
    sep_dataloader = utils.sep_label(test_dataset, selected_classes, bs=2000)
    
    eps = [0.03, 0.05, 0.07, 0.1, 0.15, 0.2]
    # build model
    for layer_num in [2]:
        
        with open(res_path + "adv_acc.txt", "a+") as f:
            # model_name = "best_ori_10l_" + str(layer_num) + ".pth"
            # model_name = "best_ori_" + str(layer_num) + "_linear.pth"
            model_name = "best_adv_" + str(layer_num) + "_linear.pth"
            # model_name = "pgdtrain_" + str(layer_num) + ".pth"
            print(f'\nNow for model {model_name}, layer {layer_num}....\n')
            
            dims = model_zoo[layer_num]
            net_H = FC_Linear(dims, layer_num)

            net_H.load_state_dict(torch.load(model_path + model_name))
            net_H = net_H.to(device)
            
            for e in eps:
                acc, adv_acc = test(net_H, sep_dataloader, eps=e, alpha=2/255, iters=40)
                f.write(f'For eps = {e}, the original acc is {acc:.3f}, adv acc is {adv_acc:.3f}...\n')
            f.write('\n')