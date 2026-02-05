import torch
from torchvision.datasets.mnist import MNIST
from torchvision.datasets.cifar import CIFAR100
import torchvision.transforms as transforms
import torchvision
import numpy as np
import random
import os
import pandas as pd
import torch.nn as nn
from collections import defaultdict
from torch.utils.data import DataLoader

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
from tools.LeNet5_custom_small_w import LeNet_custom_v2
from tools.vgg9_custom_relu import VGG9_CIFAR10

os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")


# data_train = MNIST('./data/mnist',
#                   train=True,
#                   download=True,
#                   transform=transforms.Compose([
#                       # transforms.Resize((32, 32)),
#                       transforms.ToTensor()]))

# data_test = MNIST('./data/mnist',
#                   train=False,
#                   download=True,
#                   transform=transforms.Compose([
#                       # transforms.Resize((32, 32)),
#                       transforms.ToTensor()]))

# normalize = torchvision.transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), 
#                                             #   std=(0.2023, 0.1994, 0.2010))
train_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
test_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])


data_train = CIFAR100('./data/cifar10', train=True, download=True, transform=train_transforms)
data_test = CIFAR100('./data/cifar10', train=False, download=True, transform=test_transforms)



layers = [2, 4, 5, 6, 7]

model_zoo = {
    2: [784, 200, 150, 10],
    4: [784, 15, 25, 20, 15, 10],
    5: [784, 20, 30, 30, 20, 15, 10],
    6: [784, 20, 30, 30, 35, 20, 15, 10],
    7: [784, 30, 30, 40, 50, 30, 25, 20, 10]
}


selected_classes = [0,1,2,3,4,5,6,7,8,9]

model_path = "CNN/models/cnn/"
res_path = model_path


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
    correct = 0.
    adv_acc = 0.
    total = len(loader.dataset)

    for i, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        output = n(images)
        pred = output.detach().max(1)[1]
        
        adv_img = standard_PGD(n, images, labels, eps, alpha, iters)
        adv_out = n(adv_img)
        adv_pred = adv_out.detach().max(1)[1]
        
        correct += (pred.eq(labels.data.view_as(pred)).sum().item())
        
        adv_acc += (adv_pred.eq(labels.data.view_as(adv_pred)).sum().item())

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
    
    test_loader = DataLoader(
        data_test,
        batch_size=1024,
        shuffle=False,
        num_workers=2
    )
    
    # eps = [0.03, 0.07, 0.1, 0.2]
    eps = [1/255, 2/255, 3/255, 5/255, 8/255]
    # eps = [1/255]
    # build model
    for layer_num in [2]:
        dims = model_zoo[layer_num]
        
        with open("./adv_acc.txt", "a+") as f:
            model_name = "vgg9_10_adv_relu_s2.pth"
            # net_H = LeNet(input_c = 3)
            net_H = VGG9_CIFAR10(input_c = 3,num_classes=10)
            
            net_H.load_state_dict(torch.load(model_path + model_name))
            net_H = net_H.to(device)
            
            for e in eps:
                acc, adv_acc = test(net_H, test_loader, eps=e, alpha=2/255, iters=20)
                f.write(f'For eps = {e}, the original acc is {acc:.3f}, adv acc is {adv_acc:.3f}...\n')
            f.write('\n')