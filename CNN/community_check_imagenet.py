import torch
from torchvision.datasets.cifar import CIFAR10
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
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
from tools.graph_curvature_new import graph_curvature_main_torch
from tools.VGG16 import VGG16CustomTail


np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=torch.inf)

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")



model_dims = {
    1:  {"name": "input", "dim": {"channel": 3, "out_size": 224}},  # Input: 3x224x224
    # Block 1
    2:  {"name": "cnn", "dim": {"channel": 64,  "kernel": 3, "stride": 1, "padding": 1, "pool": True,  "out_size": 112}},  # conv1_2 + pool
    # Block 2
    3:  {"name": "cnn", "dim": {"channel": 128, "kernel": 3, "stride": 1, "padding": 1, "pool": True,  "out_size": 56}},  # conv2_2 + pool
    # Block 3
    4:  {"name": "cnn", "dim": {"channel": 256, "kernel": 3, "stride": 1, "padding": 1, "pool": True,  "out_size": 28}},  # conv3_3 + pool
    # Block 4
    5: {"name": "cnn", "dim": {"channel": 512, "kernel": 3, "stride": 1, "padding": 1, "pool": True,  "out_size": 14}},  # conv4_3 + pool
    # Block 5
    6: {"name": "cnn", "dim": {"channel": 512, "kernel": 3, "stride": 1, "padding": 1, "pool": True,  "out_size": 7}},   # conv5_3 + pool

    # Fully connected layers
    7: {"name": "fc",  "dim": {"in": 512 * 7 * 7, "out_size": 4096}},  # Flatten → fc1
    8: {"name": "fc",  "dim": {"in": 4096, "out_size": 4096}},         # fc2
    9: {"name": "fc",  "dim": {"in": 4096, "out_size": 1000}},         # fc3 (ImageNet classes)
}



model_dims_small = {
    1: {"name": "cnn", "dim": {"channel": 512, "kernel": 3, "stride": 1, "padding": 1, "pool": True,  "out_size": 7}},   # conv5_3 + pool
    2: {"name": "fc",  "dim": {"in": 512 * 7 * 7, "out_size": 4096}},  # Flatten → fc1
    3: {"name": "fc",  "dim": {"in": 4096, "out_size": 4096}},         # fc2
    4: {"name": "fc",  "dim": {"in": 4096, "out_size": 1000}}        # fc3 (ImageNet classes)
}

selected_classes = [0,1,2,3,4,5,6,7,8,9]


def load_dataset_from_disk(path, batch_size=128, shuffle=True):
    data_path = f"{path}/data.pt"
    images, labels = torch.load(data_path)
    dataset = TensorDataset(images, labels)
    return dataset



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



def test(n, loader, device):    
    n.eval()
    robust_pair =[]
    succ_pair = []
 
    for i, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        output = n(images)
        pred = output.detach().max(1)[1]
        
        robust_l = pred.eq(labels.view_as(pred))
        succ_l = ~pred.eq(labels.view_as(pred))

        succ_pair.append((images[succ_l].cpu(), pred[succ_l].cpu()))
        robust_pair.append((images[robust_l].cpu(), pred[robust_l].cpu()))

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




def community_check_imagenet(args):
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
    
    val_dir = "/home/tans5/Graph_Curvature_NN/data/ImageNET" 
    
    # print("Current working directory:", os.getcwd())
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    
    # Load dataset
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)

    # Wrap in DataLoader
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=4)

    # train_loader, test_loader, valid_loader, valid_dataset, test_dataset = utils.get_new_data(selected_classes, data_train, data_test, test_bs=2000, valid_num=5000)

    # val_set = load_dataset_from_disk("./data/CIFAR10_val", batch_size=64, shuffle=False)
    # sep_dataloader = utils.sep_label(val_set, selected_classes, bs=32)
    
    dims_full = cal_dims(model_dims)
    dims = cal_dims(model_dims_small)
    edge_dims = cal_edges(model_dims_small)

    print(dims_full)
    print(dims)
    print(edge_dims)

    model_type = args.model_type
    model_pre_name = args.model_name
    res_path = args.mnist_res_path
    model_path = args.model_path
    metric = args.metric
    dataset = args.dataset
    alpha = args.alpha
    sample_size = args.sample_num
    activation = args.activation
    
    model_full_n = model_type.lower() + model_pre_name.lower()

    if not os.path.exists(res_path):
        os.makedirs(res_path)
    
    net_H = VGG16CustomTail(model_dims, None, device, pretrained_path = "CNN/models/new/vgg16.pth")
    # net_H.load_state_dict(torch.load(model_path + model_name))
    net_H = net_H.to(device)

    net_full = copy.deepcopy(net_H)

    succ_pair, robust_pair = test(net_H, val_loader, device=device)
    res_l = defaultdict(list)

    print("Finished loading model and test data..")

    # Prepare iterators for each label's data
    data_iterator = iter(robust_pair)
    round_id = 1
    total_collected = 0
    res_l = []  # hold curvature results for each processed image
    
    try:
        while total_collected < sample_size:
            images, label = next(data_iterator)
            
            for idx in range(images.shape[0]):
                with torch.no_grad():
                    net_full.eval()
                    img = images[idx].to(device, non_blocking=True).unsqueeze(0)

                    edge_array, nodes_ori, output = net_full.NN_info_batch(img)

                    weights = output.detach().to(device)
                    del output

                    if metric.lower() == "w1":
                        weights_inv1, weights_inv2 = net_full.normalization_weight_w1(nodes_ori, weights, dims, model_dims_small)
                        weights_inv = weights_inv1.detach()
                        weights_inv2 = weights_inv2.detach()
                        ricci_curvature = graph_curvature_main_torch(
                            dims, weights_inv, device=device,
                            model_dims=model_dims_small,
                            probability_w=weights_inv2, alpha=alpha
                        )
                    elif metric.lower() == "w3":
                        weights_inv1, weights_inv2 = net_full.normalization_weight_w3(nodes_ori, weights, dims, model_dims_small)
                        weights_inv = weights_inv1.detach()
                        weights_inv2 = weights_inv2.detach()
                        ricci_curvature = graph_curvature_main_torch(
                            dims, weights_inv, device=device,
                            model_dims=model_dims_small,
                            probability_w=weights_inv2, alpha=alpha,
                            pre_n=(np.sum(dims_full) - np.sum(dims)),
                            layers_to_process=[1,2,3]
                        )
                    elif metric.lower() == "w4":
                        weights_inv1, weights_inv2 = net_full.normalization_weight_w4(nodes_ori, weights, dims, model_dims_small)
                        weights_inv = weights_inv1.detach()
                        weights_inv2 = weights_inv2.detach()
                        ricci_curvature = graph_curvature_main_torch(
                            dims, weights_inv, device=device,
                            model_dims=model_dims_small,
                            probability_w=weights_inv2, alpha=alpha,
                            pre_n=(np.sum(dims_full) - np.sum(dims)),
                        )
                    else:
                        raise Exception("Invalid graph metric, should be {w1, w3, w4}!")

                    res_l.append(ricci_curvature)

                    del img, edge_array, nodes_ori
                    del weights, weights_inv1, weights_inv2, weights_inv
                    torch.cuda.empty_cache()

                    # current_batch += 1
                    total_collected += 1
                    
                    print(f'One example...')

                    if total_collected % 10 == 0:
                        save_name = f"{model_full_n}_{metric}_{dataset}_batch{round_id}.pkl"
                        save_path = os.path.join(res_path, save_name)

                        with open(save_path, 'wb') as f:
                            pickle.dump(res_l, f)

                        print(f"[Saved] Batch {round_id}: {total_collected} total examples saved to {save_name}")

                        res_l.clear()
                        round_id += 1

    except StopIteration:
        print("Finished all samples.")

    # Save remaining data if any
    if res_l:
        save_name = f"{model_full_n}_{metric}_{dataset}_batch{round_id}_final.pkl"
        save_path = os.path.join(res_path, save_name)
        with open(save_path, 'wb') as f:
            pickle.dump(res_l, f)
        print(f"[Saved] Final batch with {len(res_l)} remaining samples.")