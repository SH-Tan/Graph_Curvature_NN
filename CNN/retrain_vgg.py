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
from tools.curvature_cal import curv_cal

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
        
    print('Adversary training set: Avg. Accuracy: {}/{} ({:.2f}%)'.format(
    correct, len(loader.dataset),
    (100. * correct / len(loader.dataset))))

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



def compute_curvature_edges(
    ckpt_path,
    model_full_n,
    metric,
    dataset,
    sample_size,
    prefix_dims,
    total,
    para_dims,
    res_path,
    args,
    data_path,
):
    save_name = f"{model_full_n}_{metric}_{dataset}_{sample_size}_combined_min2.pkl"
    save_path = os.path.join(res_path, save_name)

    # recompute curvature
    curv_cal(
        model_dims,
        model_dims_small,
        selected_classes,
        args,
        ckpt_path
    )

    # load curvature results
    neg_freq_dict, pos_freq_dict = process_batches_memory_efficient(
        data_path,
        model_full_n,
        metric,
        dataset,
        sample_size,
        prefix_dims,
        total_example=total,
        para_dims=para_dims
    )

    pos_edges_only = [
        (item[0], item[1]) if item[0] == "weight" else item[0:3]
        for item in pos_freq_dict
    ]

    return pos_edges_only



from collections import Counter
def count_edge_frequency(edge_sets):
    freq = Counter()
    curvature_sum = defaultdict(list)

    # for edge_set in edge_sets:
    for i, j, c in edge_sets:
        key = tuple(sorted((i, j)))  # normalize direction for undirected edges
        freq[key] += 1
        curvature_sum[key].append(c)

    results = []
    for key in freq:
        avg_curv = np.min(curvature_sum[key])
        results.append((key[0], key[1], freq[key], avg_curv))

    return results


def get_top_c(curvature, b, prefix_dims):
    neg_e = defaultdict(list)
    pos_e = defaultdict(list)
    cnn_e = defaultdict(list)

    # Convert once
    curv = np.asarray(curvature)
    edges = curv.copy()
    edges[:, :2] = edges[:, :2].astype(int)

    # ----- PRECOMPUTE LAYER OF EACH UNIQUE NODE -----
    i_nodes = edges[:, 0].astype(int)
    j_nodes = edges[:, 1].astype(int)
    unique_nodes = np.unique(np.concatenate([i_nodes, j_nodes]))

    # Compute layer for each unique node only once
    unique_layers = np.searchsorted(prefix_dims, unique_nodes, side="right") - 1

    # Convert to a dictionary or array lookup
    node_to_layer = dict(zip(unique_nodes, unique_layers))

    # ----- PROCESS EDGES -----
    for i, j, curr in edges:
        i = int(i)
        j = int(j)

        i_layer = node_to_layer[i]
        j_layer = node_to_layer[j]

        # Adjust curvature only for some layers
        if 0 < i_layer < 8 and abs(curr - 1.0) < 1e-6:
            curr = 2.0

        # CNN edges (non-FC)
        if i_layer not in (6, 7, 8):
            cnn_e[i_layer].append((i, j, curr))
            continue

        # FC edges only between adjacent layers
        elif j_layer == i_layer + 1:
            if curr < 0:
                neg_e[i_layer].append((i, j, curr))
            else:
                pos_e[i_layer].append((i, j, curr))

    return neg_e, pos_e, cnn_e



def process_batches_memory_efficient(
    data_path,
    model_full_n,
    metric,
    dataset,
    sample_size,
    prefix_dims,
    total_example,
    para_dims
):
    # prefix = f"{model_full_n}_{metric}_{dataset}_batch"
    prefix = f"{model_full_n}_{metric}_{dataset}_label"
    suffix = ".pkl"

    # extract label + id from filename
    def extract_label_id(f):
        match = re.search(r'label(\d+)_id(\d+)', f)
        if match:
            return int(match.group(1)), int(match.group(2))
        return -1, -1

    all_files = [
        f for f in os.listdir(data_path)
        if f.startswith(prefix) and f.endswith(suffix)
    ]
    # Sort by label then id
    all_files = sorted(all_files, key=lambda f: extract_label_id(f)[1])

    print(f"Found files: {all_files}")
    
    # Precompute edge->weight mapping per CNN layer
    cnn_edge_to_weight_map = {}
    for layer in range(len(model_dims)-2):  # skip input/output placeholder layers
        layer_info = model_dims[layer+2]
        if layer_info["name"] == "cnn":
            pre_dim = model_dims[layer+1]["dim"]
            pre_ch = pre_dim["channel"]
            in_size = pre_dim["out_size"]
            cur_dim = layer_info["dim"]
            cur_ch = cur_dim["channel"]
            kernel = cur_dim["kernel"]
            stride = cur_dim["stride"]
            padding = cur_dim["padding"]

            cnn_edge_to_weight_map[layer] = build_cnn_edge_weight_map(
                pre_ch, in_size, cur_ch, kernel, stride, padding, layer, prefix_dims
            )

    label_counts = {l: 0 for l in selected_classes}
    neg_weight_sets = []
    pos_weight_sets = []
    edge_sets = []

    for f in all_files:
        label, sample_id = extract_label_id(f)
        if label not in selected_classes:
            continue
        if label_counts[label] >= sample_size:
            continue
        
        file_path = os.path.join(data_path, f)
        print(f'file_path: {file_path}')
        
        with open(file_path, 'rb') as file:
            batch_data = pickle.load(file)

        new_data = batch_data
        # available = sample_size - label_counts[l]
        use_data = [new_data]

        for ricci in use_data:
            neg_e, pos_e, cnn_e = get_top_c(ricci, b=1, prefix_dims=prefix_dims)
            
            for layer, edges in cnn_e.items():
                layer_info = model_dims[layer + 2]
                if layer_info["name"] == "cnn":
                    pos_curv_weights, neg_curv_weights = aggregate_cnn_weight_curvature(edges, cnn_edge_to_weight_map[layer])
                    # all_weight_sets.append([(w, c, f, pos_f, neg_f) for w, (c, f, pos_f, neg_f) in weight_curv.items()])
                    neg_weight_sets.append([(w, c, f, para_dims[w[0]]) for w, (c, f, z) in neg_curv_weights.items()])
                    pos_weight_sets.append([(w, c, f, para_dims[w[0]]) for w, (c, f, z) in pos_curv_weights.items()])

         
            # === Aggregate CNN edges → per-weight curvatures ===
            for layer, edge in neg_e.items():
                layer_info = model_dims[layer + 2]
                if layer_info["name"] == "cnn":
                    print("ERROR!!")
                    continue
                    weight_curv, freq = aggregate_cnn_weight_curvature(edges, cnn_edge_to_weight_map[layer])
                    neg_weight_sets.append([(w, c, f) for w, (c, f) in weight_curv.items()])
                else:
                    # FC layer — keep per-edge
                    edge_sets.extend(edge)

            for layer, edge in pos_e.items():
                layer_info = model_dims[layer + 2]
                if layer_info["name"] == "cnn":
                    print("ERROR!!")
                    continue
                    weight_curv, freq = aggregate_cnn_weight_curvature(edges, cnn_edge_to_weight_map[layer])
                    pos_weight_sets.append([(w, c, f) for w, (c, f) in weight_curv.items()])
                else:
                    edge_sets.extend(edge)
            del ricci, neg_e, pos_e

        label_counts[label] += len(use_data)

        del batch_data
        gc.collect()

        if all(label_counts[l] >= sample_size for l in selected_classes):
            break

    print("Finished processing all required batches.")
    

    freq_fc = count_edge_frequency(edge_sets)
    
    all_weight_sets = pos_weight_sets + neg_weight_sets

    freq_cnn = count_weight_frequency(all_weight_sets, model_dims, para_dims)
    

    p_all = (
        [("edge", i, j, f, c) for (i, j, f, c) in freq_fc] +
        [("weight", w, None, f, c) for (w, f, c, p) in freq_cnn]
    )

    neg_freq_edges_sorted = sorted(p_all, key=lambda x: (x[4]))  # by frequency desc, curvature asc
    pos_freq_edges_sorted = sorted(p_all, key=lambda x: (-x[4])) # by frequency desc, curvature desc

    return neg_freq_edges_sorted, pos_freq_edges_sorted




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
        freeze_smallest=True
    )

    for p in net.parameters():
        if hasattr(p, "_backward_hooks") and p._backward_hooks is not None:
            p._backward_hooks.clear()


    net.register_freeze_grad(freeze_on_mask_one=False)

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



def retrain_vgg(args):
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

    train_loader, test_loader, valid_loader, valid_dataset, test_dataset = utils.get_new_data(selected_classes, data_train, data_test, test_bs=200, valid_num=5000)

    # transform_train1 = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        # transforms.ToTensor(),
        # transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010)),
    # ])
    
    # train_loader = load_dataset_from_disk("./data/CIFAR10_train", batch_size=256, transform=transform_train1)
    # valid_loader = load_dataset_from_disk("./data/CIFAR10_val", batch_size=2000, shuffle=True)

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
        # from tools.vgg16_custom_relu_new_small_bn import VGG16_CIFAR10_small_BN
        from tools.vgg9_custom_relu import VGG9_CIFAR10
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
    
    print(f'Finish Test..')
    
    save_name = f"{model_full_n}_{metric}_{dataset}_{sample_size}_combined_min2.pkl"
    save_path = os.path.join(res_path, save_name)

    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            data_loaded = pickle.load(f)
            # Access the contents
            neg_freq_dict = data_loaded.get('neg', [])
            pos_freq_dict = data_loaded.get('pos', [])
        print("File loaded successfully!")

    print(f'It has {len(neg_freq_dict)} negative curvature edges, {len(pos_freq_dict)} positive curvature egdes.. \n')
    
    pos_edges_only = [
        (item[0], item[1]) if item[0] == "weight" else item[0:3] for item in pos_freq_dict
    ]
    
    pos_total = len(pos_edges_only)
    
    
    test_cleanacc = test_clean(net_full, test_loader)
    
    split_points = [0.55]
    num_rounds = 3
    epochs_per_round = 30
    
    with open(res_path + "retrain_adv_acc_iter_useful.txt", "a+") as f:
        f.write(f'The clean acc for the full model is {test_cleanacc}...\n')
        for ep in eps:
            test_advacc = test_adversarial(net_full,test_loader,eps=ep/255, alpha=2/255, iters=20)
            f.write(f'The adv acc for the full model is {test_advacc} - eps = {ep}...\n')
        f.write('\n')
                
        for spilt_p in split_points:
            f.write(f"\n====== Freeze ratio = {spilt_p} ======\n")
            
            prev_best_state = None  # ← THIS is the key

            for r in range(num_rounds):
                f.write(f"\n--- Mask / Retrain Round {r+1} ---\n")
                
                # define split points
                split1 = int(spilt_p * pos_total)
            
                f.write(f'Now freeze {split1} parameters...\n\n')
                
                # ----------------------------------
                # load model for THIS round
                # ----------------------------------
                net_iter = copy.deepcopy(net_H)

                if prev_best_state is not None:
                    net_iter.load_state_dict(prev_best_state)
                    
                net_iter.__build_remove_mask__(pos_edges_only, split1)
                net_iter.register_freeze_grad(freeze_on_mask_one=True)

                trainable_params = [
                    p for p in net_iter.parameters() if p.requires_grad
                ]
                
                optimizer = torch.optim.Adam(trainable_params, lr=0.002, weight_decay=0.00)
                optimizer.state_dict()["state"]


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
                
                best_acc = 0.
                
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
                ckpt_path = res_path + f"vgg9_iter{r+1}_ratio{spilt_p}.pth"
                torch.save(best_state, ckpt_path)

                prev_best_state = best_state  # ← THIS enables chaining
                
                curv_cal(model_dims, model_dims_small, selected_classes, args, ckpt_path)

            f.write("\n\n")
