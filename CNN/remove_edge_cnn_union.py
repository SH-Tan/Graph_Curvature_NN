import torch
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
import numpy as np
import random
import os
import pandas as pd
import torch.nn as nn
from collections import defaultdict
import matplotlib.pyplot as plt
import copy

import pickle
import time
import pandas as pd

import sys
sys.path.append("..")

import tools.utils as utils


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



# def get_top_c(curvature, b, prefix_dims, threshold = -50):
#     c = []
#     neg_e = set()  # Negative curvature edges
#     pos_e = set()  # Positive curvature edges
    
#     for batch in range(b):
#         ricci_curv = np.array(curvature[batch])
#         for (i, j, curr) in ricci_curv:
#             if curr > 1:
#                 continue

#             c.append((i,j,curr))

#     c.sort(key=lambda x: x[2])
    
#     for (i,j,curr) in c:
#         i1 = (int)(i)
#         j1 = (int)(j)
#         i_layer = np.searchsorted(prefix_dims, i, side='right') - 1
#         if curr < 0 and i_layer >= 2:
#             neg_e.add((i1,j1,curr))
#         elif curr >= 0 and i_layer >= 2:
#             pos_e.add((i1,j1,curr))
        
#     return neg_e, pos_e



def get_top_c(curvature, b, prefix_dims):
    c = []
    neg_e = set()
    pos_e = set()
    seen_edges = set()
    # print(prefix_dims)
    noseen = 0

    for batch in range(b):
        ricci_curv = np.array(curvature[batch])
        for (i, j, curr) in ricci_curv:
            if curr > 1:
                curr = 1.0
            i1, j1 = int(i), int(j)
            c.append((i1, j1, curr))
            seen_edges.add((i1, j1))

    c.sort(key=lambda x: x[2])
    for (i, j, curr) in c:
        i_layer = np.searchsorted(prefix_dims, i, side='right') - 1
        if i_layer >= 2: 
            if curr < 0:
                neg_e.add((i, j, curr))
            else:
                pos_e.add((i, j, curr))

    # Get indices of FC layers only
    fc_layers = [i for i in sorted(model_dims.keys()) if model_dims[i]["name"] == "fc"]
    fc_indices = [list(model_dims.keys()).index(i) for i in fc_layers]  # 0-based index

    # Generate all possible FC edges
    all_fc_edges = set()
    for l in range(fc_indices[0]-1, fc_indices[-1]):
        start_i, end_i = prefix_dims[l], prefix_dims[l + 1]
        start_j, end_j = prefix_dims[l + 1], prefix_dims[l + 2]
        for i in range(start_i, end_i):
            for j in range(start_j, end_j):
                all_fc_edges.add((i, j))
                if (i, j) not in seen_edges:
                    pos_e.add((i, j, 1.0))  # default curvature
                    noseen += 1

    return neg_e, pos_e, noseen




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


# def plot_curve(neg_clean_acc, pos_clean_acc, remove_num, label, res_path):
#     # Zip, sort, and unzip to reorder all lists by remove_numbers
#     combined = sorted(zip(remove_num, neg_clean_acc, pos_clean_acc), key=lambda x: x[0])
#     remove_sorted, neg_sorted, pos_sorted = zip(*combined)

#     # Plot
#     plt.figure(figsize=(8, 5))
#     plt.plot(remove_sorted, neg_sorted, label='Negative Edge Clean Acc', marker='o', linestyle='--')
#     plt.plot(remove_sorted, pos_sorted, label='Positive Edge Clean Acc', marker='x', linestyle='-')

#     plt.xlabel('Remove Number')
#     plt.ylabel('Clean Accuracy')
#     plt.title('Clean Accuracy vs Remove Number')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(res_path + f'{label}_curve.png')
#     plt.close()



def plot_curve(neg_acc_clean, pos_acc_clean, neg_freq_ratios, pos_freq_ratios, neg_freq_thresholds, pos_freq_thresholds, label, save_path):
    plt.figure(figsize=(8, 6))

    plt.plot(neg_freq_ratios, neg_acc_clean, 'r-o', label='Negative Edge Removal')
    plt.plot(pos_freq_ratios, pos_acc_clean, 'b-o', label='Positive Edge Removal')

    for x, y, freq in zip(neg_freq_ratios, neg_acc_clean, neg_freq_thresholds):
        plt.annotate(f"{freq}", (x, y), textcoords="offset points", xytext=(0, 10),
                     ha='center', fontsize=8, color='red')

    for x, y, freq in zip(pos_freq_ratios, pos_acc_clean, pos_freq_thresholds):
        plt.annotate(f"{freq}", (x, y), textcoords="offset points", xytext=(0, -15),
                     ha='center', fontsize=8, color='blue')

    plt.xlabel("Edge Frequency Threshold (ratio × max frequency)")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy vs Frequency Ratio for Label {label}")
    plt.xticks(neg_freq_ratios)  # or freq_ratios if shared
    plt.gca().invert_xaxis()
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'_fre_curve_label_{label}.png'))
    plt.close()


from collections import Counter
def count_edge_frequency(edge_sets):
    freq = Counter()
    curvature_sum = defaultdict(float)

    for edge_set in edge_sets:
        for i, j, c in edge_set:
            key = tuple(sorted((i, j)))  # normalize direction for undirected edges
            freq[key] += 1
            curvature_sum[key] += c

    results = []
    for key in freq:
        avg_curv = curvature_sum[key] / freq[key]
        results.append((key[0], key[1], freq[key], avg_curv))

    return results




def remove_edge_cnn_union(args):
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

    
    train_loader, test_loader, valid_loader, valid_dataset, test_dataset = utils.get_new_data(selected_classes, data_train, data_test, test_bs=2000, valid_num=5000)

    # sep_dataloader = utils.sep_label(test_dataset, selected_classes, bs=5000)
    
    eps = [0.03]
    dims = cal_dims(model_dims)
    
    model_type = args.model_type
    model_pre_name = args.model_name
    res_path = args.mnist_res_path
    model_path = args.model_path
    metric = args.metric
    dataset = args.dataset
    alpha = args.alpha
    sample_size = args.sample_num
    data_path = args.mnist_data_path
    activation = args.activation
    
    if activation.lower() == "relu":
        from tools.LeNet5_custom_small import LeNet_custom_v2
    elif activation.lower() == "tanh":
        from tools.LeNet5_custom_small_tanh import LeNet_custom_v2
    
    model_full_n = model_type.lower() + model_pre_name.lower()

    dims = cal_dims(model_dims)
    prefix_dims = np.cumsum([0] + dims).tolist()
    
    if not os.path.exists(res_path):
        os.makedirs(res_path)
        
    # build model
    if model_pre_name == 'ori':
        model_name = "cnn_ori_"
    elif model_pre_name == 'adv':
        model_name = "cnn_adv_"
    elif model_pre_name == 'wd':
        model_name = "cnn_wd_"
        
    model_name = model_name + activation + ".pth"
    
    net_H = LeNet_custom_v2(model_dims, None, device)
    net_H.load_state_dict(torch.load(model_path + model_name))
    net_H = net_H.to(device)

    net_full = copy.deepcopy(net_H)

    print(model_name)
    # remove_frac = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1]
    remove_num = []
  
    test_cleanacc = test_clean(net_full, test_loader)
    # succ_pair, robust_pair = test(net_H, sep_dataloader, eps=e, alpha=2/255, iters=40, device=device)

    correct_suffix_res = dataset + "_res_correct.pkl"
    misclassified_suffix_res = dataset + "_res_misclassified.pkl"    
    res_name = model_full_n + metric + '_' + correct_suffix_res
    misres_name = model_full_n + metric + '_' + misclassified_suffix_res

    with open(data_path + res_name, 'rb') as file:
        res_dict = pickle.load(file)    

    freq_ratios = [1, 0.9, 0.8, 0.7, 0.5, 0.3, 0.2, 0.1, 0]
    
    with open(res_path + "edge_cnn_" + ".txt", "a+") as ff:
        ff.write(f'For model {model_name}: \n')
        ff.write(f'The clean accuracy for original model is {test_cleanacc}\n')
        # print(f'Current label {l}: \n')
        # ff.write(f'Current label {l}: \n')

        neg_acc_clean = []
        pos_acc_clean = []
        neg_edge_sets = []
        pos_edge_sets = []
        noseen_num = []
        for l in selected_classes:
            idx = 0
            for (ricci, batch, dim, node) in res_dict[l]:
                neg_e_other, pos_e, noseen = get_top_c(ricci, 1, prefix_dims)
                noseen_num.append(noseen)
                neg_edge_sets.append(neg_e_other)
                pos_edge_sets.append(pos_e)
                idx += 1
                if idx >= sample_size:
                    break

        neg_freq_dict = count_edge_frequency(neg_edge_sets)
        pos_freq_dict = count_edge_frequency(pos_edge_sets)
        neg_freq_edges_sorted = sorted(neg_freq_dict, key=lambda x: (-x[2], x[3]))
        pos_freq_edges_sorted = sorted(pos_freq_dict, key=lambda x: (-x[2], -x[3]))

        print(f'It has {len(neg_freq_edges_sorted)} negative curvature edges, {len(pos_freq_edges_sorted)} positive curvature egdes .. \n')
        ff.write(f'\nIt has {len(neg_freq_edges_sorted)} negative curvature edges, {len(pos_freq_edges_sorted)} positive curvature egdes .. \n')
        ff.write(f'\n The average noseen edges is {np.mean(noseen_num)}\n')
            
        neg_edges_only = [(i, j) for ((i, j), _,_) in neg_freq_edges_sorted]
        pos_edges_only = [(i, j) for ((i, j), _,_) in pos_freq_edges_sorted]

        # Compute overlap
        # Convert to sets for fast overlap calculation
        neg_set = set(neg_edges_only)
        pos_set = set(pos_edges_only)
        overlap = neg_set & pos_set
        overlap_count = len(overlap)
        ff.write(f"\nNumber of overlapping edges: {overlap_count}\n")

        # remove_num = [0, 5000, (int)(len(neg_edges_only)*0.3), (int)(len(neg_edges_only)*0.5), (int)(len(neg_edges_only)*0.7), len(neg_edges_only), (int)(len(pos_edges_only)*0.7), (int)(len(pos_edges_only)*0.9), (int)(len(pos_edges_only))]
        
        # Step 2: Choose thresholds — you can just use them all or downsample if too many
        neg_max_freq = max(freq for (_, freq) in neg_freq_edges_sorted)
        neg_freq_thresholds = [int(r * neg_max_freq) for r in freq_ratios]

        pos_max_freq = max(freq for (_, freq) in pos_freq_edges_sorted)
        pos_freq_thresholds = [int(r * pos_max_freq) for r in freq_ratios]
        
        # Step 3: For each threshold, count how many edges would be removed
        neg_remove_num = [sum(1 for (_, freq) in neg_freq_edges_sorted if freq >= t) for t in neg_freq_thresholds]
        pos_remove_num = [sum(1 for (_, freq) in pos_freq_edges_sorted if freq >= t) for t in pos_freq_thresholds]
            
        # start remove
        for index, rem_f in enumerate(neg_remove_num):
            ff.write(f'Remove edge number {rem_f}: \n')

            # remove second layer negative curvature edges
            net_neg = copy.deepcopy(net_H)
            net_neg.__build_remove_mask__(neg_edges_only, rem_f)
            # test acc
            acc_clean_neg = test_clean(net_neg, test_loader)
            neg_acc_clean.append(acc_clean_neg)

        for index, rem_f in enumerate(pos_remove_num):
            # remove positive curvature edges
            net_pos = copy.deepcopy(net_H)
            net_pos.__build_remove_mask__(pos_edges_only, rem_f)
            # test acc
            acc_clean_pos = test_clean(net_pos, test_loader)
            pos_acc_clean.append(acc_clean_pos)

            # excel_path = res_path + f'accuracies.xlsx'
            
            # # Create DataFrame for this fraction
            # df = pd.DataFrame({
            #     'Remove Number': [rem_f],
            #     'Negative Edge Clean Acc': [neg_acc_clean[-1]], 
            #     'Positive Edge Clean Acc': [pos_acc_clean[-1]]
            # })
            
            # # If file exists, append to it, otherwise create new
            # if os.path.exists(excel_path):
            #     existing_df = pd.read_excel(excel_path)
            #     df = pd.concat([existing_df, df], ignore_index=True)
                
            # df.to_excel(excel_path, index=False)    

        # plot_curve(neg_acc_clean, pos_acc_clean, remove_num, sample_size, res_path)
        plot_curve(neg_acc_clean, pos_acc_clean, freq_ratios, freq_ratios, neg_remove_num, pos_remove_num, sample_size, res_path)
                
