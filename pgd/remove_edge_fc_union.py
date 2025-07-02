import torch
from torchvision.datasets.mnist import MNIST
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import random
import os
import pandas as pd
import torch.nn as nn
from collections import defaultdict
import seaborn as sns
import copy
import pickle
import matplotlib.pyplot as plt

import sys
sys.path.append("..")

import tools.utils as utils
from tools.graph_curvature import graph_curvature_main_torch
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



# def get_top_c(curvature, b, prefix_dims, threshold = -50):
#     c = []
#     neg_e_second = set()  # Negative edges in second layer
#     neg_e_other = set()   # Negative edges in other layers 
#     pos_e = set()
    
#     for batch in range(b):
#         ricci_curv = np.array(curvature[batch])
#         for (i, j, curr) in ricci_curv:
#             if curr > 1:
#                 continue
#             else:
#                 c.append((i,j,curr))

#     c.sort(key=lambda x: x[2])

#     for (i,j,curr) in c:
#         i_layer = np.searchsorted(prefix_dims, i, side='right') - 1
#         i1 = (int)(i)
#         j1 = (int)(j)
#         if curr < 0:
#             neg_e_other.add((i1,j1,curr))
#         elif curr >= 0:
#             pos_e.add((i1,j1,curr))

#     return neg_e_other, pos_e


def get_top_c(curvature, b, prefix_dims, threshold=-50):
    c = []
    seen_edges = set()
    neg_e = set()
    pos_e = set()
    noseen = 0
    zero = 0.

    # Step 1: Collect existing curvature edges
    for batch in range(b):
        ricci_curv = np.array(curvature[batch])
        for (i, j, curr) in ricci_curv:
            i1, j1 = int(i), int(j)
            if curr > 1:
                continue
            c.append((i1, j1, curr))
            seen_edges.add((i1, j1))

    # Step 2: Generate all edges between adjacent layers
    all_edges = set()
    for l in range(len(prefix_dims) - 2):  # skip last layer
        start_i, end_i = prefix_dims[l], prefix_dims[l+1]
        start_j, end_j = prefix_dims[l+1], prefix_dims[l+2]
        for i in range(start_i, end_i):
            for j in range(start_j, end_j):
                all_edges.add((i, j))

    # Step 3: Add missing edges with default curvature = 1
    for (i, j) in all_edges:
        if (i, j) not in seen_edges:
            c.append((i, j, 1))
            noseen += 1
            
    # print(len(c))

    # Step 4: Sort and classify edges
    c.sort(key=lambda x: x[2])
    for (i, j, curr) in c:
        # i_layer = np.searchsorted(prefix_dims, i, side='right') - 1
        if curr < 0:
            neg_e.add((i, j, curr))
        elif curr >= 0:
            pos_e.add((i, j, curr))
            if curr == 0:
                zero += 1
    return neg_e, pos_e, noseen, zero



# def plot_curve(neg_acc_clean, pos_acc_clean, neg_freq_ratios, pos_freq_ratios, neg_freq_thresholds, pos_freq_thresholds, label, save_path):
#     plt.figure(figsize=(8, 6))

#     plt.plot(neg_freq_ratios, neg_acc_clean, 'r-o', label='Negative Edge Removal')
#     plt.plot(pos_freq_ratios, pos_acc_clean, 'b-o', label='Positive Edge Removal')

#     for x, y, freq in zip(neg_freq_ratios, neg_acc_clean, neg_freq_thresholds):
#         plt.annotate(f"{freq}", (x, y), textcoords="offset points", xytext=(0, 10),
#                      ha='center', fontsize=8, color='red')

#     for x, y, freq in zip(pos_freq_ratios, pos_acc_clean, pos_freq_thresholds):
#         plt.annotate(f"{freq}", (x, y), textcoords="offset points", xytext=(0, -15),
#                      ha='center', fontsize=8, color='blue')

#     plt.xlabel("Edge Frequency Threshold (ratio × max frequency)")
#     plt.ylabel("Accuracy")
#     plt.title(f"Accuracy vs Frequency Ratio for Label {label}")
#     plt.xticks(neg_freq_ratios)  # or freq_ratios if shared
#     plt.gca().invert_xaxis()
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_path, f'_fre_curve_label_{label}.png'))
#     plt.close()
    
    
def plot_curve(neg_clean_acc, pos_clean_acc, neg_remove_num, pos_remove_num, neg_end, pos_end, label, res_path):
    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(neg_remove_num, neg_clean_acc, label='Negative Edge Clean Acc', marker='o', linestyle='--')
    plt.plot(pos_remove_num, pos_clean_acc, label='Positive Edge Clean Acc', marker='x', linestyle='-')

    # Vertical lines
    plt.axvline(x=neg_end, color='red', linestyle=':', label=f'Neg End ({neg_end})')
    plt.axvline(x=pos_end, color='green', linestyle=':', label=f'Pos End ({pos_end})')

    plt.xlabel('Remove Number')
    plt.ylabel('Clean Accuracy')
    plt.title('Clean Accuracy vs Remove Number')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(res_path + f'{label}_curve_all.png')
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


def get_all_edges_sorted_by_weight(dims, weights, device='cuda'):
    """
    Returns:
        - edges_weights: List of tuples (edge_pair, weight)
    """
    batch_idx = 0
    weight_idx = 0
    global_node_offset = 0

    edges_weights = defaultdict(list)

    for i in range(len(dims) - 1):
        src_size, dst_size = dims[i], dims[i+1]
        num_edges = src_size * dst_size

        # Extract weights for this layer
        direct_dist = weights[batch_idx, weight_idx:weight_idx + num_edges].flatten()

        # Global node indices
        src_nodes = torch.arange(src_size, device=device) + global_node_offset
        dst_nodes = torch.arange(dst_size, device=device) + global_node_offset + src_size
        global_node_offset += src_size

        src_grid, dst_grid = torch.meshgrid(src_nodes, dst_nodes, indexing='ij')
        
        # Create edge-weight pairs
        for idx in range(len(direct_dist)):
            edge = (src_grid.flatten()[idx].item(), dst_grid.flatten()[idx].item())
            weight = direct_dist[idx].item()
            edges_weights[edge].append(weight)

        weight_idx += num_edges

    return edges_weights





def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def remove_edge_fc_union(args):
    set_seed(59)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    train_loader, test_loader, valid_loader, valid_dataset, test_dataset = utils.get_new_data(selected_classes, data_train, data_test, test_bs=2000, valid_num=5000)

    # sep_dataloader = utils.sep_label(test_dataset, selected_classes, bs=5000)
    
    eps = [0.03]

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
        from tools.small_model_relu import FC_MD
    elif activation.lower() == "tanh":
        from tools.small_model_tanh import FC_MD
    
    
    model_full_n = model_type.lower() + model_pre_name.lower()

    if not os.path.exists(res_path):
        os.makedirs(res_path)

    layer = [2]
    if model_type.lower() == "fc" and "big" not in model_pre_name.lower():
        layer = [2,4]

    for layer_num in layer:
        if model_type.lower() == "fc":
            correct_suffix_res = dataset + "_res_correct.pkl"
            misclassified_suffix_res = dataset + "_res_misclassified.pkl"  
            res_name = model_full_n + metric + '_' + str(layer_num) + correct_suffix_res
            misres_name = model_full_n + metric + '_' + str(layer_num) + misclassified_suffix_res                 
        # cnn model
        elif model_type.lower() == "cnn" and dataset.lower() == "mnist":
            correct_suffix_res = dataset + "_res_correct.pkl"
            misclassified_suffix_res = dataset + "_res_misclassified.pkl"    
            res_name = model_full_n + metric + '_' + correct_suffix_res
            misres_name = model_full_n + metric + '_' + misclassified_suffix_res    
        else:
            raise Exception("Invalid model type, model type should be {fc, fc_linear, cnn}!")
        
        with open(data_path + res_name, 'rb') as file:
            res_dict = pickle.load(file)
        
        remove_num = []
        
        # build model
        dims = model_zoo[layer_num]
        
        if model_pre_name.lower() == "ori" or model_pre_name.lower() == "decay":
            model_name = "best_ori_10l_" + str(layer_num) + ".pth"
        elif model_pre_name.lower() == "adv":
            model_name = "pgdtrain_" + str(layer_num) + ".pth"
        elif model_pre_name.lower() == 'big_adv':
            model_name = "big_adv_"
            dims = model_zoo[21]
        elif model_pre_name.lower() == 'big_ori':
            model_name = "big_ori_"
            dims = model_zoo[21]
        elif model_pre_name.lower() == 'big_wd':
            model_name = "big_wd_"
            dims = model_zoo[21]
        else:
            raise Exception("Invalid model name, model name should be {ori, decay, adv}!")
        
        model_name = model_name + activation + ".pth"
        
        prefix_dims = np.cumsum([0] + dims).tolist()
        
        print(f'Now for model {model_path + model_name}....\n')

        net_H = FC_MD(dims, layer_num)

        net_H.load_state_dict(torch.load(model_path + model_name))
        net_H = net_H.to(device)

        net_full = copy.deepcopy(net_H)
        test_cleanacc = test_clean(net_full, test_loader)

        # Define proportional thresholds
        freq_ratios = [1, 0.9, 0.8, 0.7, 0.5, 0.3, 0.2, 0.1, 0]

        # img = None
        # for count, (images, labels) in enumerate(train_loader):
        #     img = images.to(device)
        #     break

        # edge_array, nodes_ori, output, all_node = net_full.NN_info_batch(img.unsqueeze(0))
        # weights = output.detach().clone().to(device) 

        # edges_weights = get_all_edges_sorted_by_weight(dims, weights)

        # for l in selected_classes:
        with open(res_path + "edge_fc_" + str(layer_num) + ".txt", "w+") as ff:
            ff.write(f'For model {model_name}: \n')
            ff.write(f'The clean accuracy for original model is {test_cleanacc}\n')
            # print(f'Current label {l}: \n')
            # ff.write(f'Current label {l}: \n')
            
            neg_acc_clean = []
            pos_acc_clean = []
            neg_edge_sets = []
            pos_edge_sets = []
            noseen_num = []
            zero_c = 0
            for l in selected_classes:
                idx = 0
                for (ricci, batch, dim, node) in res_dict[l]:
                    neg_e, pos_e, noseen, zero = get_top_c(ricci, 1, prefix_dims)
                    noseen_num.append(noseen)
                    neg_edge_sets.append(neg_e)
                    pos_edge_sets.append(pos_e)

                    # for (i, j, c) in neg_e:
                    #     if c < -1000:
                    #         ff.write(f'{i}-{j}: {c}, node: {node[0][i]}, {node[0][j]}, weight: {edges_weights[(i,j)]}\n')
        
                    zero_c += zero
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
            ff.write(f'\n The average zero curvature edges is {zero_c/len(pos_edge_sets)}\n')
            
            neg_edges_only = [(i, j) for ((i, j), _,_) in neg_freq_edges_sorted]
            pos_edges_only = [(i, j) for ((i, j), _,_) in pos_freq_edges_sorted]
            
            # Convert to sets for fast overlap calculation
            neg_set = set(neg_edges_only)
            pos_set = set(pos_edges_only)

            # Compute overlap
            overlap = neg_set & pos_set
            overlap_count = len(overlap)
            ff.write(f"\nNumber of overlapping edges: {overlap_count}\n\n")
            
            neg_total = len(neg_edges_only)
            pos_total = len(pos_edges_only)

            # Generate uniformly spaced points (including 0 and total) for each list
            neg_remove_num = list(np.linspace(0, neg_total, num=10, dtype=int))
            pos_remove_num = list(np.linspace(0, pos_total, num=20, dtype=int))

            # # Step 2: Choose thresholds — you can just use them all or downsample if too many
            # neg_max_freq = max(freq for (_, freq) in neg_freq_edges_sorted)
            # neg_freq_thresholds = [int(r * neg_max_freq) for r in freq_ratios]

            # pos_max_freq = max(freq for (_, freq) in pos_freq_edges_sorted)
            # pos_freq_thresholds = [int(r * pos_max_freq) for r in freq_ratios]

            # # Step 3: For each threshold, count how many edges would be removed
            # neg_remove_num = [sum(1 for (_, freq) in neg_freq_edges_sorted if freq >= t) for t in neg_freq_thresholds]
            # pos_remove_num = [sum(1 for (_, freq) in pos_freq_edges_sorted if freq >= t) for t in pos_freq_thresholds]

            # start remove
            for index, rem_f in enumerate(neg_remove_num):
                print(f'Remove edge number {rem_f}:')
                cur_n = model_full_n + '_' + str(layer_num) + '_' + str(rem_f) + '_' + str(l)

                # remove negative curvature edges
                net_H.load_state_dict(torch.load(model_path + model_name))
                edge_r = Edge_Remove(net_H, dims, min(rem_f, len(neg_edges_only)), res_path)
                edge_r.e_remove(neg_edges_only, cur_n + "other_neg.pth")
                
                # test acc
                net_neg = FC_MD(dims, layer_num)

                net_neg.load_state_dict(torch.load(res_path + cur_n + "other_neg.pth"))
                net_neg = net_neg.to(device)
                os.remove(res_path + cur_n + "other_neg.pth")

                acc_clean_neg_other = test_clean(net_neg, test_loader)
                neg_acc_clean.append(acc_clean_neg_other)

            
            for index, rem_f in enumerate(pos_remove_num):
                # ff.write(f'Remove edge number {rem_f}: \n')
                cur_n = model_full_n + '_' + str(layer_num) + '_' + str(rem_f) + '_' + str(l)
                # remove positive curvature edges
                net_H.load_state_dict(torch.load(model_path + model_name))
                edge_r = Edge_Remove(net_H, dims, min(rem_f, len(pos_edges_only)), res_path)
                edge_r.e_remove(pos_edges_only, cur_n + "pos.pth")
                
                # test acc
                net_pos = FC_MD(dims, layer_num)

                net_pos.load_state_dict(torch.load(res_path + cur_n + "pos.pth"))
                net_pos = net_pos.to(device)
                os.remove(res_path + cur_n + "pos.pth")

                acc_clean_pos = test_clean(net_pos, test_loader)
                pos_acc_clean.append(acc_clean_pos)

                # excel_path = res_path + f'accuracies_layer{layer_num}.xlsx'
                
                # ## Create DataFrame for this fraction
                # df = pd.DataFrame({
                #     'Remove Number': [rem_f],
                #     'Negative Edge Clean Acc Other': [neg_acc_clean[-1]],
                #     'Positive Edge Clean Acc': [pos_acc_clean[-1]]
                # })

                # # If file exists, append to it, otherwise create new
                # if os.path.exists(excel_path):
                #     existing_df = pd.read_excel(excel_path)
                #     df = pd.concat([existing_df, df], ignore_index=True)
                    
                # df.to_excel(excel_path, index=False)     
                
            ff.write(f'\n\n')

            plot_curve(neg_acc_clean, pos_acc_clean, neg_remove_num, pos_remove_num, neg_total, pos_total, sample_size, res_path)
            # plot_curve(neg_acc_clean, pos_acc_clean, freq_ratios, freq_ratios, neg_remove_num, pos_remove_num, sample_size, res_path)



    
