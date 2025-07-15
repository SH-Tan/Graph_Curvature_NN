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
import re
import gc

import sys
sys.path.append("..")

import tools.utils as utils
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



def process_batches_memory_efficient(
    data_path: str,
    model_full_n: str,
    metric: str,
    dataset: str,
    sample_size: int,
    prefix_dims
):

    prefix = f"{model_full_n}_{metric}_{dataset}_batch"
    suffix = ".pkl"

    def extract_batch_num(filename: str) -> int:
        match = re.search(r'batch(\d+)', filename)
        return int(match.group(1)) if match else -1

    all_files = sorted(
        [f for f in os.listdir(data_path) if f.startswith(prefix) and f.endswith(suffix)],
        key=extract_batch_num
    )

    print(f"Found files: {all_files}")

    label_counts = {l: 0 for l in selected_classes}
    neg_edges_all = []  # list of (i, j, curvature)
    pos_edges_all = []

    for filename in all_files:
        file_path = os.path.join(data_path, filename)
        with open(file_path, 'rb') as f:
            batch_data = pickle.load(f)

        for l in selected_classes:
            if label_counts[l] >= sample_size:
                continue

            new_data = batch_data.get(l, [])
            available = sample_size - label_counts[l]
            use_data = new_data[:available]

            for ricci in use_data:
                neg_e, pos_e, _ = get_top_c(ricci, b=1, prefix_dims=prefix_dims)
                neg_edges_all.extend(neg_e)
                pos_edges_all.extend(pos_e)
                label_counts[l] += 1

            del new_data, use_data
            gc.collect()

        del batch_data
        gc.collect()

        if all(label_counts[l] >= sample_size for l in selected_classes):
            break

    print("Finished processing all required batches.")

    # === Frequency + curvature summarization ===
    def count_edge_frequency(edge_list):
        freq = Counter()
        curvature_sum = defaultdict(float)
        for i, j, c in edge_list:
            key = tuple(sorted((i, j)))  # undirected
            freq[key] += 1
            curvature_sum[key] += c
        results = [
            (i, j, freq[(i, j)], curvature_sum[(i, j)] / freq[(i, j)])
            for (i, j) in freq
        ]
        return results

    neg_summary = sorted(count_edge_frequency(neg_edges_all), key=lambda x: x[3])      # sort by avg curvature ↑
    pos_summary = sorted(count_edge_frequency(pos_edges_all), key=lambda x: -x[3])     # sort by avg curvature ↓

    return neg_summary, pos_summary


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



def get_top_c(curvature, b, prefix_dims):
    c = []
    seen_edges = set()
    neg_e = set()
    pos_e = set()
    noseen = 0

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
    # all_edges = set()
    # for l in range(len(prefix_dims) - 2):  # skip last layer
    #     start_i, end_i = prefix_dims[l], prefix_dims[l+1]
    #     start_j, end_j = prefix_dims[l+1], prefix_dims[l+2]
    #     for i in range(start_i, end_i):
    #         for j in range(start_j, end_j):
    #             all_edges.add((i, j))

    # # Step 3: Add missing edges with default curvature = 1
    # for (i, j) in all_edges:
    #     if (i, j) not in seen_edges:
    #         c.append((i, j, 1))
    #         noseen += 1
            
    # print(len(c))

    # Step 4: Sort and classify edges
    c.sort(key=lambda x: x[2])
    for (i, j, curr) in c:
        # i_layer = np.searchsorted(prefix_dims, i, side='right') - 1
        if curr < 0:
            neg_e.add((i, j, curr))
        elif curr > 0:
            pos_e.add((i, j, curr))

    return neg_e, pos_e, noseen




def plot_curve_all_eps(
    neg_acc_adv, pos_acc_adv,
    neg_remove_num, pos_remove_num,
    neg_end, pos_end,
    eps_list, label, res_path
):
    plt.figure(figsize=(10, 6))

    # Separate CMYK-safe colors for NEG and POS
    neg_colors = ['#00A3E0', '#6CACE4', '#00AB84', '#9E1B32']  # Cyan, Blue-gray, Greenish cyan, Dark red
    pos_colors = ['#EC008C', '#FF6F61', '#FEDD00', '#000000']  # Magenta, Warm red, Yellow, Black

    # Distinct markers per curve
    neg_markers = ['o', 's', '^', 'D']
    pos_markers = ['v', 'x', '*', '+']

    # Keep handles for separate legends
    curve_handles = []
    stop_handles = []

    for i, eps in enumerate(eps_list):
        neg_color = neg_colors[i % len(neg_colors)]
        pos_color = pos_colors[i % len(pos_colors)]
        neg_mk = neg_markers[i % len(neg_markers)]
        pos_mk = pos_markers[i % len(pos_markers)]
        eps_label = f"ε={eps}"

        # Plot negative accuracy
        h1, = plt.plot(
            neg_remove_num,
            neg_acc_adv[eps],
            label=f'Neg {eps_label}',
            linestyle='-',
            marker=neg_mk,
            color=neg_color,
            linewidth=2,
            markersize=7
        )
        curve_handles.append(h1)

        # Plot positive accuracy
        h2, = plt.plot(
            pos_remove_num,
            pos_acc_adv[eps],
            label=f'Pos {eps_label}',
            linestyle='--',
            marker=pos_mk,
            color=pos_color,
            linewidth=2,
            markersize=7
        )
        curve_handles.append(h2)

    # Vertical lines
    h3 = plt.axvline(
        x=neg_end, color='black', linestyle=':', linewidth=1.8,
        label=f'Neg End @ {neg_end}'
    )
    h4 = plt.axvline(
        x=pos_end, color='gray', linestyle=':', linewidth=1.8,
        label=f'Pos End @ {pos_end}'
    )
    stop_handles.extend([h3, h4])

    # Labels and styles
    plt.xlabel('Number of Edges Removed', fontsize=20)
    plt.ylabel('Adversarial Accuracy', fontsize=20)
    plt.title('Adversarial Accuracy vs. Edge Removal', fontsize=22)
    plt.ylim(0.0, 1.0)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)

    # First legend: curves
    first_legend = plt.legend(handles=curve_handles, fontsize=12, loc='upper right', ncol=2, title='Adversarial Curves')
    plt.gca().add_artist(first_legend)  # Add it before second

    # Second legend: vertical lines
    plt.legend(handles=stop_handles, fontsize=12, loc='lower right', title='Edge Removal End')

    plt.tight_layout()
    plt.savefig(res_path + f'{label}_curve_all.pdf', dpi=300, format='pdf')  # CMYK-friendly
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




def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def remove_edge_fc_union_pgd(args):
    set_seed(59)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    train_loader, test_loader, valid_loader, valid_dataset, test_dataset = utils.get_new_data(selected_classes, data_train, data_test, test_bs=2000, valid_num=5000)

    # sep_dataloader = utils.sep_label(test_dataset, selected_classes, bs=5000)
    
    eps = [0.03, 0.07, 0.1]

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

        # for l in selected_classes:
        with open(res_path + "edge_fc_" + str(layer_num) + ".txt", "w+") as ff:
            ff.write(f'For model {model_name}: \n')
            ff.write(f'The clean accuracy for original model is {test_cleanacc}\n')
            # print(f'Current label {l}: \n')
            # ff.write(f'Current label {l}: \n')
            
            neg_acc_adv = defaultdict(list)
            pos_acc_adv = defaultdict(list)
            
            neg_summary, pos_summary = process_batches_memory_efficient(
                data_path,
                model_full_n,
                metric,
                dataset,
                sample_size,
                prefix_dims
            )

            print(f'It has {len(neg_summary)} negative curvature edges, {len(pos_summary)} positive curvature edges .. \n')
            ff.write(f'\nIt has {len(neg_summary)} negative curvature edges, {len(pos_summary)} positive curvature edges .. \n')

            neg_edges_only = [(i, j) for (i, j, _, _) in neg_summary]
            pos_edges_only = [(i, j) for (i, j, _, _) in pos_summary]

            neg_total = len(neg_edges_only)
            pos_total = len(pos_edges_only)

            neg_remove_num = list(np.linspace(0, neg_total, num=10, dtype=int))
            pos_remove_num = list(np.linspace(0, pos_total, num=20, dtype=int))

            # Step 2: Choose thresholds — you can just use them all or downsample if too many
            # neg_max_freq = max(freq for (_, _, freq, _) in neg_summary)
            # neg_freq_thresholds = [int(r * neg_max_freq) for r in freq_ratios]

            # pos_max_freq = max(freq for (_, _, freq, _) in pos_summary)
            # pos_freq_thresholds = [int(r * pos_max_freq) for r in freq_ratios]

            # # Step 3: For each threshold, count how many edges would be removed
            # neg_remove_num = [sum(1 for (_, _, freq, _) in neg_summary if freq >= t) for t in neg_freq_thresholds]
            # pos_remove_num = [sum(1 for (_, _, freq, _) in pos_summary if freq >= t) for t in pos_freq_thresholds]

            # start remove
            for index, rem_f in enumerate(neg_remove_num):
                print(f'Remove edge number {rem_f}:')
                cur_n = model_full_n + '_' + str(layer_num) + '_' + str(rem_f)

                # remove negative curvature edges
                net_H.load_state_dict(torch.load(model_path + model_name))
                edge_r = Edge_Remove(net_H, dims, min(rem_f, len(neg_edges_only)), res_path)
                edge_r.e_remove(neg_edges_only, cur_n + "other_neg.pth")
                
                # test acc
                net_neg = FC_MD(dims, layer_num)

                net_neg.load_state_dict(torch.load(res_path + cur_n + "other_neg.pth"))
                net_neg = net_neg.to(device)
                os.remove(res_path + cur_n + "other_neg.pth")
                
                for e in eps:
                    test_advacc = test_adversarial(net_neg, test_loader, eps=e, alpha=2/255, iters=40)
                    neg_acc_adv[e].append(test_advacc)

            for index, rem_f in enumerate(pos_remove_num):
                # ff.write(f'Remove edge number {rem_f}: \n')
                cur_n = model_full_n + '_' + str(layer_num) + '_' + str(rem_f) 
                # remove positive curvature edges
                net_H.load_state_dict(torch.load(model_path + model_name))
                edge_r = Edge_Remove(net_H, dims, min(rem_f, len(pos_edges_only)), res_path)
                edge_r.e_remove(pos_edges_only, cur_n + "pos.pth")
                
                # test acc
                net_pos = FC_MD(dims, layer_num)

                net_pos.load_state_dict(torch.load(res_path + cur_n + "pos.pth"))
                net_pos = net_pos.to(device)
                os.remove(res_path + cur_n + "pos.pth")

                for e in eps:
                    test_advacc = test_adversarial(net_pos, test_loader, eps=e, alpha=2/255, iters=40)
                    pos_acc_adv[e].append(test_advacc)
      
            ff.write(f'\n\n')

            plot_curve_all_eps(neg_acc_adv, pos_acc_adv, neg_remove_num, pos_remove_num, neg_total, pos_total, eps, sample_size, res_path)
            # plot_curve(neg_acc_clean, pos_acc_clean, freq_ratios, freq_ratios, neg_remove_num, pos_remove_num, sample_size, res_path)



    
