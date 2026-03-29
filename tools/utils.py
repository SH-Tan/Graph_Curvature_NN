import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from GraphRicciCurvature.OllivierRicci import OllivierRicci
import networkx as nx


def plot_acc(acc_1, acc_2, name, path, acc_3 = None, acc_4 = None, model_size = None, avg_w = None, avg_c = None):
    fig, ax = plt.subplots()
    
    x = range(len(acc_1))
    x2 = range(len(acc_2))

    max_n = max(max(max(acc_1), max(acc_2)), 0)
    # plot curves
    ax.plot(x, acc_1, label="ori acc", color = 'red', linewidth = 1)
    ax.plot(x2, acc_2, label="ori adv acc", color='green', linewidth = 1)
    
    if (acc_3):
        ax.plot(x, acc_3, label="sub acc", color = 'blue', linewidth = 1)
    if (acc_4):
        ax.plot(x, acc_4, label= "sub adv acc", color = 'cyan', linewidth = 1)
    if model_size:
        ax.plot(x, model_size, label="Model size", color='yellow', linewidth = 2)
    
    ax.set_ylabel('Number', fontsize = 12)
    ax.set_xlabel('Sample', fontsize = 12)
    
    ax.set_xlim(0, len(acc_1))
    ax.set_ylim(0, max_n + 0.1)
    
    x_ticks = np.arange(0, len(acc_1), 1)
    y_ticks = np.arange(0, max_n + 0.1, 0.05)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    
    ax2 = ax.twinx() 
    if avg_w:   
        x1 = range(len(avg_w))
        ax2.plot(x1, avg_w, label="Weights variance", color='black', linewidth = 2)
        ax2.set_ylabel('Variance', fontsize = 12)
        ax2.set_ylim(0, max(avg_w) + 20)
        y2_ticks = np.arange(0, max(avg_w) + 20, 100)
        ax2.set_yticks(y2_ticks)
        
    if avg_c:
        ax2.plot(x, avg_c, label="Curvature variance", color='gray', linewidth = 2)
        # ax2.set_ylim(min(avg_c)-1.0, 0)

        # y2_ticks = np.arange(min(avg_c)-1.0, 0, 0.01)
        # ax2.set_yticks(y2_ticks)
        
    ax2.legend()

    ax.set_title(name, fontsize=14)
    ax.legend()

    plt.savefig(path + name + ".png")
    plt.close()
    
    
def remove_e(adj_m, edge_set):
    for (n1, n2) in edge_set:
        adj_m[n1, n2] = 0
    return adj_m



def build_adjm(loader, net, nodes_num, dims, device, flag = False):
    eps = 10e9
    
    for i, (img, label) in enumerate(loader):    
        img = img.to(device)
        edge_array, nodes = net.edge_w_batch(img)
        edge_array = edge_array.cpu().detach().numpy() 
        output = net.get_weights(img)
        output = output.cpu().detach().numpy() 
        
        output[edge_array == 0] = 0.
        
        if (i == 0):
            data = output
        else:
            data = np.vstack((data,output))
    
    w_avg = np.mean(data, axis=0) # (edge num,)
    w_avg = np.abs(w_avg) # absolate edge value for one image
    
    print(f'Tha max weight is {np.max(w_avg)}, min is {np.min(w_avg)}....')
            
    # build adjacent matrix
    adjacent_m = np.zeros((nodes_num, nodes_num), dtype=np.float32)

    d_num = len(dims)
    cur_s_col = dims[0]
    cur_e_col = dims[0] + dims[1]

    cur_layer = 1
    start_col = 0
    end_col = dims[1]

    for i in range(nodes_num - dims[d_num-1]):
        # print(f'i : {i}, start col : {cur_s_col}, end_col : {cur_e_col}, from {start_col} to {end_col}')
        adjacent_m[i, cur_s_col : cur_e_col] = w_avg[start_col : end_col]
        
        if (cur_layer < d_num-1 and i == cur_s_col - 1):
            cur_layer += 1
            start_col = end_col
            end_col = end_col + dims[cur_layer]
            cur_s_col = cur_e_col
            cur_e_col = cur_e_col + dims[cur_layer]
        else:
            start_col = end_col
            end_col = end_col + dims[cur_layer]
            
    # ind = np.where(adjacent_m < 0)

    # i = ind[0]
    # j = ind[1]

    # adjacent_m[j,i] = -adjacent_m[i,j]
    # adjacent_m[adjacent_m<0] = 0
            
    return adjacent_m



def get_new_data(l1, data_train, data_test, train_bs = 128, test_bs = 2000, valid_num = 2500):
    # selected classes
    train_i1 = torch.tensor([i for i, (_, label) in enumerate(data_train) if label in l1])
    test_i1 = torch.tensor([i for i, (_, label) in enumerate(data_test) if label in l1])
    
    # valid_num = min(valid_num, 0.1*len(train_i1))
    
    train_index = torch.randperm(len(train_i1))
    valid_dataset = torch.utils.data.Subset(data_train, train_i1[train_index[0:valid_num]])
    train_dataset = torch.utils.data.Subset(data_train, train_i1[train_index[valid_num:,]])
    test_dataset = torch.utils.data.Subset(data_test, test_i1)
    
    train_loader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=test_bs, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=test_bs, num_workers=2)
    
    return train_loader, test_loader, valid_loader, valid_dataset, test_dataset


import random
def sep_label(dataset, ls, bs = 5000, seed=4):
    # Global seeds
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)
    
    sep_dataloader = dict()
    for l in ls:
        index = torch.tensor([i for i, (_, label) in enumerate(dataset) if label == l])
        subset = torch.utils.data.Subset(dataset, index)
        loader = DataLoader(subset, batch_size=bs, num_workers=2, shuffle=True, worker_init_fn=seed_worker, generator=g,)

        sep_dataloader[l] = loader
        
    return sep_dataloader



def get_net_info(net):
    neural_list = []
    nodes_num = 0
    edges_num = 0
    i = 0
    for p in net.parameters():
        if i == 0:
            nodes_num += p.shape[1]
        if i%2 == 0:
            nodes_num += p.shape[0]
            edges_num += (p.shape[0] * p.shape[1])
            neural_list.append(p.shape[0])
        i += 1
        
    print(f'Total nodes are {nodes_num}, total edges are {edges_num}.')
    
    return nodes_num, edges_num



def test(n, loader, device):
    n.eval()
    total_correct = 0
    
    for i, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        output = n(images)
        pred = output.detach().max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()

    # print(f'Test Accuracy for label {l}: {(float(total_correct) / len(loader.dataset)):.3f}')
    
    acc = float(total_correct) / len(loader.dataset)
    return acc


def standard_PGD(model, images, labels, device, eps=.1, alpha=.1, iters=100):
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


def standard_PGD_test(loader, net, device, eps=.1, alpha=.1, iters=100):
    net.eval()
    
    ori_acc = 0.
    adv_acc = 0.
    
    for i, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        output1 = net(images)
        pred1 = output1.detach().max(1)[1]
        ori_acc += pred1.eq(labels.view_as(pred1)).sum()
        
        # adv_img = pgd_attack(net, images, labels, dims, mask_dict, target=target)
        adv_img = standard_PGD(net, images, labels, device, eps, alpha, iters)
        
        output3 = net(adv_img)
        pred3 = output3.detach().max(1)[1]
        adv_acc += pred3.eq(labels.view_as(pred3)).sum()
        

    ori_acc = float(ori_acc) / len(loader.dataset)
    adv_acc = float(adv_acc) / len(loader.dataset)

    return adv_acc


import torch.nn.functional as F

def build_cnn_adj(src_n, dst_n, model_dims, layer, w):
    w_avg = w
    
    # build adjacent matrix
    adjacent_m = np.zeros((src_n, dst_n), dtype=np.float32)
    adjacent_m = np.full(adjacent_m.shape, -1.)

    current_l = layer
    start_col = 0
    cur_s_col = 0
    nodes_total = 0
    next_l= current_l + 1

    cur_name = model_dims[current_l]["name"]
    cur_dim = model_dims[current_l]["dim"]
    cur_size = cur_dim['out_size']

    cur_channel = 1 if (cur_name == "fc") else cur_dim['channel']

    if cur_name == "input":
        cur_nodes = cur_channel * cur_size**2
    else:
        cur_nodes = cur_size if (cur_name == "fc") else cur_dim['channel']*(cur_size**2)

    nxt_name = model_dims[next_l]["name"]
    nxt_dim = model_dims[next_l]["dim"]
    out_size = nxt_dim['out_size']

    # cnn layer
    if (nxt_name == "cnn" or nxt_name == "pooling"):
        k = nxt_dim['kernel']
        s = nxt_dim['stride']
        c = nxt_dim['channel']
        
        step = k**2      
        n = 0
        tensor_2d = torch.arange(cur_nodes).reshape(1,cur_channel,cur_size,cur_size).float()
        
        if (nxt_name == "cnn"):
            indices = F.unfold(tensor_2d, (k,k), stride = s).transpose(1,2).int()
            end_col = start_col + step*cur_channel
            for ur_c in range(c): 
                for l in range(indices.shape[1]):
                    cur_idx = indices[0,l] + nodes_total 
                    cur_idx = cur_idx.tolist()
                    assert(len(cur_idx) == step*cur_channel)
                    adjacent_m[cur_idx, nodes_total+cur_nodes+n] = w_avg[start_col : end_col]
        
                    start_col = end_col
                    end_col = start_col + step*cur_channel
                    n += 1
        else:
            indices = F.unfold(tensor_2d, (k,k), stride = s)
            i_unf = indices.view(1, cur_channel, k*k, -1).transpose(2,3)
            indices = i_unf.reshape(i_unf.shape[0], i_unf.shape[1]*i_unf.shape[2], i_unf.shape[3]).int()
            
            for l in range(indices.shape[1]):
                end_col = start_col + step
                cur_idx = indices[0,l] + nodes_total 
                cur_idx = cur_idx.tolist()
                assert(len(cur_idx) == step)
                adjacent_m[cur_idx, nodes_total+cur_nodes+n] = w_avg[start_col : end_col]

                start_col = end_col
                end_col = start_col + step
                n += 1
                
        nodes_total += cur_nodes
        # print(start_col)  

    # fc layer
    elif (nxt_name == "fc"):  
        cur_s_col = nodes_total + cur_nodes
        cur_e_col = nodes_total + cur_nodes + out_size

        end_col = start_col + out_size

        for node in range(nodes_total, nodes_total + cur_nodes, 1):
            # print(f'i : {node}, start col : {cur_s_col}, end_col : {cur_e_col}, from {start_col} to {end_col}')
            adjacent_m[node, cur_s_col : cur_e_col] = w_avg[start_col : end_col]        
            
            start_col = end_col
            end_col = end_col + out_size            
                
        nodes_total += cur_nodes
    # print(nodes_total)
        
    return adjacent_m



def build_cnn_unfold_index_table(in_ch: int, in_size: int, k: int, stride: int, padding: int, device = 'cpu'):
    """
    Returns:
      unfolded (torch.LongTensor): shape (patches, L) where L = in_ch * k * k, containing input local indices
      patches (int)
      L (int) = in_ch * k * k
      per_patch_map (List[dict]): per_patch_map[p] maps input_local_index -> column_idx
    """
    src_size = in_ch * in_size * in_size
    dummy = torch.arange(src_size, device=device).reshape(1, in_ch, in_size, in_size).float()
    unfolded = F.unfold(dummy, kernel_size=(k, k), stride=stride, padding=padding).transpose(1, 2).long()
    unfolded = unfolded[0]  # (patches, L)
    patches = unfolded.shape[0]
    L = unfolded.shape[1]

    per_patch_map = []
    unfolded_np = unfolded.cpu().numpy()
    for p in range(patches):
        col_indices = unfolded_np[p]
        d = {int(col_indices[c]): int(c) for c in range(len(col_indices))}
        per_patch_map.append(d)

    return unfolded, patches, L, per_patch_map



def min_plus_mult(a, b, chunk_k=256, chunk_p=256):
    """
    Fully chunked min-plus matrix multiplication.
    a: [batch, m, n]
    b: [batch, n, p]
    Returns: [batch, m, p]
    """
    batch_size, m, n = a.shape
    _, n2, p = b.shape
    assert n == n2, "Dimension mismatch"

    device = a.device
    result = torch.full((batch_size, m, p), float('inf'), device=device)

    for start_k in range(0, n, chunk_k):
        end_k = min(start_k + chunk_k, n)

        # Preselect the shared chunk
        a_chunk = a[:, :, start_k:end_k]  # [B, m, chunk_k]
        b_chunk = b[:, start_k:end_k, :]  # [B, chunk_k, p]

        for start_p in range(0, p, chunk_p):
            end_p = min(start_p + chunk_p, p)

            # Compute partial block
            b_sub = b_chunk[:, :, start_p:end_p]  # [B, chunk_k, chunk_p]

            # [B, m, chunk_k, chunk_p]
            # Add in broadcasted manner
            partial = (a_chunk.unsqueeze(3) + b_sub.unsqueeze(1)).min(dim=2)[0]  # [B, m, chunk_p]

            # Update the result for this block
            result[:, :, start_p:end_p] = torch.minimum(result[:, :, start_p:end_p], partial)

    return result


# For CNN
def cnn_layerwise_shortest_path_torch(model_dims, weights, prefix_dims, device='cuda'):
    batch_size, weight_num = weights.shape
    
    total_layers = len(model_dims) - 1  # transitions

    shortest_paths = {}
    inf = torch.tensor(float('inf'), device=device)
    
    # ---- Determine computation region ----
    start_layer = 0
    end_layer = total_layers - 1
    
    weight_idx = 0
    for i in range(start_layer, end_layer + 1):
        l = i + 1
        current_layer = model_dims[l+1]
        src_size = prefix_dims[i+1] - prefix_dims[i]
        dst_size = prefix_dims[i+2] - prefix_dims[i+1]
        
        if current_layer['name'] == 'fc':
            # FC layer handling
            direct_dist = weights[:, weight_idx:weight_idx+src_size*dst_size]
            direct_dist = direct_dist.view(batch_size, src_size, dst_size)
            
            shortest_paths[(i, i+1)] = torch.where(direct_dist > 0, direct_dist, inf)
            weight_idx += src_size * dst_size
            
            # f.write(f'{i}-{i+1}: {shortest_paths[(i, i+1)]}\n')
            
        elif current_layer['name'] in ['cnn', 'pooling']:
            # CNN layer handling
            k = current_layer['dim']['kernel']
            s = current_layer['dim']['stride']
            # p = current_layer['dim']['padding']
            in_size = model_dims[l]['dim']['out_size']
            pre_ch = model_dims[l]['dim'].get('channel', 1)
            cur_ch = current_layer['dim']['channel']
            padding = current_layer['dim'].get('padding', 0)
            pool = current_layer['dim'].get('pool', False)
            if pool:
                dst_size = dst_size * 2 * 2
            else:
                dst_size = dst_size
            
            adjacent_m = torch.zeros((batch_size, src_size, dst_size), dtype=torch.float32, device=device)
            
            # Generate receptive field indices
            dummy = torch.arange(src_size, device=device).reshape(1, pre_ch, in_size, in_size).float()

            # Unfold operation to get receptive field indices
            unfolded = F.unfold(dummy, kernel_size=k, stride=s, padding=padding).transpose(1, 2).int()
            patches = unfolded.shape[1]

            step = k**2 
            
            # Create indices matrix
            n = 0
            end_col = weight_idx + step*pre_ch
            for c in range(cur_ch):
                for p in range(patches):
                    cur_idx = unfolded[0,p].tolist()
                    adjacent_m[:, cur_idx, n] = weights[:, weight_idx : end_col]
                    
                    weight_idx = end_col
                    end_col = weight_idx + step*pre_ch
                    n += 1
            
            shortest_paths[(i, i+1)] = torch.where(adjacent_m > 0, adjacent_m, inf)
            
    def adaptive_chunksize(max_chunk=512):
        free_mem, total_mem = torch.cuda.mem_get_info()
        gb_free = free_mem / (1024**3) 
        if gb_free < 10:
            return 64, 64
        elif gb_free < 16:
            return 128, 128
        elif gb_free < 32:
            return 256, 256
        else:
            return max_chunk, max_chunk

    max_hops = 3

    for d in range(2, max_hops + 1):   # only 2-hop and 3-hop
        for i in range(start_layer, end_layer - d + 2):
            j = i + d
            if j > end_layer + 1:
                continue

            # Initialize current_min tensor
            current_min = torch.full(
                (batch_size,
                prefix_dims[i + 1] - prefix_dims[i],
                prefix_dims[j + 1] - prefix_dims[j]),
                float('inf'),
                device=device
            )
            
            # Adaptive chunk sizes for available memory
            chunk_k, chunk_p = adaptive_chunksize()
            print(f"({i}, {j}) -> shape={current_min.shape}, chunk=({chunk_k},{chunk_p})")

            # Combine intermediate paths
            for k in range(i + 1, j):
                if (i, k) not in shortest_paths or (k, j) not in shortest_paths:
                    continue

                current_min = torch.minimum(
                    current_min,
                    min_plus_mult(
                        shortest_paths[(i, k)],
                        shortest_paths[(k, j)],
                        chunk_k=chunk_k,
                        chunk_p=chunk_p
                    )
                )

            # Save computed shortest path for this (i, j)
            shortest_paths[(i, j)] = current_min
            
    return shortest_paths


