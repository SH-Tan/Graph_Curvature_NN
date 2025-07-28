import torch
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
import numpy as np
import random
import os
import copy
import yaml
from .Car import World

import matplotlib.pyplot as plt
import pandas as pd

import sys
sys.path.append("..")

import tools.utils as utils
from tools.controller import Controller
from tools.edge_remove import Edge_Remove

np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=torch.inf)

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

model_zoo = {
    '64': [21, 64, 64, 1],
    '128': [21, 128, 128, 1]
}

series = [1,2,3]

def normalize(s):
    mean = [2.5]
    spread = [5.0]
    return (s - mean) / spread


def test(controller, device):
    numTrajectories = 1000
    hallWidths = [1.5, 1.5, 1.5, 1.5]
    hallLengths = [20, 20, 20, 20]
    turns = ['right', 'right', 'right', 'right']
    car_dist_s = hallWidths[0]/2.0
    car_dist_f = 9.9
    car_heading = 0
    episode_length = 70
    time_step = 0.1

    lidar_field_of_view = 115
    lidar_num_rays = 21

    # Change this to 0.1 or 0.2 to generate Figure 3 or 5 in the paper, respectively
    lidar_noise = 0.

    # Change this to 0 or 5 to generate Figure 3 or 5 in the paper, respectively
    missing_lidar_rays = 0

    num_unsafe = 0
    
    w = World(hallWidths, hallLengths, turns,\
                car_dist_s, car_dist_f, car_heading,\
                episode_length, time_step, lidar_field_of_view,\
                lidar_num_rays, lidar_noise, missing_lidar_rays, False)

    throttle = 16
    
    controller.eval()

    for step in range(numTrajectories):
        w.reset()
        observation = w.scan_lidar()

        rew = 0
        traj = []
        for e in range(episode_length):
            observation = normalize(observation)
            traj.append(observation)
            
            obs_tensor = torch.tensor(observation, dtype=torch.float64, device=device).unsqueeze(0)  # shape: (1, len(observation))
            output = controller(obs_tensor)
            pred = output.cpu().detach().numpy()[0][0]
            delta = 15 * pred
            # delta = delta.detach().max(1)[1]
            
            observation, reward, done, info = w.step(delta, throttle)
            if done:
                if e < episode_length - 1:
                    num_unsafe += 1
                break
            rew += reward

    print('number of crashes: ' + str(num_unsafe))
    return num_unsafe



def get_all_edges_sorted_by_weight(dims, weights, device='cuda'):
    """
    Returns:
        - sorted_edges: [2, total_edges], LongTensor, globally indexed
        - sorted_weights: [total_edges], FloatTensor, sorted descending
    """
    batch_idx = 0
    weight_idx = 0
    global_node_offset = 0

    all_edges = []
    all_weights = []

    for i in range(len(dims) - 1):
        src_size, dst_size = dims[i], dims[i+1]
        num_edges = src_size * dst_size

        # Extract weights for this layer
        direct_dist = weights[batch_idx, weight_idx:weight_idx + num_edges]

        # Global node indices
        src_nodes = torch.arange(src_size, device=device) + global_node_offset
        dst_nodes = torch.arange(dst_size, device=device) + global_node_offset + src_size
        global_node_offset += src_size

        src_grid, dst_grid = torch.meshgrid(src_nodes, dst_nodes, indexing='ij')
        edges = torch.stack([src_grid.flatten(), dst_grid.flatten()])  # [2, num_edges]

        all_edges.append(edges)
        all_weights.append(direct_dist.flatten())

        weight_idx += num_edges

    # Combine all
    all_edges = torch.cat(all_edges, dim=1)      # [2, total_edges]
    all_weights = torch.cat(all_weights, dim=0)  # [total_edges]
    
    abs_weights = torch.abs(all_weights)

    # Sort by descending |weight|
    sorted_indices_high = torch.argsort(abs_weights, descending=True)
    sorted_edges_high = all_edges[:, sorted_indices_high]

    # Sort by ascending |weight|
    sorted_indices_low = torch.argsort(abs_weights, descending=False)
    sorted_edges_low = all_edges[:, sorted_indices_low]

    # # Sort by descending weight
    # sorted_indices = torch.argsort(all_weights, descending=True)
    # sorted_edges_high = all_edges[:, sorted_indices]
    # # sorted_weights = all_weights[sorted_indices]
    
    # # Sort by ascending weight
    # sorted_indices = torch.argsort(all_weights, descending=False)
    # sorted_edges_low = all_edges[:, sorted_indices]
    # # sorted_weights = all_weights[sorted_indices]

    return sorted_edges_high, sorted_edges_low


from collections import defaultdict
def separate_edges_by_layer_in_order(sorted_edges, prefix_dims):
    layer_to_edges = defaultdict(list)
    src_nodes = sorted_edges[0]
    dst_nodes = sorted_edges[1]

    for i_raw, j_raw in zip(src_nodes, dst_nodes):
        i = i_raw
        j = j_raw

        src_layer = np.searchsorted(prefix_dims, i, side='right') - 1
        dst_layer = np.searchsorted(prefix_dims, j, side='right') - 1

        # Detect layer transition from src_layer → dst_layer
        if dst_layer == src_layer + 1:
            layer_to_edges[src_layer].append((i_raw, j_raw))  # keep global indices

    return layer_to_edges



def plot_curve(high_clean_acc, low_clean_acc, remove_num, res_path, name):
    # Colors
    high_color = "#29E000" 
    low_color = "#1F00EC"   

    # Sort by remove_num
    combined = sorted(zip(remove_num, high_clean_acc, low_clean_acc), key=lambda x: x[0])
    remove_sorted, high_sorted, low_sorted = zip(*combined)

    plt.figure(figsize=(10, 6))

    # Plot lines
    plt.plot(remove_sorted, high_sorted, label='Large weight removed first',
             marker='o', linestyle='--', linewidth=3., markersize=13, color=high_color)

    plt.plot(remove_sorted, low_sorted, label='Small weight removed first',
             marker='x', linestyle='-', linewidth=3., markersize=13, color=low_color)

    # Labels and title
    plt.xlabel('Number of Edges Removed', fontsize=33, fontweight='semibold')
    plt.ylabel('Accuracy', fontsize=33, fontweight='semibold')
    plt.ylim(-0.1, 1.2)  # Actual data limits
    plt.yticks(np.linspace(0.0, 1.0, num=6))  # Only show ticks from 0 to 1

    # Scientific x-axis
    ax = plt.gca()
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.xaxis.get_offset_text().set_fontsize(20)
    ax.xaxis.get_offset_text().set_fontweight('semibold')

    # Ticks
    plt.xticks(fontsize=22, fontweight='semibold')
    plt.yticks(fontsize=22, fontweight='semibold')

    # Grid and legend
    plt.grid(True, linestyle='--', linewidth=2.5, color='gray', alpha=0.85)
    legend = plt.legend(fontsize=22, loc='best')
    for text in legend.get_texts():
        text.set_fontweight('semibold')

    plt.tight_layout()
    plt.savefig(os.path.join(res_path, f'{name}_remove_w_curve.pdf'), dpi=300)
    plt.close()




def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def remove_w_lidar(args):
    seed = 59
    set_seed(seed)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    
    model_type = args.model_type
    sz = args.model_name
    res_path = args.mnist_res_path
    model_path = args.model_path
    
    dims = model_zoo[sz]

    if not os.path.exists(res_path):
        os.makedirs(res_path)
    
    with open("Lidar/trajectory.yml", 'r') as f:
        lidar_traj_by_controller = yaml.full_load(f)
    
    # build model
    for s in series:
        print(f'Now is for {model_type}-{sz} - {str(s)}')
        model = Controller(dims, 2)
        # name = model_type + '_L21_' + sz + 'x' + sz + '_C' + str(s) + '.yml'
        # model_name = model_type + "_" + sz + "_C" + str(s) + ".pth"
        # model = load_yaml_weights_to_model(model, data_path+name)
        model = model.double()
        model = model.to(device)
        model_name = model_type + "_" + sz + "_C" + str(s) + ".pth"
        model.load_state_dict(torch.load(model_path + model_name))
        
        prefix_dims = np.cumsum([0] + dims).tolist()

        net_full = copy.deepcopy(model)
        model = model.to(device)
        crashes = test(net_full, device)

        name = '_'.join([model_type,sz,str(s)])
        traj_l = lidar_traj_by_controller[name]
        traj = np.array(traj_l[0])
        input = np.array(traj[0], dtype=np.float64)
        input = input.reshape(1, -1)
        input = torch.tensor(input, dtype=torch.float64)
        edge_array, nodes_ori, output, all_node = model.NN_info_batch(input.to(device)) # [21,64,64,1]
        
        weights = output.detach().clone().to(device) 
        sorted_edges_high, sorted_edges_low = get_all_edges_sorted_by_weight(dims, weights, device) 
        
        # Step 1: Convert tensors to numpy BEFORE separating by layer
        sorted_edges_high_np = sorted_edges_high.cpu().numpy()
        sorted_edges_low_np = sorted_edges_low.cpu().numpy()
        
        # # Step 2: Define total number of edges and removal schedule
        # neg_total = len(sorted_edges_low_np)
        # pos_total = len(sorted_edges_high_np)

        # low_remove_num = list(np.linspace(0, neg_total, num=10, dtype=int))
        # high_remove_num = list(np.linspace(0, pos_total, num=10, dtype=int))
        
        # high_acc_clean = []
        # low_acc_clean = []
        
        # for index, rem_f in enumerate(high_remove_num):
        #     cur_n = model_type + '_' + str(s) + '_' + str(rem_f)

        #     # remove high weight edges
        #     model.load_state_dict(torch.load(model_path + model_name))
        #     edge_r = Edge_Remove(model, dims, min(rem_f, len(sorted_edges_high_np)), res_path)
        #     edge_r.e_remove(sorted_edges_high_np, cur_n + "other_neg.pth")
            
        #     # test acc
        #     net_high = Controller(dims, 2)
        #     net_high = net_high.double()
        #     net_high.load_state_dict(torch.load(res_path + cur_n + "other_neg.pth"))
        #     net_high = net_high.to(device)
        #     os.remove(res_path + cur_n + "other_neg.pth")

        #     num_high = test(net_high, device)
        #     acc_high= (1000-num_high)/1000
        #     high_acc_clean.append(acc_high)
        
        # for index, rem_f in enumerate(low_remove_num):
        #     # remove low weight edges
        #     model.load_state_dict(torch.load(model_path + model_name))
        #     edge_r = Edge_Remove(model, dims, min(rem_f, len(sorted_edges_low_np)), res_path)
        #     edge_r.e_remove(sorted_edges_low_np, cur_n + "pos.pth")
            
        #     # test acc
        #     net_low = Controller(dims, 2)
        #     net_low = net_low.double()
        #     net_low.load_state_dict(torch.load(res_path + cur_n + "pos.pth"))
        #     net_low = net_low.to(device)
        #     os.remove(res_path + cur_n + "pos.pth")

        #     num_low = test(net_low, device)
        #     acc_lows = (1000-num_low)/1000
        #     low_acc_clean.append(acc_lows)
            
        # plot_curve(high_acc_clean, low_acc_clean, low_remove_num, res_path, name)                


        # Step 2: Separate edges by layer
        sorted_edges_high_by_layer = separate_edges_by_layer_in_order(sorted_edges_high_np, prefix_dims)
        sorted_edges_low_by_layer = separate_edges_by_layer_in_order(sorted_edges_low_np, prefix_dims)
        print(sorted_edges_low_by_layer.keys())
        
        # Step 3: Per-layer analysis
        for layer in sorted(sorted_edges_low_by_layer.keys() | sorted_edges_high_by_layer.keys()):
            high_acc_clean = []
            low_acc_clean = []
            
            # These are just lists of (i, j), not 4-tuples
            neg_edges = sorted_edges_low_by_layer.get(layer, [])
            pos_edges = sorted_edges_high_by_layer.get(layer, [])
            
            neg_total = len(neg_edges)
            pos_total = len(pos_edges)
            
            low_remove_num = list(np.linspace(0, neg_total, num=3, dtype=int))
            high_remove_num = list(np.linspace(0, pos_total, num=6, dtype=int))

            print(f"Layer {layer}:")
            print(f"  Low remove nums: {low_remove_num}")
            print(f"  High remove nums: {high_remove_num}")
        
            # start remove
            for index, rem_f in enumerate(high_remove_num):
                cur_n = model_type + '_' + str(s) + '_' + str(rem_f)

                # remove high weight edges
                model.load_state_dict(torch.load(model_path + model_name))
                edge_r = Edge_Remove(model, dims, min(rem_f, len(pos_edges)), res_path)
                edge_r.e_remove(pos_edges, cur_n + "other_neg.pth")
                
                # test acc
                net_high = Controller(dims, 2)
                net_high = net_high.double()
                net_high.load_state_dict(torch.load(res_path + cur_n + "other_neg.pth"))
                net_high = net_high.to(device)
                os.remove(res_path + cur_n + "other_neg.pth")

                num_high = test(net_high, device)
                acc_high= (1000-num_high)/1000
                high_acc_clean.append(acc_high)
            
            for index, rem_f in enumerate(low_remove_num):
                # remove low weight edges
                model.load_state_dict(torch.load(model_path + model_name))
                edge_r = Edge_Remove(model, dims, min(rem_f, len(neg_edges)), res_path)
                edge_r.e_remove(neg_edges, cur_n + "pos.pth")
                
                # test acc
                net_low = Controller(dims, 2)
                net_low = net_low.double()
                net_low.load_state_dict(torch.load(res_path + cur_n + "pos.pth"))
                net_low = net_low.to(device)
                os.remove(res_path + cur_n + "pos.pth")

                num_low = test(net_low, device)
                acc_lows = (1000-num_low)/1000
                low_acc_clean.append(acc_lows)
                    
            plot_curve(high_acc_clean, low_acc_clean, low_remove_num, res_path, name + '_' + str(layer))                  
                            
    
    
            