import numpy as np
import random
import matplotlib.pyplot as plt
import sys
import yaml
import torch
import torch.nn as nn
import os
import pickle
import copy
from collections import defaultdict
from collections import Counter
import matplotlib.pyplot as plt

from .Car import World

import sys
sys.path.append("..")

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


def plot_curve(
    neg_acc_clean, pos_acc_clean,
    neg_freq_ratios, pos_freq_ratios,
    neg_freq_thresholds, pos_freq_thresholds,
    label, save_path
):
    # CMYK-like colors (safe RGB approximations)
    neg_colors = ['#00A3E0', '#6CACE4']  # Cyan, Blue-gray
    pos_colors = ['#EC008C', '#FF6F61']  # Magenta, Warm red

    plt.figure(figsize=(10, 6))

    # Negative Edge Plot
    plt.plot(
        neg_freq_ratios, neg_acc_clean,
        label='Negative Edge Removal',
        marker='o',
        linestyle='--',
        linewidth=2,
        markersize=6,
        color=neg_colors[0]
    )

    # Positive Edge Plot
    plt.plot(
        pos_freq_ratios, pos_acc_clean,
        label='Positive Edge Removal',
        marker='s',
        linestyle='-',
        linewidth=2,
        markersize=6,
        color=pos_colors[0]
    )

    # Annotate frequencies
    for x, y, freq in zip(neg_freq_ratios, neg_acc_clean, neg_freq_thresholds):
        plt.annotate(
            f"{freq}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=9,
            color=neg_colors[1]
        )

    for x, y, freq in zip(pos_freq_ratios, pos_acc_clean, pos_freq_thresholds):
        plt.annotate(
            f"{freq}",
            (x, y),
            textcoords="offset points",
            xytext=(0, -15),
            ha='center',
            fontsize=9,
            color=pos_colors[1]
        )

    # Axes and title
    plt.xlabel("Edge Frequency Threshold (ratio × max frequency)", fontsize=22)
    plt.ylabel("Clean Accuracy", fontsize=22)
    plt.title(f"Clean Accuracy vs. Frequency Ratio (Label {label})", fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(0.0, 1.0)
    plt.gca().invert_xaxis()

    # Legend and grid
    plt.legend(fontsize=20, loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Save
    filename = os.path.join(save_path, f'{label}_freq_curve.png')
    plt.savefig(filename, dpi=300)
    plt.close()
    
    
    
# def plot_curve(neg_clean_acc, pos_clean_acc, neg_remove_num, pos_remove_num, neg_end, pos_end, label, res_path):
#     # CMYK-like colors (manually mapped to RGB approximations)
#     # Separate CMYK-safe colors for NEG and POS
#     neg_colors = ['#00A3E0', '#6CACE4', '#00AB84', '#9E1B32']  # Cyan, Blue-gray, Greenish cyan, Dark red
#     pos_colors = ['#EC008C', '#FF6F61', '#FEDD00', '#000000']  # Magenta, Warm red, Yellow, Black
    
#     # Plot
#     plt.figure(figsize=(10, 6))  # Slightly wider for spacing

#     plt.plot(
#         neg_remove_num, neg_clean_acc,
#         label='Negative Edge Accuracy',
#         marker='o',
#         linestyle='--',
#         linewidth=2,
#         markersize=6,
#         color=neg_colors[0]
#     )

#     plt.plot(
#         pos_remove_num, pos_clean_acc,
#         label='Positive Edge Accuracy',
#         marker='s',
#         linestyle='-',
#         linewidth=2,
#         markersize=6,
#         color=pos_colors[0]
#     )

#     # Vertical lines for termination points
#     plt.axvline(
#         x=neg_end, color=neg_colors[1], linestyle=':', linewidth=2,
#         label=f'Neg Stop @ {neg_end}'
#     )
#     plt.axvline(
#         x=pos_end, color=pos_colors[1], linestyle=':', linewidth=2,
#         label=f'Pos Stop @ {pos_end}'
#     )

#     # Axes and title
#     plt.xlabel('Number of Edges Removed', fontsize=23)
#     plt.ylabel('Clean Accuracy', fontsize=23)
#     plt.title('Clean Accuracy vs. Edge Removal Count', fontsize=23)
#     plt.ylim(0.0, 1.0)
#     plt.xticks(fontsize=22)
#     plt.yticks(fontsize=22)

#     plt.legend(fontsize=22, loc='best')
#     plt.grid(True, linestyle='--', alpha=0.6)
#     plt.tight_layout()

#     # Save
#     plt.savefig(res_path + f'{label}_curve_all.png', dpi=300)
#     plt.close()
  
    
def load_yaml_weights_to_model(model, yaml_path: str):
    # Load YAML file
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    weights = data["weights"]
    offsets = data["offsets"]

    # Convert model to double precision
    model = model.double()

    # Get model layers
    layers = [m for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d))]

    for i, layer in enumerate(layers):
        key = i + 1
        if key in weights:
            w = torch.tensor(weights[key], dtype=torch.double)
            if isinstance(layer, nn.Linear):
                w = w.view(layer.out_features, layer.in_features)

            layer.weight.data = w

        if key in offsets:
            b = torch.tensor(offsets[key], dtype=torch.double)
            layer.bias.data = b

    return model

    

def normalize(s):
    mean = [2.5]
    spread = [5.0]
    return (s - mean) / spread


def get_top_c(curvature, b, prefix_dims):
    neg_e = set()
    pos_e = set()
    
    for batch in range(b):
        ricci_curv = np.array(curvature[batch])  # shape (N, 3)

        # Filter values with valid curvature (<= 1)
        valid = ricci_curv[ricci_curv[:, 2] <= 1]

        # Convert to int for indexing
        valid[:, 0:2] = valid[:, 0:2].astype(int)

        for i, j, curr in valid:
            i, j = int(i), int(j)
            if curr < 0:
                neg_e.add((i, j, curr))
            elif curr > 0:
                pos_e.add((i, j, curr))

    return neg_e, pos_e


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


def count_edge_frequency(edge_sets):
    freq = Counter()
    curvature_sum = defaultdict(float)

    for edge_set in edge_sets:
        for i, j, c in edge_set:
            # Normalize undirected edge direction efficiently
            key = (min(i, j), max(i, j))
            freq[key] += 1
            curvature_sum[key] += c

    # Use list comprehension for speed and clarity
    results = [
        (i, j, count, curvature_sum[(i, j)] / count)
        for (i, j), count in freq.items()
    ]

    return results


def main_lidar(args):
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
    
    model_type = args.model_type
    sz = args.model_name
    res_path = args.mnist_res_path
    model_path = args.model_path
    metric = args.metric
    sample_size = args.sample_num
    data_path = args.mnist_data_path
    
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    
    dims = model_zoo[sz]
    prefix_dims = np.cumsum([0] + dims).tolist()
    
    freq_ratios = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]
    
    for s in series:
        print(f'Now is for {model_type}-{sz} - {s}')
        model = Controller(dims, 2)
        # name = model_type + '_L21_' + sz + 'x' + sz + '_C' + str(s) + '.yml'
        # model_name = model_type + "_" + sz + "_C" + str(s) + ".pth"
        # model = load_yaml_weights_to_model(model, data_path+name)
        model = model.double()
        model = model.to(device)
        model_name = model_type + "_" + sz + "_C" + str(s) + ".pth"
        model.load_state_dict(torch.load(model_path + model_name))

        net_full = copy.deepcopy(model)
        model = model.to(device)
        crashes = test(net_full, device)
        
        # torch.save(model.state_dict(), model_path+model_name)

        print(f'clean model crashes: {crashes}')
   
        res_name = metric + model_type + '_' + sz + '_' + str(s) + '_res.pkl'
        with open(data_path + res_name, 'rb') as file:
            res_dict = pickle.load(file)
            
        print(f'Finish read pickle file..')
            
        neg_acc_clean = []
        pos_acc_clean = []
        neg_edge_sets = []
        pos_edge_sets = []
        for l in range(20):
            idx = 0
            for ricci in res_dict[l]:
                neg_e, pos_e = get_top_c(ricci, 1, prefix_dims)
                neg_edge_sets.append(neg_e)
                pos_edge_sets.append(pos_e)
                
                idx += 1
                if idx >= sample_size:
                    break
                
        neg_freq_dict = count_edge_frequency(neg_edge_sets)
        pos_freq_dict = count_edge_frequency(pos_edge_sets)
        neg_freq_edges_sorted = sorted(neg_freq_dict, key=lambda x: (-x[2], x[3]))
        pos_freq_edges_sorted = sorted(pos_freq_dict, key=lambda x: (-x[2], -x[3]))

        print(f'It has {len(neg_freq_edges_sorted)} negative curvature edges, {len(pos_freq_edges_sorted)} positive curvature egdes .. \n')
            
        neg_edges_only = [(i, j) for (i, j, _, _) in neg_freq_edges_sorted]
        pos_edges_only = [(i, j) for (i, j, _, _) in pos_freq_edges_sorted]

        neg_total = len(neg_edges_only)
        pos_total = len(pos_edges_only)

        # Generate uniformly spaced points (including 0 and total) for each list
        # neg_remove_num = list(np.linspace(0, neg_total, num=10, dtype=int))
        # pos_remove_num = list(np.linspace(0, pos_total, num=20, dtype=int))
        
        # neg_remove_num = [0, 50, 70, 100, 110, 120, 125, 130, 135, 140, 150, 160, 180, 200, 250, 300, 500, 6000]
        # pos_remove_num = neg_remove_num
        
        # Step 2: Choose thresholds — you can just use them all or downsample if too many
        neg_max_freq = max(freq for (_, _, freq, _) in neg_freq_edges_sorted)
        neg_freq_thresholds = [int(r * neg_max_freq) for r in freq_ratios]

        pos_max_freq = max(freq for (_, _, freq, _) in pos_freq_edges_sorted)
        pos_freq_thresholds = [int(r * pos_max_freq) for r in freq_ratios]
        
        # Step 3: For each threshold, count how many edges would be removed
        neg_remove_num = [sum(1 for (_, _, freq, _) in neg_freq_edges_sorted if freq >= t) for t in neg_freq_thresholds]
        pos_remove_num = [sum(1 for (_, _, freq, _) in pos_freq_edges_sorted if freq >= t) for t in pos_freq_thresholds]
        
        # start remove
        for index, rem_f in enumerate(neg_remove_num):
            print(f'Remove neg edge number {rem_f}:')
            cur_n = model_type + '_' + sz + '_' + str(s) + str(rem_f) + '_' + str(l)

            # remove negative curvature edges
            model.load_state_dict(torch.load(model_path + model_name))
            edge_r = Edge_Remove(model, dims, min(rem_f, len(neg_edges_only)), res_path)
            edge_r.e_remove(neg_edges_only, cur_n + "_neg.pth")
            
            # test acc
            net_neg = Controller(dims, 2)
            net_neg = net_neg.double()
            net_neg.load_state_dict(torch.load(res_path + cur_n + "_neg.pth"))
            net_neg = net_neg.to(device)
            os.remove(res_path + cur_n + "_neg.pth")
            
            num_unsafe_neg = test(net_neg, device)
            acc_neg = (1000-num_unsafe_neg)/1000
            neg_acc_clean.append(acc_neg)
            
        for index, rem_f in enumerate(pos_remove_num):
            print(f'Remove pos edge number {rem_f}:')
            # ff.write(f'Remove edge number {rem_f}: \n')
            cur_n = model_type + '_' + sz + '_' + str(s) + str(rem_f) + '_' + str(l)
            # remove positive curvature edges
            model.load_state_dict(torch.load(model_path + model_name))
            edge_r = Edge_Remove(model, dims, min(rem_f, len(pos_edges_only)), res_path)
            edge_r.e_remove(pos_edges_only, cur_n + "pos.pth")
            
            # test acc
            net_pos = Controller(dims, 2)
            net_pos = net_pos.double()
            net_pos.load_state_dict(torch.load(res_path + cur_n + "pos.pth"))
            net_pos = net_pos.to(device)
            os.remove(res_path + cur_n + "pos.pth")

            num_unsafe_pos = test(net_pos, device)
            acc_pos = (1000-num_unsafe_pos)/1000
            pos_acc_clean.append(acc_pos)
        
        # plot_curve(neg_acc_clean, pos_acc_clean, neg_remove_num, pos_remove_num, neg_total, pos_total, s, res_path)
        plot_curve(neg_acc_clean, pos_acc_clean, freq_ratios, freq_ratios, neg_remove_num, pos_remove_num, str(sample_size) + '_' + str(s), res_path)
    
        
        
        
        
        
        
        
        
