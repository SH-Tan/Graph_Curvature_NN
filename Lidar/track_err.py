# load targets and input data

# to get confidence intervals for each of the rays, 
# compute outputs on validation set (sigmoids)

import yaml
import torch
import numpy as np
import os
import random
import pickle
from collections import defaultdict
from scipy.interpolate import interp1d
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import itertools
import matplotlib.pyplot as plt
# from lidar_tools.getNNweights import getNN_info
# from lidar_tools.graph_cal import *
# from lidar_tools.graph_curvature_multihops import graph_curvature_main_torch

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

Keys =  ['DDPG_128_1', 'DDPG_128_2', 'DDPG_128_3', 'DDPG_64_1', 'DDPG_64_2', 'DDPG_64_3', 'TD3_128_1', 'TD3_128_2', 'TD3_128_3', 'TD3_64_1', 'TD3_64_2', 'TD3_64_3']


def custom_predict_yaml(model, inputs):
    weights = {}
    offsets = {}

    layerCount = 1
    activations = []

    for layer in range(1, len(model['weights']) + 1):
        
        weights[layer] = np.array(model['weights'][layer])
        offsets[layer] = np.array(model['offsets'][layer])

        layerCount += 1
        activations.append(model['activations'][layer])

    curNeurons = inputs

    for layer in range(layerCount-1):

        curNeurons = curNeurons.dot(weights[layer + 1].T) + offsets[layer + 1]

        if 'Sigmoid' in activations[layer]:
            curNeurons = 1/(1 + np.exp(-curNeurons))
        elif 'Tanh' in activations[layer]:
            curNeurons = np.tanh(curNeurons)

    return np.array(curNeurons)


def vectorize_predict_yaml(model, inputs, temps=1):
    weights = {}
    offsets = {}

    layerCount = 1
    activations = []
    

    for layer in range(1, len(model['weights']) + 1):
        
        weights[layer] = np.array(model['weights'][layer])
        offsets[layer] = np.array(model['offsets'][layer])

        layerCount += 1
        activations.append(model['activations'][layer])

    curNeurons = inputs.T
    
    print('Layer num: ', layerCount)

    for layer in range(layerCount-1):

        curNeurons = weights[layer+1]@curNeurons + offsets[layer+1].reshape(len(offsets[layer+1]),1)
        if 'Sigmoid' in activations[layer]:
            curNeurons = 1/(1 + np.exp(-curNeurons))
        elif 'Tanh' in activations[layer]:
            curNeurons = np.tanh(curNeurons)

    return np.array(curNeurons).T


def load_all_controllers(controller_dir = './controllers/'):
    # load a dictionary of all other controller?
    types = ['DDPG', 'TD3']
    cs = ['1','2','3']
    szs = ['64','128']
    controllers = {}
    for t in types:
        for s in szs:
            for c in cs:
                controllers_filename = controller_dir+t+'_L21_'+s+'x'+s+'_C'+c+'.yml'
                with open(controllers_filename, 'rb') as f:
                    controllers['_'.join([t,s,c])] = yaml.full_load(f)

    print("controller keys: ", list(controllers.keys()))
    return controllers



# def get_fraction(curvature, b):
#     c = []
#     neg = []
#     total_e = []
#     for i in range(b):
#         curr = np.array(curvature[i])
#         neg.append(len(curr[curr<0]))
#         total_e.append(len(curr))
#     return np.array(neg), np.array(total_e), curr


def get_fraction(curvature, b, dims):
    c = []
    layer_num = len(dims) - 1
    neg = np.zeros((layer_num), dtype=np.float32)
    top_neg = np.zeros((layer_num), dtype=np.float32)
    total_e = np.zeros((layer_num), dtype=np.float32)
    # neg = 0.
    # total_e = 0.
    
    for batch in range(b):
        ricci_curv = np.array(curvature[batch])
        for (i, j, curr) in ricci_curv:
            if curr > 1:
                continue
    
            l = int(i)
            if curr < 0:
                neg[l] += 1
            if curr < -10:
                top_neg[l] += 1
            total_e[l] += 1
            c.append(curr)
    return neg, total_e, top_neg, c




def calculate_normalized_pointwise_error(traj1, traj2, time1=None, time2=None, num_points=None, interp_kind='linear'):
    n1, state_dim1 = traj1.shape
    n2, state_dim2 = traj2.shape

    if time1 is None:
        time1 = np.linspace(0, 1, n1)
    if time2 is None:
        time2 = np.linspace(0, 1, n2)

    start_time = max(np.min(time1), np.min(time2))
    end_time = min(np.max(time1), np.max(time2))

    if num_points is None:
        num_points = max(n1, n2)

    common_time = np.linspace(start_time, end_time, num_points)
    interpolated_traj1 = np.zeros((num_points, state_dim1))
    interpolated_traj2 = np.zeros((num_points, state_dim2))

    for i in range(state_dim1):
        interp_func1 = interp1d(time1, traj1[:, i], kind=interp_kind, fill_value="extrapolate")
        interpolated_traj1[:, i] = interp_func1(common_time)

        interp_func2 = interp1d(time2, traj2[:, i], kind=interp_kind, fill_value="extrapolate")
        interpolated_traj2[:, i] = interp_func2(common_time)

    pointwise_error = np.linalg.norm(interpolated_traj1 - interpolated_traj2, axis=1, ord=2)

    # Calculate the squared Euclidean distance between the initial points
    initial_distance_sq = np.linalg.norm(traj1[0] - traj2[0], ord=2)

    if initial_distance_sq > 1e-9:  # Avoid division by zero for very close initial points
        normalized_error = pointwise_error / initial_distance_sq
    else:
        normalized_error = np.zeros_like(pointwise_error) # Or handle as needed

    return common_time, normalized_error


def calculate_tracking_error_dtw(traj1, traj2):
    # Convert lists to numpy arrays if needed
    traj1 = np.array(traj1)
    traj2 = np.array(traj2)

    # Define the distance metric (Euclidean distance)
    dist = lambda x, y: np.linalg.norm(x - y, ord=2)

    # Perform DTW alignment
    distance, path = fastdtw(traj1, traj2, dist=dist)

    # Extract the warped indices
    warped_time1 = np.array([p[0] for p in path])
    warped_time2 = np.array([p[1] for p in path])
    
    initial_error = np.linalg.norm(traj1[0] - traj2[0], ord=2)

    # Calculate the point-wise error for the aligned points
    pointwise_error = np.linalg.norm(traj1[warped_time1] - traj2[warped_time2], axis=1)
    normalized_error = pointwise_error / initial_error
    return normalized_error


if __name__=='__main__':
    seed = 59
    
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
    
    controllers = load_all_controllers()
    
    # res_path = args.lidar_res_path
    # metric = args.metric
    # alpha = args.alpha
    # hops = args.hops
    
    # if not os.path.exists(res_path):
    #     os.makedirs(res_path)
    
    
    with open("./lidar_trajectories_by_controller.yml", 'r') as f:
        lidar_traj_by_controller = yaml.full_load(f)
    
    keys = lidar_traj_by_controller.keys()
    print("Keys = ", keys)
  
  
    types = ['DDPG', 'TD3']
    cs = ['1','2','3']
    szs = ['64','128']
    
    num_samples = 100
    results = {}

    for t in types:
        for s in szs:
            for c in cs:
                name = '_'.join([t,s,c])
                cur_c = controllers[name]
                
                print(f'Controller: {name}')
                
                traj_l = lidar_traj_by_controller[name]
                num_trajectories = len(traj_l)
                
                count = 0
                results[name] = []
                
                pair_count = 0
                num_pairs = num_trajectories * (num_trajectories - 1) // 2
                fig, axes = plt.subplots(num_pairs, 1, figsize=(10, 6 * num_pairs), sharex=True)
                
                for i, j in itertools.combinations(range(num_trajectories), 2):
                    traj1 = np.array(traj_l[i])
                    traj2 = np.array(traj_l[j])
                    # print(f'  Comparing trajectory {i+1} (length: {traj1.shape}) with trajectory {j+1} (length: {traj2.shape})')

                    # --- Calculate normalized point-wise error using interpolation ---
                    pointwise_error = calculate_tracking_error_dtw(traj1, traj2)

                    results[name].append(np.mean(pointwise_error))
                    count += 1
                    
                #     ax = axes[pair_count]
                #     ax.plot(range(len(pointwise_error)), pointwise_error, label=f'Pair {i+1} vs {j+1}')
                #     ax.set_title(f'Controller: {name} - Normalized Error (Pair {i+1} vs {j+1})')
                #     ax.set_xlabel('Aligned Time Step')
                #     ax.set_ylabel('L2 Error')
                #     ax.grid(True)
                #     ax.legend()
                #     pair_count += 1

                # plt.tight_layout()
                # plt.suptitle(f'Normalized Point-wise Error for Each Trajectory Pair - Controller: {name}', y=1.02)
                # filename = f"controller_{name.replace('_', '-')}_error_plots.png" # Replace underscores for filename safety
                # filepath = os.path.join('./img_syn_norm/', filename)
                # plt.savefig(filepath)
                # plt.close(fig) # Close the figure to free up memory
                                
                # results[name] = results[name]/count if count > 0 else results[name]
                print(f'The average error is {np.mean(results[name])}')
                
    # print(results)