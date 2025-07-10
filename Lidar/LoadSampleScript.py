# load targets and input data

# compute control error on the training controllers and data
# compute control error on the holdout validation controllers
# to get confidence intervals


# to get confidence intervals for each of the rays, 
# compute outputs on validation set (sigmoids)

import yaml
import torch
import numpy as np
import os
import random
import pickle
from collections import defaultdict
from Lidar.lidar_tools.getNNweights import getNN_info
from Lidar.lidar_tools.graph_cal import *
from Lidar.lidar_tools.graph_curvature import graph_curvature_main_torch
from tools.controller import Controller

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

Keys =  ['DDPG_128_1', 'DDPG_128_2', 'DDPG_128_3', 'DDPG_64_1', 'DDPG_64_2', 'DDPG_64_3', 'TD3_128_1', 'TD3_128_2', 'TD3_128_3', 'TD3_64_1', 'TD3_64_2', 'TD3_64_3']

model_zoo = {
    '64': [21, 64, 64, 1],
    '128': [21, 128, 128, 1]
}

series = [1,2,3]


def normalize(s):
    mean = [2.5]
    spread = [5.0]
    return (s - mean) / spread


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


def load_all_controllers(controller_dir = 'Lidar/controllers/'):
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



def start_lidar(args):
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
    
    # controllers = load_all_controllers()
    
    res_path = args.lidar_res_path
    metric = args.metric
    alpha = args.alpha
    model_path = args.model_path
    sample_size = args.sample_num
    
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    
    types = ['DDPG', 'TD3']
    cs = ['1','2','3']
    szs = ['64','128']
    
    num_samples = sample_size
    
    with open("Lidar/trajectory.yml", 'r') as f:
        lidar_traj_by_controller = yaml.full_load(f)
    
    keys = lidar_traj_by_controller.keys()
    print("Keys = ", keys)
    
    for t in types:
        for s in szs:
            dims = model_zoo[s]
            for c in cs:
                res_l = defaultdict(list)
    
                name = '_'.join([t,s,c])
                # cur_c = controllers[name]
                
                model = Controller(dims, 2)
                model = model.double()
                model_name = t + "_" + s + "_C" + str(c) + ".pth"
                model.load_state_dict(torch.load(model_path + model_name))
                cur_c = model.to(device)
                
                print(f'Controller: {name}')
                
                traj_l = lidar_traj_by_controller[name]

                # all the trajectories for one controller
                for i in range(min(20,len(traj_l))):
                    traj = np.array(traj_l[i]) # 219 * 21
                    sampled_inputs = traj[np.random.randint(traj.shape[0], size=num_samples), :] # sample * 21

                    for index, input in enumerate(sampled_inputs):
                        input = np.array(input, dtype=np.float64)
                        input = input.reshape(1, -1)
                        input = normalize(input)
                        input = torch.tensor(input, dtype=torch.float64)
                        edge_array, nodes_ori, output, all_node = cur_c.NN_info_batch(input.to(device)) # [21,64,64,1]

                        weights = output.detach().to(device)
                        del output
                        # weights[edge_array == 0] = 0.
                        
                        if metric.lower() == "w1":
                            weights_inv1, weights_inv2 = normalization_weight_w1(nodes_ori, weights, dims)
                            weights_inv = weights_inv1.detach()
                            weights_inv2 = weights_inv2.detach()
                            ricci_curvature = graph_curvature_main_torch(
                                dims, weights_inv, device=device,
                                probability_w=weights_inv2, alpha=alpha
                            )
                        elif metric.lower() == "w3":
                            weights_inv1, weights_inv2 = normalization_weight_w3(nodes_ori, weights, dims)
                            weights_inv = weights_inv1.detach()
                            weights_inv2 = weights_inv2.detach()
                            ricci_curvature = graph_curvature_main_torch(
                                dims, weights_inv, device=device,
                                probability_w=weights_inv2, alpha=alpha
                            )
                        else:
                            raise Exception("Invalid graph metric, should be {w1, w3}!")
                        
                        res_l[i].append(ricci_curvature)
                        
                        # GPU memory cleanup
                        del input, edge_array, all_node
                        del weights, weights_inv1, weights_inv2, weights_inv
                        torch.cuda.empty_cache()
                 
                    print(f'Finish {i} trajectory...')

                with open(res_path + metric + name + "_res.pkl", 'wb') as file:
                    pickle.dump(res_l, file)
                    

 
 
        
