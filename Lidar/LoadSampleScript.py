# load targets and input data

# compute control error on the training controllers and data
# compute control error on the holdout validation controllers
# to get confidence intervals


# get 

# to get confidence intervals for each of the rays, 
# compute outputs on validation set (sigmoids)

import yaml
import torch
import numpy as np
import os
import random
import pickle
from getNNweights import getNN_info
from graph_cal import *

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

Keys =  ['DDPG_128_1', 'DDPG_128_2', 'DDPG_128_3', 'DDPG_64_1', 'DDPG_64_2', 'DDPG_64_3', 'TD3_128_1', 'TD3_128_2', 'TD3_128_3', 'TD3_64_1', 'TD3_64_2', 'TD3_64_3']


metric = "w7"
res_path = "res/" + metric + "/"
# data_path = "res/w6_c/"


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


def load_all_controllers(controller_dir = 'controllers/'):
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

def load_all_data(data_filename = "cleandata_by_controller.yml"):
    with open(data_filename, 'rb') as f:
        data_dict = yaml.full_load(f)
    print("data key: ", data_dict.keys())
    return data_dict


if __name__ == "__main__":
    seed = 59
    
    # set random seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    controllers = load_all_controllers()
    
    #data_dict = load_all_data() Thise contains a lot for info if you need it for someting later, but it takes a while to load.
    #print(len(data_dict["Real"]))

    # this loads the basic 21 lidar scans from the real sensor data. It has been normalized between -0.5 and 0.5. 
    # all_inputs = np.load("Fixed_Noisy_lidar_uncovered.npy")
    # print("all input size: ", all_inputs.shape)

    # # # # all_outputs = vectorize_predict_yaml(controllers["DDPG_64_1"], all_inputs)
    # # # # print("all output size", all_outputs.shape)

    # num_samples = 200
    # sampled_inputs = all_inputs[np.random.randint(all_inputs.shape[0], size=num_samples), :]
    # print("sample input size: ", sampled_inputs.shape)
    
     # print("prints the shape of each trajectory: (number of datapoint, size of lidar) ")
    # # the below list contains a numpy array for each trajectory for the given controller
    # example_lidar_trajectories = [np.array(traj) for traj in lidar_traj_by_controller["DDPG_64_1"]]
    # for traj in example_lidar_trajectories:
    #     print(traj.shape)
    
    
    with open("lidar_trajectories_by_controller.yml", 'r') as f:
        lidar_traj_by_controller = yaml.full_load(f)
    
    keys = lidar_traj_by_controller.keys()
    print("Keys = ", keys)
    
    # traj_data = defaultdict(list)
    # for k in keys:
    #     index = np.random.randint(len(lidar_traj_by_controller[k]), size=4)
    #     example = lidar_traj_by_controller[k][index]
    #     traj_data[k].append(np.array(example))
    #     print(traj_data[k][0].shape)
    
    
    # with open("lidar_trajectories_by_controller.yml", 'r') as f:
    #     lidar_traj_by_controller = yaml.full_load(f)
        
    # with open("crashes_by_controller.yml", 'r') as f:
    #     crashes_by_controller = yaml.full_load(f)

    
    c_l = defaultdict(list)
    f_l = defaultdict(list)

    
    types = ['DDPG', 'TD3']
    cs = ['1','2','3']
    szs = ['64', '128']
    
    
    # for controller_name in controllers.keys():
    #     example_lidar_trajectories = [np.array(traj) for traj in lidar_traj_by_controller[controller_name]]
    #     crashes =  [bool(crash) for crash in crashes_by_controller[controller_name]]
    #     print(controller_name + "  Crashes: " + str(sum(crashes)) + " Trajectories: " + str(len(crashes)))
    #     for traj, crash in zip(example_lidar_trajectories, crashes):
    #         print("\t", end="")
    #         print(traj.shape, crash)
    
    sample = 5
    
    for t in types:
        for s in szs:
            for c in cs:
                c_l = defaultdict(list)
                f_l = defaultdict(list)
    
                name = '_'.join([t,s,c])
                cur_c = controllers[name]
                
                print(f'Controller: {name}')
                
                # crash_d = crashes_by_controller[name]
                # traj_d = lidar_traj_by_controller[name]
                
                # assert(len(crash_d) == len(traj_d))
                
                # safe_traj = []
                # unsafe_traj = []
                # for i in range(len(crash_d)):
                #     if (crash_d[i] == 1):
                #         unsafe_traj.append(np.array(traj_d[i]))
                #     else:
                #         safe_traj.append(np.array(traj_d[i]))
                
                # print(f'Safe: {len(safe_traj)}, Unsafe: {len(unsafe_traj)}')
                # if (len(safe_traj) == 0 or len(unsafe_traj) == 0):
                #     continue
                
                traj_l = lidar_traj_by_controller[name]

                for i in range(sample, 8):
                    traj = traj_l[i]
                    tmp_f = []
                    tmp_c = []
                    
                    for index, input in enumerate(traj):
                        input = np.array(input)
                        input = input.reshape(1, -1)
                        input = torch.tensor(input)
                        dims, nodes_num, edges_num, NN_w, edge_v, nodes, nodes_ori = getNN_info(cur_c, input)
                        
                        
                        # build graph
                        adj = build_adjm(nodes_num, dims, edge_v, nodes, NN_w)
                        adj = adj.cpu().detach().numpy()
                        curv, ratio, G = cal_curvature(adj, nodes, dims, nodes_ori.cpu())
                        # print(ratio)
                        # with open("./graph_w.txt", "a+") as ff:
                        #     ff.write(f'{G.edges(data=True)}\n\n')
                        
                        # tmp_f.append(ratio)
                        # tmp_c.append(curv)
                        
                        # c_l[i].append(tmp_c)
                        # f_l[i].append(tmp_f)
                        
                        # with open(res_path + metric + name + "_frac1.pkl", 'wb') as file:
                        #     pickle.dump(f_l, file)

                        # with open(res_path + metric + name + "_curvature1.pkl", 'wb') as file:
                        #     pickle.dump(c_l, file)
                    
                        # input()
                        
                        # print(f'Total edges is {edges_num}, length of alpha is {len(alp_l)}, average is {np.mean(alp_l)}')
                        
                        tmp_f.append(ratio)
                        tmp_c.append(curv)
                  
                    c_l[i].append(tmp_c)
                    f_l[i].append(tmp_f)
                    
                    
                with open(res_path + metric + name + "_frac2.pkl", 'wb') as file:
                    pickle.dump(f_l, file)

                with open(res_path + metric + name + "_curvature2.pkl", 'wb') as file:
                    pickle.dump(c_l, file)
                    

                # traj = unsafe_traj[0]
                # tmp_f = []
                # tmp_c = []
                # for input in traj:
                #     input = np.array(input)
                #     input = input.reshape(1, -1)
                #     input = torch.tensor(input)
                #     dims, nodes_num, edges_num, NN_w, edge_v, nodes, nodes_ori = getNN_info(cur_c, input)
   
                #     # build graph
                #     adj = build_adjm(nodes_num, dims, edge_v, nodes, NN_w)
                #     adj = adj.cpu().detach().numpy()
                #     curv, ratio = cal_curvature(adj, nodes, dims, nodes_ori.cpu())
                    
                #     tmp_f.append(ratio)
                #     tmp_c.append(curv)
                        
                # c_l['1'].append(tmp_c)
                # f_l['1'].append(tmp_f)
                
                # print(f'Controller {name} has {nodes_num} nodes, {edges_num} edges...')
                # print(f'Finish controller {name}....')
            
 
        
