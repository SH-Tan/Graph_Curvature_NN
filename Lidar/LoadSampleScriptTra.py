# load targets and input data

# compute control error on the training controllers and data
# compute control error on the holdout validation controllers
# to get confidence intervals


# get 

# to get confidence intervals for each of the rays, 
# compute outputs on validation set (sigmoids)

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Subset, TensorDataset
import matplotlib.pyplot as plt
import numpy as np


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

    print(list(controllers.keys()))
    return controllers

def load_all_data(data_filename = "cleandata_by_controller.yml"):
    with open(data_filename, 'rb') as f:
        data_dict = yaml.full_load(f)
    print(data_dict.keys())
    return data_dict


if __name__ == "__main__":
    controllers = load_all_controllers()
    #data_dict = load_all_data()
    #print(len(data_dict["Real"]))

    # this loads the basic 21 lidar scans from the real sensor data. It has been normalized between -0.5 and 0.5. 
    all_inputs = np.load("Fixed_Noisy_lidar_uncovered.npy")
    print(all_inputs.shape)

    all_outputs = vectorize_predict_yaml(controllers["DDPG_64_1"], all_inputs)
    print(all_outputs.shape)

    num_samples = 1000

    sampled_inputs = all_inputs[np.random.randint(all_inputs.shape[0], size=num_samples), :]

    print(sampled_inputs.shape)
    sampled_outputs = vectorize_predict_yaml(controllers["DDPG_64_1"], sampled_inputs)

    print(sampled_outputs.shape)

    print(sampled_inputs[0])

    with open("lidar_trajectories_by_controller.yml", 'r') as f:
        lidar_traj_by_controller = yaml.full_load(f)

    print("prints the shape of each trajectory: (number of datapoint, size of lidar) ")
    # the below list contains a numpy array for each trajectory for the given controller
    example_lidar_trajectories = [np.array(traj) for traj in lidar_traj_by_controller["DDPG_64_1"]]
    for traj in example_lidar_trajectories:
        print(traj.shape)

