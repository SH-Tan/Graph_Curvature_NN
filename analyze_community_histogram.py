import matplotlib.pyplot as plt
import pickle
import numpy as np

from collections import defaultdict
import os
import networkx as nx
import sympy

import sys
sys.path.append("..")

import tools.utils as utils
from tools.get_community import *

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")




def cal_community_npy(args):
    model_type = args.model_type
    model_pre_name = args.model_name
    res_path = args.mnist_res_path
    data_path = args.mnist_data_path
    metric = args.metric
    dataset = args.dataset
    cifar = args.cifar
    thre = args.threshold
    
    model_full_n = model_type.lower() + model_pre_name.lower()

    if not os.path.exists(res_path):
        os.makedirs(res_path)

    layer = [2]
    if model_type.lower() == "fc" and "big" not in model_pre_name.lower():
        layer = [2,4]
        
    selected_classes = [0,1,2,3,4,5,6,7,8,9]

    # Plot the data
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
            
        elif model_type.lower() == "cnn" and dataset.lower() == "cifar":
            robust_suffix_gs = "_graphsize_robust_cifar.pkl"
            norobust_suffix_gs = "_graphsize_norobust_cifar.pkl"
            robust_suffix_res = "_res_robust_cifar.pkl"
            norobust_suffix_res = "_res_norobust_cifar.pkl"
            
            if cifar.lower() == 'big':
                robust_suffix = "frac_robust_cifar_big.pkl"
                norobust_suffix = "frac_norobust_cifar_big.pkl"
            
        else:
            raise Exception("Invalid model type, model type should be {fc, fc_linear, cnn}!")
        
        with open(data_path + res_name, 'rb') as file:
            res_dict = pickle.load(file)
            
        for l in selected_classes:
            # Accumulate across examples
            non_input_in_degrees_all_examples = []

            for (ricci, batch, dim, node_ori) in res_dict[l]:
                prefix_dims = np.cumsum([0] + dim).tolist()
                
                result = analyze_graph_structure_with_community_indegree(
                    ricci, batch, prefix_dims, node_ori, threshold=thre
                )
                
                in_degree_info = result["in_degree"]
                neg_deg_dict = in_degree_info["filtered"]
                
                # Get non-input nodes only
                non_input_node_indices = range(prefix_dims[1], prefix_dims[-1])
                example_in_degrees = [neg_deg_dict.get(node, 0.0) for node in non_input_node_indices]
                non_input_in_degrees_all_examples.append(example_in_degrees)
                    
            # ==== After all examples processed for label l ====
            in_deg_array = np.array(non_input_in_degrees_all_examples)
            num_examples = len(in_deg_array)
            selected_examples = in_deg_array[:min(5, num_examples)]
            mean_in_degrees = in_deg_array.mean(axis=0)
            
            out_data = {f"example{i+1}": selected_examples[i] for i in range(len(selected_examples))}
            out_data["mean"] = mean_in_degrees
            out_data["num_nodes"] = mean_in_degrees.shape[0]
            
            out_file = os.path.join(res_path, f"in_degree_class_{l}.npz")
            np.savez(out_file, **out_data)

                    
                    

                