import matplotlib.pyplot as plt
import pickle
import numpy as np

from collections import defaultdict
import os

import sys
sys.path.append("..")

import tools.utils as utils
from tools.get_community import *

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")



def cal_community(args):
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
        # cnn model
        elif model_type.lower() == "cnn" and dataset.lower() == "mnist":
            robust_suffix_gs = "_graphsize_robust_cnn.pkl"
            norobust_suffix_gs = "_graphsize_norobust_cnn.pkl"
            robust_suffix_res = "_res_robust_cnn.pkl"
            norobust_suffix_res = "_res_norobust_cnn.pkl"
            
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
        
        res_name = model_full_n + metric + '_' + str(layer_num) + correct_suffix_res
        misres_name = model_full_n + metric + '_' + str(layer_num) + misclassified_suffix_res

        with open(data_path + res_name, 'rb') as file:
            res_dict = pickle.load(file)
            
        for l in selected_classes:
            edge_list_correct = []
            node_list_correct = []
            curv_correct = []
            edge_list_other = []
            node_list_other= []     
            curv_other = [] 
            
            # Accumulate across examples
            output_layer_neg_indegrees_all = []  # List of lists
            second_layer_high_indegree_counts = []  # List of ints      
            output_layer_in_degrees_all = []
             
            with open(res_path + "community_" + str(l) + ".txt", "w+") as ff:
                ff.write(f'W = {metric}: For model {model_type} - {model_pre_name}, layer {layer_num}: \n')
                ff.write(f'For the correct examples - label {l}: \n')
                for (ricci, batch, dim) in res_dict[l]:
                    prefix_dims = np.cumsum([0] + dim).tolist()
                    
                    result = analyze_graph_structure_with_community_indegree(
                        ricci, batch, prefix_dims, threshold=thre
                    )
                    
                    in_degree_info = result["in_degree"]
                    summary = result["communities_wo_input_layer"] 
                    node_communities = result["node_communities"] 
                    inter_layer_info = result["inter_layer_edges"]
                    inter_layer_neg_info = result["inter_layer_edges_neg"]
                    inter_layer_filteres_info = result["inter_layer_edges_filtered"] 
                    neg_deg_dict = in_degree_info["filtered"]
                    in_deg_dict = result["curvature_weighted_in_degree"]
                    
                    # Get node indices for true and predicted labels
                    true_output_node = prefix_dims[-2] + l
                    # Get community IDs that contain each output node
                    true_communities  = node_communities.get(true_output_node, set())
                                        
                    # 1. Negative in-degree for each output neuron
                    output_layer_start = prefix_dims[-2]
                    output_layer_end = prefix_dims[-1]

                    neg_output_indegrees = [
                        neg_deg_dict.get(node, 0) for node in range(output_layer_start, output_layer_end)
                    ]
                    output_layer_neg_indegrees_all.append(neg_output_indegrees)
                    
                    
                    output_layer_in_degrees = [
                        in_deg_dict.get(node, 0.0) for node in range(output_layer_start, output_layer_end)
                    ]
                    output_layer_in_degrees_all.append(output_layer_in_degrees)
                    
                    ff.write("=== Curvature-Weighted In-Degree for Output Neurons (This Example) ===\n")
                    for i, val in enumerate(output_layer_in_degrees):
                        ff.write(f"  Output neuron {i}: {val:.4f}\n")
                    ff.write("\n")

                    # 2. Count neurons in second layer with in-degree > 200 (assuming second layer is index 1)
                    second_layer_start = prefix_dims[1]
                    second_layer_end = prefix_dims[2]

                    high_indegree_count = sum(
                        1 for node in range(second_layer_start, second_layer_end)
                        if neg_deg_dict.get(node, 0) > 200
                    )
                    second_layer_high_indegree_counts.append(high_indegree_count)

                    ff.write("=== Inter-Layer Edge Counts (All Edges) ===\n")
                    for (layer_from, layer_to), count in sorted(inter_layer_info.items()):
                        ff.write(f"  From layer {layer_from} to layer {layer_to}: {count} edges\n")
                    ff.write("\n")

                    ff.write("=== Inter-Layer Edge Counts (Negative Edges) ===\n")
                    for (layer_from, layer_to), count in sorted(inter_layer_neg_info.items()):
                        ff.write(f"  From layer {layer_from} to layer {layer_to}: {count} edges\n")
                    ff.write("\n")
                    
                    ff.write(f"=== Inter-Layer Edge Counts (Filtered Edges < {thre}) ===\n")
                    for (layer_from, layer_to), count in sorted(inter_layer_filteres_info.items()):
                        ff.write(f"  From layer {layer_from} to layer {layer_to}: {count} edges\n")
                    ff.write("\n")
                    
                    # Containers for accumulating averages
                    per_layer_avg_indegree = defaultdict(list)
                    per_layer_avg_cwdegree = defaultdict(list)
                    gt_comm_layer_avg_indegree = defaultdict(list)
                    gt_comm_layer_avg_cwdegree = defaultdict(list)

                    # === Per-layer stats for negative graph ===
                    for layer_idx in range(len(prefix_dims) - 1):
                        start, end = prefix_dims[layer_idx], prefix_dims[layer_idx + 1]
                        nodes = list(range(start, end))
                        indegs = [neg_deg_dict.get(n, 0) for n in nodes]
                        cwdegs = [in_deg_dict.get(n, 0.0) for n in nodes]
                        
                        if nodes:
                            avg_indeg = np.mean(indegs)
                            avg_cwdeg = np.mean(cwdegs)

                            per_layer_avg_indegree[layer_idx].append(avg_indeg)
                            per_layer_avg_cwdegree[layer_idx].append(avg_cwdeg)

                            ff.write(f"Layer {layer_idx}: Avg. Neg In-Degree = {avg_indeg:.2f}, Avg. CW-Degree = {avg_cwdeg:.2f}\n")

                    ff.write("\n=== Node Counts by Layer for In-degree Thresholds ===\n")
                    thresholds = [50, 100, 150, 200]
                    num_layers = len(prefix_dims) - 1

                    for deg_type in ["full", "negative", "filtered"]:
                        deg_dict = in_degree_info.get(deg_type, {})
                        ff.write(f"=== Node Counts by Layer for {deg_type.capitalize()} In-degree Thresholds ===\n")

                        for layer_idx in range(num_layers):
                            layer_start = prefix_dims[layer_idx]
                            layer_end = prefix_dims[layer_idx + 1]

                            nodes_in_layer = range(layer_start, layer_end)

                            ff.write(f"Layer {layer_idx} (nodes {layer_start} to {layer_end - 1}):\n")
                            for threshold in thresholds:
                                count = sum(1 for node in nodes_in_layer if deg_dict.get(node, 0) > threshold)
                                ff.write(f"  Nodes with in-degree > {threshold}: {count}\n")
                            ff.write("\n")

                        # Print individual node in-degree in output layer
                        output_layer_start = prefix_dims[-2]
                        output_layer_end = prefix_dims[-1]
                        ff.write(f"=== Individual Node In-Degrees in Output Layer ({deg_type.capitalize()}) ===\n")
                        for node in range(output_layer_start, output_layer_end):
                            deg = deg_dict.get(node, 0)
                            ff.write(f"  Node {node}: {deg}\n")
                        ff.write("\n")

                    # Extract stats for the true label community (only one community expected)
                    if true_communities:
                        true_community_id = next(iter(true_communities))
                        true_comm_data = summary[true_community_id]

                        true_nodes = set(true_comm_data["nodes"])
                        true_edges = set(map(tuple, true_comm_data["edges"]))
                        total_curvature_true = true_comm_data.get("total_internal_curvature", 0.0)

                        ff.write("\n--- Ground Truth Community ---\n")
                        ff.write(f"  Community ID: {true_community_id}\n")
                        ff.write(f"  Node count: {len(true_nodes)}\n")
                        ff.write(f"  Edge count: {len(true_edges)}\n")
                        ff.write(f"  Total curvature: {total_curvature_true:.4f}\n")

                        curv_correct.append(total_curvature_true)
                        node_list_correct.append(len(true_nodes))
                        edge_list_correct.append(len(true_edges))
                        
                        for layer_idx in range(len(prefix_dims) - 1):
                            start, end = prefix_dims[layer_idx], prefix_dims[layer_idx + 1]
                            layer_nodes = set(range(start, end))
                            comm_nodes = true_nodes & layer_nodes

                            if comm_nodes:
                                indegs = [neg_deg_dict.get(n, 0) for n in comm_nodes]
                                cwdegs = [in_deg_dict.get(n, 0.0) for n in comm_nodes]

                                avg_indeg = np.mean(indegs)
                                avg_cwdeg = np.mean(cwdegs)

                                gt_comm_layer_avg_indegree[layer_idx].append(avg_indeg)
                                gt_comm_layer_avg_cwdegree[layer_idx].append(avg_cwdeg)

                                ff.write(f"GT Community Layer {layer_idx}: Avg. In-Deg = {avg_indeg:.2f}, Avg. CW-Deg = {avg_cwdeg:.2f}\n")

                        # Find largest community other than true community
                        largest_community_id = None
                        largest_size = -1

                        for cid, comm in summary.items():
                            if cid == true_community_id:
                                continue
                            comm_size = len(comm["nodes"])
                            if comm_size > largest_size:
                                largest_community_id = cid
                                largest_size = comm_size

                        if largest_community_id is not None:
                            largest_comm_data = summary[largest_community_id]
                            largest_nodes = set(largest_comm_data["nodes"])
                            largest_edges = set(map(tuple, largest_comm_data["edges"]))
                            total_curvature_largest = largest_comm_data.get("total_internal_curvature", 0.0)

                            node_list_other.append(len(largest_nodes))
                            edge_list_other.append(len(largest_edges))
                            curv_other.append(total_curvature_largest)

                            ff.write("\n--- Largest Non-GT Community ---\n")
                            ff.write(f"  Community ID: {largest_community_id}\n")
                            ff.write(f"  Node count: {len(largest_nodes)}\n")
                            ff.write(f"  Edge count: {len(largest_edges)}\n")
                            ff.write(f"  Total curvature: {total_curvature_largest:.4f}\n")
                        else:
                            node_list_other.append(0)
                            edge_list_other.append(0)
                            curv_other.append(0)
                            ff.write("\n--- Largest Non-GT Community ---\n")
                            ff.write("  Not found (only GT community exists)\n")

                    else:
                        ff.write("\n--- Ground Truth Community ---\n")
                        ff.write(f"  Not found for node {true_output_node}\n")

                        node_list_correct.append(0)
                        edge_list_correct.append(0)
                        curv_correct.append(0)


                node_list_correct = np.array(node_list_correct)
                edge_list_correct = np.array(edge_list_correct)
                node_list_other = np.array(node_list_other)
                edge_list_other = np.array(edge_list_other)
                curv_correct = np.array(curv_correct)
                curv_other = np.array(curv_other)
                
                num_examples = len(output_layer_neg_indegrees_all)
                output_size = len(output_layer_neg_indegrees_all[0]) if output_layer_neg_indegrees_all else 0

                avg_output_neg_indegree = [
                    sum(example[i] for example in output_layer_neg_indegrees_all) / num_examples
                    for i in range(output_size)
                ]
                
                avg_output_cw_degree = [
                    sum(example[i] for example in output_layer_in_degrees_all) / num_examples
                    for i in range(output_size)
                ]

                # Average count of high in-degree neurons in 2nd layer
                avg_second_layer_high = sum(second_layer_high_indegree_counts) / num_examples if num_examples > 0 else 0

                ff.write(f'\nFor the correctly classified examples:\n')
                ff.write(f'There are total {len(node_list_correct)} communities, {np.sum(node_list_correct == 0)} are zero (not exist).\n')

                ff.write(f'The average node number (non-zero) is {np.mean(node_list_correct[node_list_correct != 0]):.3f}, ')
                ff.write(f'edge number is {np.mean(edge_list_correct[edge_list_correct != 0]):.2f}\n')
                ff.write(f'curvature is {np.mean(curv_correct[curv_correct != 0]):.4f}\n')

                ff.write(f'The average node number (non-zero) for largest non-GT communities is {np.mean(node_list_other[node_list_other != 0]):.3f}, ')
                ff.write(f'edge number is {np.mean(edge_list_other[edge_list_other != 0]):.2f}\n')
                ff.write(f'curvature is {np.mean(curv_other[curv_other != 0]):.4f}\n')
                
                ff.write("\n=== Overall Avg. In-Degree and CW-Degree per Layer (Negative Graph) ===\n")
                for layer_idx in sorted(per_layer_avg_indegree.keys()):
                    avg_indeg = np.mean(per_layer_avg_indegree[layer_idx])
                    avg_cwdeg = np.mean(per_layer_avg_cwdegree[layer_idx])
                    ff.write(f"Layer {layer_idx}: Avg In-Deg = {avg_indeg:.2f}, Avg CW-Deg = {avg_cwdeg:.2f}\n")

                ff.write("\n=== Overall Avg. In-Degree and CW-Degree per Layer (GT Community Nodes) ===\n")
                for layer_idx in sorted(gt_comm_layer_avg_indegree.keys()):
                    avg_indeg = np.mean(gt_comm_layer_avg_indegree[layer_idx])
                    avg_cwdeg = np.mean(gt_comm_layer_avg_cwdegree[layer_idx])
                    ff.write(f"Layer {layer_idx}: Avg In-Deg = {avg_indeg:.2f}, Avg CW-Deg = {avg_cwdeg:.2f}\n")
                
                ff.write("\n=== Average Negative In-Degree of Output Neurons ===\n")
                for i, (val1, val2) in enumerate(zip(avg_output_neg_indegree,avg_output_cw_degree)):
                    ff.write(f"  Output neuron {i}: in-degee edge: {val1:.2f} - curvature: {val2:.2f}\n")

                ff.write(f"\n=== Average # of 2nd-layer neurons with neg. in-degree > 200: {avg_second_layer_high:.2f} ===\n")
                
                print(f'Finish label {l}.')
