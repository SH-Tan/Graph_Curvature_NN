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
        
        # res_name = model_full_n + metric + '_' + str(layer_num) + correct_suffix_res
        # misres_name = model_full_n + metric + '_' + str(layer_num) + misclassified_suffix_res

        with open(data_path + res_name, 'rb') as file:
            res_dict = pickle.load(file)
            
        for l in selected_classes:
            edge_list_gt = []
            node_list_gt = []
            curv_gt = []
            edge_list_gt_wo_input = []
            node_list_gt_wo_input = []
            curv_gt_wo_input = []
            edge_list_gt_wo_output = []
            node_list_gt_wo_output = []
            curv_gt_wo_output = []

            # Accumulate across examples
            output_layer_neg_indegrees_all = []  # List of lists
            high_indegree_counts = []  # List of ints      
            output_layer_in_degrees_all = []
            single_node_communities_list = []
            zero_node_negative_outgoing_list = []
            zero_node_count_list = []
            nonzero_node_negative_outgoing_list = []
            nonzero_node_count_list = []
            spanning_t_list = []
             
            with open(res_path + "community_" + str(l) + ".txt", "w+") as ff:
                ff.write(f'W = {metric}: For model {model_type} - {model_pre_name}, layer {layer_num}: \n')
                ff.write(f'For the correct examples - label {l}: \n')
                for (ricci, batch, dim, node_ori) in res_dict[l]:
                    prefix_dims = np.cumsum([0] + dim).tolist()
                    
                    result = analyze_graph_structure_with_community_indegree(
                        ricci, batch, prefix_dims, node_ori, threshold=thre
                    )
                    
                    in_degree_info = result["in_degree"]
                    summary = result["communities_w_input_layer"] 
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

                    # 2. Count all neurons with neg in-degree > 70
                    # high_indegree_count = sum(
                    #     1 for node, deg in neg_deg_dict.items() if deg > 70
                    # )

                    last_3_start = prefix_dims[-4]  # inclusive
                    last_3_end = prefix_dims[-1]    # exclusive

                    high_indegree_count = sum(
                        1 for node in range(last_3_start, last_3_end)
                        if neg_deg_dict.get(node, 0.0) > 60
                    )

                    high_indegree_counts.append(high_indegree_count)  # You may want to rename this variable
                    
                    # 3. Single-hop communities
                    single_node_communities = sum(1 for comm in summary.values() if len(comm['nodes']) == 2)
                    ff.write(f"Number of single-hop communities: {single_node_communities}\n")
                    single_node_communities_list.append(single_node_communities)
                    
                    # 4. zero input node and total outgoing negtaive curvature edges 
                    total_zero_nodes = result["total_zero_nodes"] 
                    total_out_neg_edge = result["zero_node_negative_outgoing_edges"] 
                    ff.write(f"Number of zero input node: {total_zero_nodes}, total negative outgoing edge number of theses nodes is {total_out_neg_edge}\n")

                    total_nonzero_nodes = result["total_nonzero_nodes"] 
                    total_out_neg_edge_non = result["nonzero_node_negative_outgoing_edges"] 
                    ff.write(f"Number of non-zero input node: {total_nonzero_nodes}, total negative outgoing edge number of theses nodes is {total_out_neg_edge_non}\n")
                    
                    # Accumulate for averaging
                    zero_node_count_list.append(total_zero_nodes)
                    zero_node_negative_outgoing_list.append(total_out_neg_edge)
                    nonzero_node_count_list.append(total_nonzero_nodes)
                    nonzero_node_negative_outgoing_list.append(total_out_neg_edge_non)

                    ff.write("\n=== Inter-Layer Edge Counts (All Edges) ===\n")
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
                    thresholds = [20, 30, 50, 100, 150, 200]
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
                        gt_nodes = true_comm_data["nodes"]

                        true_nodes = true_comm_data["node_count"]
                        true_edges = true_comm_data["edge_count"]
                        total_curvature_true = true_comm_data.get("total_internal_curvature", 0.0)
                        
                        true_nodes_wo_input = true_comm_data["stats_wo_input_layer"]
                        total_curvature_true_wo_input = true_nodes_wo_input.get("total_internal_curvature", 0.0)
                        
                        true_nodes_wo_output = true_comm_data["stats_wo_output_layer"]
                        total_curvature_true_wo_output = true_nodes_wo_output.get("total_internal_curvature", 0.0)

                        ff.write("\n--- Ground Truth Community ---\n")
                        ff.write(f"  Community ID: {true_community_id}\n")
                        ff.write(f"  Node count: {true_nodes}\n")
                        ff.write(f"  Edge count: {true_edges}\n")
                        ff.write(f"  Total curvature: {total_curvature_true:.4f}\n")
                        ff.write(f"  Node count wo input: {true_nodes_wo_input['node_count']}\n")
                        ff.write(f"  Edge count wo input: {true_nodes_wo_input['edge_count']}\n")
                        ff.write(f"  Total curvature wo input: {total_curvature_true_wo_input:.4f}\n")
                        ff.write(f"  Node count wo output: {true_nodes_wo_output['node_count']}\n")
                        ff.write(f"  Edge count wo output: {true_nodes_wo_output['edge_count']}\n")
                        ff.write(f"  Total curvature wo output: {total_curvature_true_wo_output:.4f}\n")

                        curv_gt.append(total_curvature_true)
                        node_list_gt.append(true_nodes)
                        edge_list_gt.append(true_edges)
                        
                        curv_gt_wo_input.append(total_curvature_true_wo_input)
                        node_list_gt_wo_input.append(true_nodes_wo_input["node_count"])
                        edge_list_gt_wo_input.append(true_nodes_wo_input["edge_count"])
                        
                        curv_gt_wo_output.append(total_curvature_true_wo_output)
                        node_list_gt_wo_output.append(true_nodes_wo_output["node_count"])
                        edge_list_gt_wo_output.append(true_nodes_wo_output["edge_count"])
                        
                        for layer_idx in range(len(prefix_dims) - 1):
                            start, end = prefix_dims[layer_idx], prefix_dims[layer_idx + 1]
                            layer_nodes = set(range(start, end))
                            comm_nodes = set(gt_nodes) & layer_nodes

                            if comm_nodes:
                                indegs = [neg_deg_dict.get(n, 0) for n in comm_nodes]
                                cwdegs = [in_deg_dict.get(n, 0.0) for n in comm_nodes]

                                avg_indeg = np.mean(indegs)
                                avg_cwdeg = np.mean(cwdegs)

                                gt_comm_layer_avg_indegree[layer_idx].append(avg_indeg)
                                gt_comm_layer_avg_cwdegree[layer_idx].append(avg_cwdeg)

                                ff.write(f"GT Community Layer {layer_idx}: Avg. In-Deg = {avg_indeg:.2f}, Avg. CW-Deg = {avg_cwdeg:.2f}\n") 
                    else:
                        ff.write("\n--- Ground Truth Community ---\n")
                        ff.write(f"  Not found for node {true_output_node}\n")

                        edge_list_gt.append(0)
                        node_list_gt.append(0)
                        curv_gt.append(0)
                        edge_list_gt_wo_input.append(0)
                        node_list_gt_wo_input.append(0)
                        curv_gt_wo_input.append(0)
                        edge_list_gt_wo_output.append(0)
                        node_list_gt_wo_output.append(0)
                        curv_gt_wo_output.append(0)


                edge_list_gt = np.array(edge_list_gt)
                node_list_gt = np.array(node_list_gt)
                curv_gt = np.array(curv_gt)
                edge_list_gt_wo_input = np.array(edge_list_gt_wo_input)
                node_list_gt_wo_input = np.array(node_list_gt_wo_input)
                curv_gt_wo_input = np.array(curv_gt_wo_input)
                edge_list_gt_wo_output = np.array(edge_list_gt_wo_output)
                node_list_gt_wo_output = np.array(node_list_gt_wo_output)
                curv_gt_wo_output = np.array(curv_gt_wo_output)
                
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
                avg_second_layer_high = sum(high_indegree_counts) / num_examples if num_examples > 0 else 0
                avg_single_hop_communities = np.mean(single_node_communities_list)

                ff.write(f'\nFor the correctly classified examples:\n')
                ff.write(f'There are total {len(node_list_gt)} GT communities, {np.sum(node_list_gt == 0)} are zero (not exist).\n')

                ff.write(f' (overall) The average node number (non-zero) is {np.mean(node_list_gt[node_list_gt != 0]):.3f}, ')
                ff.write(f'edge number is {np.mean(edge_list_gt[edge_list_gt != 0]):.2f}\n')
                ff.write(f'curvature is {np.mean(curv_gt[curv_gt != 0]):.4f}\n')
                
                ff.write(f'\n (without input layer) The average node number (non-zero) is {np.mean(node_list_gt_wo_input[node_list_gt_wo_input != 0]):.3f}, ')
                ff.write(f'edge number is {np.mean(edge_list_gt_wo_input[edge_list_gt_wo_input != 0]):.2f}\n')
                ff.write(f'curvature is {np.mean(curv_gt_wo_input[curv_gt_wo_input != 0]):.4f}\n')
                
                ff.write(f'\n (without output layer) The average node number (non-zero) is {np.mean(node_list_gt_wo_output[node_list_gt_wo_output != 0]):.3f}, ')
                ff.write(f'edge number is {np.mean(edge_list_gt_wo_output[edge_list_gt_wo_output != 0]):.2f}\n')
                ff.write(f'curvature is {np.mean(curv_gt_wo_output[curv_gt_wo_output != 0]):.4f}\n')
                
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

                ff.write("\n=== Aggregated Zero Input Node Statistics ===\n")
                ff.write(f"Average number of zero input nodes: {np.mean(zero_node_count_list):.2f}\n")
                ff.write(f"Average number of negative outgoing edges from zero input nodes: {np.mean(zero_node_negative_outgoing_list):.2f}\n")
                
                ff.write("\n=== Aggregated Non - Zero Input Node Statistics ===\n")
                ff.write(f"Average number of non-zero input nodes: {np.mean(nonzero_node_count_list):.2f}\n")
                ff.write(f"Average number of negative outgoing edges from zero input nodes: {np.mean(nonzero_node_negative_outgoing_list):.2f}\n")
                
                # ff.write(f"\nThe average number of spanning trees of the GT community is {np.mean(spanning_t_list)}\n")

                input_layer_size = prefix_dims[1] - prefix_dims[0]
                ff.write(f"Average percentage of zero input nodes: {np.mean(zero_node_count_list) / input_layer_size * 100:.2f}%\n")

                ff.write(f"\n=== Average # of neurons with neg. in-degree > 60 for overall graph: {avg_second_layer_high:.2f} ===\n")
                ff.write(f"\n=== Average # of single hop communities: {avg_single_hop_communities:.2f} ===\n")
                
                print(f'Finish label {l}.')
