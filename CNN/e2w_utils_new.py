import torch
import torch.nn.functional as F
from collections import defaultdict
import numpy as np
from collections import Counter

def build_cnn_edge_weight_map(in_ch, in_size, out_ch, k, stride, padding, layer, prefix_dim, device="cpu"):
    """
    Build a mapping from each edge (input_node_idx, output_node_idx)
    to a CNN kernel weight index (out_ch, in_ch, kh, kw).

    Returns:
        edge_to_weight: dict[(int, int)] -> (out_ch, in_ch, kh, kw)
        unfolded_indices: torch.Tensor of shape (num_patches, receptive_field_size)
    """
    dummy = torch.arange(in_ch * in_size * in_size, device=device).reshape(1, in_ch, in_size, in_size).float()
    unfolded = F.unfold(dummy, kernel_size=k, stride=stride, padding=padding).transpose(1, 2).int()

    num_patches = unfolded.shape[1]  # number of output spatial positions
    rf_size = unfolded.shape[2]      # receptive field size = in_ch * k * k

    edge_to_weight = {}
    # Offset for this layer in the flattened graph
    node_offset_in = prefix_dim[layer]
    node_offset_out = prefix_dim[layer+1]

    for oc in range(out_ch):
        for patch_idx in range(num_patches):
            # receptive field input node indices
            input_nodes = unfolded[0, patch_idx].tolist()
            for in_c in range(in_ch):
                for kh in range(k):
                    for kw in range(k):
                        w_idx = (layer, oc, in_c, kh, kw)
                        in_node = input_nodes[in_c * k * k + kh * k + kw]
                        out_node = node_offset_out + oc * num_patches + patch_idx
                        edge_to_weight[(in_node + node_offset_in, out_node)] = w_idx

    return edge_to_weight


def clipped_median(arr):
    arr = np.asarray(arr)
    if len(arr) <= 2:
        # After removing min & max, nothing remains → return regular median
        return np.median(arr)
    
    sorted_arr = np.sort(arr)
    trimmed = sorted_arr[1:-1]   # remove smallest and largest
    return np.mean(trimmed)      # average of the remaining values


def aggregate_cnn_weight_curvature(edge_curvatures, edge_to_weight):
    """
    Aggregate per-edge curvature into average curvature per CNN kernel weight.
    Separate positive and negative curvature averages.

    Args:
        edge_curvatures (list[tuple[int,int,float]]): (i, j, curvature)
        edge_to_weight (dict): (i,j) -> (out_ch,in_ch,kh,kw)

    Returns:
        weight_curv_dict: dict[(out_ch,in_ch,kh,kw)] = avg_curvature
        weight_freq_dict: dict[(out_ch,in_ch,kh,kw)] = frequency
    """
    curv_sum_neg = defaultdict(list)
    curv_sum_pos = defaultdict(list)
    pos_freq = defaultdict(int)
    neg_freq = defaultdict(int)
    zero_freq = defaultdict(int)
    freq = defaultdict(int)

    for (i, j, c) in edge_curvatures:
        key = tuple(sorted((i, j)))
        if key not in edge_to_weight:
            continue
        w = edge_to_weight[key]
        if c < 0:
            curv_sum_neg[w].append(c)
        else:
            curv_sum_pos[w].append(c)
            
        freq[w] += 1
        
        if c >= 0:
            pos_freq[w] += 1
        # elif c == 0:
        #     zero_freq[w] += 1
        else:
            neg_freq[w] += 1

    # weight_curv = dict()
    neg_curv_weights = dict()
    pos_curv_weights = dict()
    
    for w in freq:
        negs = curv_sum_neg[w]
        poss = curv_sum_pos[w]

        if len(curv_sum_neg[w]) > 0:
            neg_curv_weights[w] = (np.min(negs), neg_freq[w], zero_freq[w])
        elif len(poss) > 0:
            pos_curv_weights[w] = (np.min(poss), pos_freq[w], zero_freq[w])
    # weight_curv = {w: (curv_sum[w], freq[w], pos_freq[w], neg_freq[w], zero_freq[w]) for w in freq}
    
    # neg_curv_weights = {
    #     w: (curv_sum[w]/neg_freq[w], neg_freq[w], zero_freq[w])
    #     for w in weight_curv
    #     if curv_sum[w] < 0
    # }

    # pos_curv_weights = {
    #     w: (curv_sum[w]/pos_freq[w], pos_freq[w], zero_freq[w])
    #     for w in weight_curv
    #     if (curv_sum[w] >= 0) # and (curv_sum[w]/pos_freq[w] < 1)
    # }
    
    return pos_curv_weights, neg_curv_weights


def count_weight_frequency(weight_sets, model_dims= None, para_dims=None):
    freq = Counter()
    pos_freq = Counter()
    neg_freq = Counter()
    zero_freq = Counter()
    curvature_sum = defaultdict(float)

    for weight_set in weight_sets:
        for w, c, f, p in weight_set:
            freq[w] += f
            zero_freq[w] += 1
            curvature_sum[w] += c

    results = []
    for w in freq:
        l, _, _, _, _ = w
        layer_info = model_dims[l + 2]
        out_s = layer_info["dim"]["out_size"]
        avg_c = curvature_sum[w] / zero_freq[w]
        # if not ((avg_c == 1) and (freq[w] < (out_s**2))):
        results.append((w, freq[w]/(out_s**2), avg_c, para_dims[l]))
    return results



if __name__ == '__main__':
    edge_to_weight, unfold = build_cnn_edge_weight_map(2, 4, 3, 3, 1, 0)

    print(edge_to_weight)

    print(unfold)

