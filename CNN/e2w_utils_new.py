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



def aggregate_cnn_weight_curvature(edge_curvatures, edge_to_weight):
    """
    Aggregate per-edge curvature into average curvature per CNN kernel weight.
    Separate positive and negative curvature averages.
    """

    # Store the minimum curvature for each weight
    min_curv = {}
    pos_cnt = defaultdict(int)
    neg_cnt = defaultdict(int)
    freq = defaultdict(int)

    for i, j, c in edge_curvatures:

        # Fastest possible normalized key
        key = (i, j) if i < j else (j, i)

        # Skip if edge not linked to any weight
        w = edge_to_weight.get(key)
        if w is None:
            continue

        # Track minimum curvature
        if (w not in min_curv) or (c < min_curv[w]):
            min_curv[w] = c

        # Frequency tracking
        freq[w] += 1
        if c >= 0:
            pos_cnt[w] += 1
        else:
            neg_cnt[w] += 1

    # Separate positive/negative weights
    pos_curv_weights = {}
    neg_curv_weights = {}

    for w in freq:
        cmin = min_curv[w]
        if cmin < 0:
            neg_curv_weights[w] = (cmin, neg_cnt[w], 0)
        else:
            pos_curv_weights[w] = (cmin, pos_cnt[w], 0)

    return pos_curv_weights, neg_curv_weights


def count_weight_frequency(weight_sets, model_dims= None, para_dims=None):
    freq = Counter()
    pos_freq = Counter()
    neg_freq = Counter()
    zero_freq = Counter()
    curvature_sum = defaultdict(list)

    for weight_set in weight_sets:
        for w, c, f, p in weight_set:
            freq[w] += f
            zero_freq[w] += 1
            curvature_sum[w].append(c)

    results = []
    for w in freq:
        l, _, _, _, _ = w
        layer_info = model_dims[l + 2]
        out_s = layer_info["dim"]["out_size"]
        avg_c = np.min(curvature_sum[w])
        # if not ((avg_c == 1) and (freq[w] < (out_s**2))):
        results.append((w, freq[w]/(out_s**2), avg_c, para_dims[l]))
    return results


def cnn_results(min_cnn, count_cnn, model_dims, para_dims):
    out = []

    for w, cmin in min_cnn.items():
        l = w[0]  # layer index
        layer_info = model_dims[l + 2]
        out_size = layer_info["dim"]["out_size"]

        norm_freq = count_cnn[w] / (out_size**2)

        out.append((w, norm_freq, cmin, para_dims[l]))

    return out



if __name__ == '__main__':
    edge_to_weight, unfold = build_cnn_edge_weight_map(2, 4, 3, 3, 1, 0)

    print(edge_to_weight)

    print(unfold)

