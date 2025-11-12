import torch
import torch.nn.functional as F
from collections import defaultdict

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

    Args:
        edge_curvatures (list[tuple[int,int,float]]): (i, j, curvature)
        edge_to_weight (dict): (i,j) -> (out_ch,in_ch,kh,kw)

    Returns:
        weight_curv_dict: dict[(out_ch,in_ch,kh,kw)] = avg_curvature
        weight_freq_dict: dict[(out_ch,in_ch,kh,kw)] = frequency
    """
    curv_sum = defaultdict(float)
    freq = defaultdict(int)

    for (i, j, c) in edge_curvatures:
        key = tuple(sorted((i, j)))
        if key not in edge_to_weight:
            continue
        w = edge_to_weight[key]
        curv_sum[w] += c
        freq[w] += 1

    weight_curv = {w: (curv_sum[w], freq[w]) for w in freq}

    return weight_curv, freq


def count_weight_frequency(weight_sets, model_dims= None):
    freq = Counter()
    curvature_sum = defaultdict(float)

    for weight_set in weight_sets:
        for w, c, f in weight_set:
            freq[w] += f
            curvature_sum[w] += c

    results = []
    for w in freq:
        l, _, _, _, _ = w
        layer_info = model_dims[l + 2]
        out_s = layer_info["dim"]["out_size"]
        avg_c = curvature_sum[w] / freq[w]
        results.append((w, freq[w] / (out_s**2), avg_c))
    return results



if __name__ == '__main__':
    edge_to_weight, unfold = build_cnn_edge_weight_map(2, 4, 3, 3, 1, 0)

    print(edge_to_weight)

    print(unfold)

