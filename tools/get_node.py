import numpy as np
from collections import defaultdict, Counter

def get_key_nodes(curvature, b, prefix_dims):
    edge_curvatures = {}
    layer_edges = defaultdict(list)
    all_edges = []

    num_layers = len(prefix_dims) - 1

    in_deg = Counter()
    out_deg = Counter()

    # Step 1: Collect edges and negative curvature edges by layer
    for batch in range(b):
        ricci_curv = np.array(curvature[batch])
        for (i, j, curr) in ricci_curv:
            if curr > 1:
                continue
            i, j = int(i), int(j)
            edge_curvatures[(i, j)] = curr
            all_edges.append((i, j, curr))

            if curr < 0:
                i_layer = np.searchsorted(prefix_dims, i, side='right') - 1
                j_layer = np.searchsorted(prefix_dims, j, side='right') - 1
                if j_layer == i_layer + 1:
                    layer_edges[i_layer].append((i, j, curr))

                    # Only count degrees for internal (hidden layer) nodes
                    if 0 < i_layer < num_layers - 1:
                        out_deg[i] += 1
                    if 0 < j_layer < num_layers - 1:
                        in_deg[j] += 1

    # Step 2: Identify terminal single-hop paths
    next_edge_sources = set(edge[0] for layer in layer_edges.values() for edge in layer)
    terminal_single_hop_nodes = set()

    for edge in layer_edges[0]:  # edges from input layer
        _, j, _ = edge
        if j not in next_edge_sources:  # j doesn't serve as a source in any other edge
            terminal_single_hop_nodes.add(j)

    # Step 3: Create internal node list, excluding single-hop terminals
    internal_nodes = set(in_deg.keys()).union(out_deg.keys())
    filtered_internal_nodes = internal_nodes - terminal_single_hop_nodes

    sorted_internal_nodes = sorted(
        filtered_internal_nodes,
        key=lambda node: in_deg[node] + out_deg[node],
        reverse=True
    )

    return list(terminal_single_hop_nodes), list(sorted_internal_nodes) 
