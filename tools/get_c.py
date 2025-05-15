import numpy as np
from collections import defaultdict, Counter

def get_c(curvature, b, prefix_dims, label_neurons):
    edge_curvatures = {}
    layer_edges = defaultdict(list)
    all_edges = []

    num_layers = len(prefix_dims) - 1  # number of weight layers

    # Step 1: Collect all edges and negative curvature edges layer-by-layer
    for batch in range(b):
        ricci_curv = np.array(curvature[batch])
        for (i, j, curr) in ricci_curv:
            if curr > 1:
                continue
            i, j = int(i), int(j)
            all_edges.append((i, j, curr))
            edge_curvatures[(i, j)] = curr

            if curr < 0:
                i_layer = np.searchsorted(prefix_dims, i, side='right') - 1
                j_layer = np.searchsorted(prefix_dims, j, side='right') - 1
                if j_layer == i_layer + 1:
                    layer_edges[i_layer].append((i, j, curr))

    # Step 2: Build negative curvature paths from first layer forward
    current_paths = [[edge] for edge in layer_edges[0]]
    completed_paths = []

    for l in range(1, num_layers):
        next_edges_map = defaultdict(list)
        for edge in layer_edges[l]:
            next_edges_map[edge[0]].append(edge)

        next_paths = []
        for path in current_paths:
            last_node = path[-1][1]
            if last_node in next_edges_map:
                for next_edge in next_edges_map[last_node]:
                    next_paths.append(path + [next_edge])
            else:
                completed_paths.append(path)

        current_paths = next_paths

    completed_paths.extend(current_paths)

    # Step 3: Find single-hop paths and paths reaching label neurons
    single_hop_paths = []
    label_paths = []
    path_length_counts = Counter()

    for path in completed_paths:
        path_length = len(path)
        path_length_counts[path_length] += 1  # Count path length

        total_curv = sum(edge[2] for edge in path)

        if path_length == 1:
            single_hop_paths.append(path)

        last_edge = path[-1]
        j = last_edge[1]
        j_layer = np.searchsorted(prefix_dims, j, side='right') - 1
        j_idx = j - prefix_dims[j_layer]

        if j_layer == num_layers-1 and j_idx == label_neurons:
            label_paths.append((path, total_curv))

    # Step 4: Sort results
    single_hop_paths.sort(key=lambda path: path[0][2])
    label_paths.sort(key=lambda p: p[1])

    return all_edges, single_hop_paths, label_paths, edge_curvatures, dict(path_length_counts)
