import numpy as np
from collections import defaultdict

def get_c(curvature, b, prefix_dims):
    edge_curvatures = {}
    layer_edges = defaultdict(list)  # key: layer index, value: list of (i, j, curv)
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
                # No continuation from this path
                completed_paths.append(path)

        current_paths = next_paths

    # Add any remaining paths that survived to the end
    completed_paths.extend(current_paths)

    # Step 3: Separate paths by whether they reach the last layer
    neg_paths_last = []
    neg_paths_other = []

    for path in completed_paths:
        last_node = path[-1][1]
        j_layer = np.searchsorted(prefix_dims, last_node, side='right') - 1
        if j_layer == num_layers - 1:
            neg_paths_last.append(path)
        else:
            neg_paths_other.append(path)


    neg_paths_last.sort(key=lambda path: sum(e[2] for e in path))
    neg_paths_other.sort(key=lambda path: sum(e[2] for e in path))


    return all_edges, neg_paths_last, neg_paths_other, edge_curvatures
