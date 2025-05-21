from collections import defaultdict, deque
import networkx as nx
import community as community_louvain 
from typing import Dict, List, Tuple, Set
import numpy as np

import pandas as pd
from collections import defaultdict


def multi_community_from_output(curvature, b, prefix_dims):
    node_communities = defaultdict(set)
    community_sizes = defaultdict(int)

    all_nodes = set()
    all_edges = set()
    negative_edges = set()

    for batch in range(b):
        ricci_curv = curvature[batch]
        reverse_edges = defaultdict(list)

        for i, j, c in ricci_curv:
            i, j = int(i), int(j)
            all_nodes.update([i, j])
            all_edges.add((i, j))

            if c < 0:
                reverse_edges[j].append(i)
                negative_edges.add((i, j))

        output_nodes = range(prefix_dims[-2], prefix_dims[-1])

        for output in output_nodes:
            visited = set()
            queue = deque([output])
            while queue:
                node = queue.popleft()
                if node in visited:
                    continue
                visited.add(node)
                node_communities[node].add(output)
                community_sizes[output] += 1
                for prev in reverse_edges.get(node, []):
                    queue.append(prev)

    graph_info = {
        "total_nodes": len(all_nodes),
        "total_edges": len(all_edges),
        "negative_edges": len(negative_edges)
    }

    return dict(community_sizes), dict(node_communities), graph_info




def negative_edge_communities(curvature, b):
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        root_x = find(x)
        root_y = find(y)
        if root_x != root_y:
            parent[root_y] = root_x

    all_nodes = set()
    all_edges = set()
    negative_edges = []

    for batch in range(b):
        for i, j, c in curvature[batch]:
            i, j = int(i), int(j)
            all_nodes.update([i, j])
            all_edges.add((i, j))
            if c < 0:
                negative_edges.append((i, j))
                if i not in parent:
                    parent[i] = i
                if j not in parent:
                    parent[j] = j
                union(i, j)

    # Collect communities
    communities = defaultdict(set)
    for node in parent:
        root = find(node)
        communities[root].add(node)

    graph_info = {
        "total_nodes": len(all_nodes),
        "total_edges": len(all_edges),
        "negative_edges": len(negative_edges),
        "num_communities": len(communities)
    }

    return dict(communities), graph_info





def community_split_by_community_louvain(curvature, b):
    G = nx.Graph()
    all_nodes = set()
    all_edges = set()
    negative_edges = 0

    for batch in range(b):
        for u, v, curv in curvature[batch]:
            u, v = int(u), int(v)
            if curv < 0:
                all_nodes.update([u, v])
                all_edges.add((u, v))
                negative_edges += 1

                # Use magnitude of negative curvature as similarity weight
                G.add_edge(u, v, weight=abs(curv), curvature=curv)

    # Run Louvain on the graph with only negative curvature edges
    partition = community_louvain.best_partition(G, weight='weight')

    # Group nodes by community
    communities = defaultdict(list)
    for node, comm_id in partition.items():
        communities[comm_id].append(node)

    # Graph statistics
    graph_info = {
        "total_nodes": len(all_nodes),
        "total_edges": len(all_edges),
        "negative_edges": negative_edges,
        "num_communities": len(communities)
    }

    return G, partition, communities, graph_info




from collections import defaultdict, deque
import numpy as np

def find_all_backward_communities(curvature, b, prefix_dims, threshold=0.0):
    node_communities = defaultdict(set)
    communities = dict()
    community_edges = dict()
    community_layer_internal = dict()

    # Graph-wide tracking
    full_nodes = set()
    full_edges = set()

    neg_nodes = set()
    neg_edges = set()

    threshold_nodes = set()
    threshold_edges = set()

    neg_outgoing = defaultdict(set)
    neg_incoming = defaultdict(set)

    # Track layer edge counts
    layer_edge_counts = {
        "full": defaultdict(int),
        "negative": defaultdict(int),
        "filtered": defaultdict(int),
    }

    # Parse curvature info
    for batch in range(b):
        ricci_curv = curvature[batch]
        for i, j, c in ricci_curv:
            i, j = int(i), int(j)
            full_nodes.update([i, j])
            full_edges.add((i, j))

            i_layer = np.searchsorted(prefix_dims, i, side='right') - 1
            layer_edge_counts["full"][i_layer] += 1

            if c < 0:
                neg_nodes.update([i, j])
                neg_edges.add((i, j))
                layer_edge_counts["negative"][i_layer] += 1

            if c < threshold:
                threshold_nodes.update([i, j])
                threshold_edges.add((i, j))
                neg_outgoing[i].add(j)
                neg_incoming[j].add(i)
                layer_edge_counts["filtered"][i_layer] += 1

    # 1. Find community root nodes — no outgoing filtered edges
    candidate_roots = threshold_nodes - set(neg_outgoing.keys())
    community_id = 0

    for root in candidate_roots:
        visited = set()
        queue = deque([root])
        community_nodes = set()
        community_edge_set = set()

        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            community_nodes.add(node)
            node_communities[node].add(community_id)

            for prev in neg_incoming.get(node, []):
                community_edge_set.add((prev, node))
                queue.append(prev)

        if community_nodes:
            communities[community_id] = community_nodes
            community_edges[community_id] = community_edge_set

            # 3. Count internal community edges by source layer
            layer_counts = defaultdict(int)
            for u, v in community_edge_set:
                u_layer = np.searchsorted(prefix_dims, u, side='right') - 1
                layer_counts[u_layer] += 1

            community_layer_internal[community_id] = dict(layer_counts)
            community_id += 1

    # Summary by community
    summary = {
        cid: {
            "node_count": len(nodes),
            "edge_count": len(community_edges[cid]),
            "nodes": list(nodes),
            "edges": list(community_edges[cid]),
            "internal_edges_by_layer": community_layer_internal[cid]
        }
        for cid, nodes in communities.items()
    }

    # 2. Overall graph info
    graph_info = {
        "full_graph": {
            "total_nodes": len(full_nodes),
            "total_edges": len(full_edges)
        },
        "negative_graph": {
            "total_nodes": len(neg_nodes),
            "total_edges": len(neg_edges)
        },
        "filtered_graph": {
            "total_nodes": len(threshold_nodes),
            "total_edges": len(threshold_edges)
        },
        "num_communities": len(summary),
        "layer_edge_counts": {
            "full": dict(layer_edge_counts["full"]),
            "negative": dict(layer_edge_counts["negative"]),
            "filtered": dict(layer_edge_counts["filtered"]),
        }
    }

    return summary, dict(node_communities), graph_info




def write_graph_info_to_excel(graph_info, summary, prefix_dims, output_path="graph_summary.xlsx"):
    full_nodes = graph_info["full_graph"]["total_nodes"]
    full_edges = graph_info["full_graph"]["total_edges"]
    neg_nodes = graph_info["negative_graph"]["total_nodes"]
    neg_edges = graph_info["negative_graph"]["total_edges"]
    filtered_nodes = graph_info["filtered_graph"]["total_nodes"]
    filtered_edges = graph_info["filtered_graph"]["total_edges"]

    layer_edge_counts = graph_info.get("layer_edge_counts", {
        "full": defaultdict(int),
        "negative": defaultdict(int),
        "filtered": defaultdict(int)
    })

    # === Sheet 1: Global Graph Statistics ===
    global_stats = {
        "Metric": ["Total Nodes", "Total Edges"],
        "Full Graph": [full_nodes, full_edges],
        "Negative Curvature": [neg_nodes, neg_edges],
        "Negative Fraction": [neg_nodes / full_nodes, neg_edges / full_edges],
        "Below Threshold": [filtered_nodes, filtered_edges],
        "Threshold Fraction": [filtered_nodes / full_nodes, filtered_edges / full_edges],
    }
    global_df = pd.DataFrame(global_stats)

    # === Sheet 2: Community-Level Statistics ===
    community_data = []

    for cid, data in sorted(summary.items(), key=lambda x: -x[1]['node_count']):
        nodes = set(data["nodes"])
        edges = set(map(tuple, data["edges"]))

        node_fracs = {
            "full": len(nodes) / full_nodes if full_nodes else 0,
            "negative": len(nodes) / neg_nodes if neg_nodes else 0,
            "thresholded": len(nodes) / filtered_nodes if filtered_nodes else 0,
        }
        edge_fracs = {
            "full": len(edges) / full_edges if full_edges else 0,
            "negative": len(edges) / neg_edges if neg_edges else 0,
            "thresholded": len(edges) / filtered_edges if filtered_edges else 0,
        }

        community_data.append({
            "Community ID": cid,
            "Node Count": len(nodes),
            "Edge Count": len(edges),
            "Node Frac (Full)": node_fracs["full"],
            "Node Frac (Negative)": node_fracs["negative"],
            "Node Frac (Thresh)": node_fracs["thresholded"],
            "Edge Frac (Full)": edge_fracs["full"],
            "Edge Frac (Negative)": edge_fracs["negative"],
            "Edge Frac (Thresh)": edge_fracs["thresholded"]
        })

    community_df = pd.DataFrame(community_data)

    # === Optional Sheet 3: Internal Layer Edge Fractions per Community ===
    layer_edge_rows = []
    for cid, data in summary.items():
        layer_internal = data.get("internal_edges_by_layer", {})
        for layer_idx in sorted(layer_internal):
            count = layer_internal[layer_idx]
            full_total = layer_edge_counts["full"].get(layer_idx, 0)
            neg_total = layer_edge_counts["negative"].get(layer_idx, 0)
            filt_total = layer_edge_counts["filtered"].get(layer_idx, 0)

            layer_edge_rows.append({
                "Community ID": cid,
                "Layer": layer_idx,
                "Internal Edge Count": count,
                "Frac (Full)": count / full_total if full_total else 0,
                "Frac (Negative)": count / neg_total if neg_total else 0,
                "Frac (Thresholded)": count / filt_total if filt_total else 0,
                "Full Total": full_total,
                "Negative Total": neg_total,
                "Filtered Total": filt_total,
            })

    layer_df = pd.DataFrame(layer_edge_rows)

    # === Write to Excel ===
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        global_df.to_excel(writer, sheet_name="Global Stats", index=False)
        community_df.to_excel(writer, sheet_name="Community Stats", index=False)
        if not layer_df.empty:
            layer_df.to_excel(writer, sheet_name="Layer Edges", index=False)

    print(f"Graph summary written to: {output_path}")

