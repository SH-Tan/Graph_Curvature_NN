from collections import defaultdict, deque
import networkx as nx
import community as community_louvain 
from typing import Dict, List, Tuple, Set


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




def find_all_backward_communities(curvature, b):
    node_communities = defaultdict(set)  # node -> set of community IDs
    communities = dict()                 # community ID -> set of nodes
    community_edges = dict()             # community ID -> set of (i, j) edges

    all_nodes = set()
    all_edges = set()
    neg_outgoing = defaultdict(set)      # node -> set of outgoing neighbors via negative edges
    neg_incoming = defaultdict(set)      # node -> set of incoming neighbors via negative edges

    # Parse curvature info
    for batch in range(b):
        ricci_curv = curvature[batch]
        for i, j, c in ricci_curv:
            i, j = int(i), int(j)
            all_edges.add((i, j))
            all_nodes.update([i, j])
            if c < 0:
                neg_outgoing[i].add(j)
                neg_incoming[j].add(i)
                
    # Find root nodes = those with no outgoing negative edges
    candidate_roots = set(all_nodes) - set(neg_outgoing.keys())

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
            community_id += 1

    # Prepare summary output
    summary = {
        cid: {
            "node_count": len(nodes),
            "edge_count": len(community_edges[cid]),
            "nodes": list(nodes),
            "edges": list(community_edges[cid])
        }
        for cid, nodes in communities.items()
    }

    graph_info = {
        "total_nodes": len(all_nodes),
        "total_edges": len(all_edges),
        "num_communities": len(summary)
    }

    return summary, dict(node_communities), graph_info



