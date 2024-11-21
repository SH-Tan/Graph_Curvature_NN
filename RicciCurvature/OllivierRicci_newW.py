"""
A class to compute the Ollivier-Ricci curvature of a given NetworkX graph.
"""
import heapq
import math
import multiprocessing as mp
import time
from functools import lru_cache
from importlib import util

import networkit as nk
import networkx as nx
import numpy as np
import torch.nn as nn
import torch
import ot

from .util import logger, set_verbose, cut_graph_by_cutoff, get_rf_metric_cutoff

EPSILON = 1e-7  # to prevent divided by zero

# ---Shared global variables for multiprocessing used.---
_Gk = nk.graph.Graph()
_alpha = 0.5
_weight = "weight"
_method = "OTDSinkhornMix"
_base = math.e
_exp_power = 2
_proc = mp.cpu_count()
_cache_maxsize = 1000000
_shortest_path = "all_pairs"
_nbr_topk = 3000
_OTDSinkhorn_threshold = 2000
_apsp = {}
_node_value = None
_dirG = None


# -------------------------------------------------------

@lru_cache(_cache_maxsize)
def _get_single_node_neighbors_distributions(node, direction="successors"):
    if _Gk.isDirected():
        if direction == "predecessors":
            neighbors = list(_Gk.iterInNeighbors(node))
        else:  # successors
            neighbors = list(_Gk.iterNeighbors(node))
    else:
        neighbors = list(_Gk.iterNeighbors(node))

    # Get sum of distributions from x's all neighbors
    heap_weight_node_pair = []
    for nbr in neighbors:
        if direction == "predecessors":
            w = _base ** (-_Gk.weight(nbr, node) ** _exp_power)
        else:  # successors
            w = _base ** (-_Gk.weight(node, nbr) ** _exp_power)

        if len(heap_weight_node_pair) < _nbr_topk:
            heapq.heappush(heap_weight_node_pair, (w, nbr))
        else:
            heapq.heappushpop(heap_weight_node_pair, (w, nbr))

    nbr_edge_weight_sum = sum([x[0] for x in heap_weight_node_pair])

    if not neighbors:
        # No neighbor, all mass stay at node
        return [1], [node]

    if nbr_edge_weight_sum > EPSILON:
        # Sum need to be not too small to prevent divided by zero
        distributions = [(1.0 - _alpha) * w / nbr_edge_weight_sum for w, _ in heap_weight_node_pair]
    else:
        # Sum too small, just evenly distribute to every neighbors
        logger.warning("Neighbor weight sum too small, list:", heap_weight_node_pair)
        distributions = [(1.0 - _alpha) / len(heap_weight_node_pair)] * len(heap_weight_node_pair)

    nbr = [x[1] for x in heap_weight_node_pair]
    return distributions + [_alpha], nbr + [node]
    # return distributions + [_alpha], nbr


def _distribute_densities(source, target):
    # Distribute densities for source and source's neighbors as x
    t0 = time.time()

    if _Gk.isDirected():
        x, source_topknbr = _get_single_node_neighbors_distributions(source, "predecessors")
    else:
        x, source_topknbr = _get_single_node_neighbors_distributions(source, "successors")

    # Distribute densities for target and target's neighbors as y
    y, target_topknbr = _get_single_node_neighbors_distributions(target, "successors")

    logger.debug("%8f secs density distribution for edge." % (time.time() - t0))

    # construct the cost dictionary from x to y
    t0 = time.time()

    if _shortest_path == "pairwise":
        d = []
        for src in source_topknbr:
            tmp = []
            for tgt in target_topknbr:
                tmp.append(_source_target_shortest_path(src, tgt))
            d.append(tmp)
        d = np.array(d)
    else:  # all_pairs
        d = _apsp[np.ix_(source_topknbr, target_topknbr)]  # transportation matrix

    x = np.array(x)     # the mass that source neighborhood initially owned
    y = np.array(y)     # the mass that target neighborhood needs to received

    logger.debug("%8f secs density matrix construction for edge." % (time.time() - t0))

    return x, y, d, source_topknbr, target_topknbr


@lru_cache(_cache_maxsize)
def _source_target_shortest_path(source, target):

    length = nk.distance.BidirectionalDijkstra(_Gk, source, target).run().getDistance()
    assert length < 1e300, "Shortest path between %d, %d is not found" % (source, target)
    return length


def _get_all_pairs_shortest_path():
    """Pre-compute all pairs shortest paths of the assigned graph `_Gk`."""
    logger.trace("Start to compute all pair shortest path.")

    global _Gk

    t0 = time.time()
    apsp = nk.distance.APSP(_Gk).run().getDistances()
    logger.trace("%8f secs for all pair by NetworKit." % (time.time() - t0))

    return np.array(apsp)


def _optimal_transportation_distance(x, y, d):
    t0 = time.time()
    m = ot.emd2(x, y, d)
    logger.debug(
        "%8f secs for Wasserstein dist. \t#source_nbr: %d, #target_nbr: %d" % (time.time() - t0, len(x), len(y)))

    return m


def _new_transportation_distance(x, y, d, source_topknbr, target_topknbr):
    global _dirG
    
    _nbr_dict = dict()
    m = 0.
    pairs = np.ix_(source_topknbr, target_topknbr)
    target_node_sum = 0.
    total_flow = 0.
    activation = nn.ReLU()
    
    # print(f'pairs[0] = {pairs[0].shape}, pairs[1][0] = {pairs[1].shape}, d shape = {d.shape}, x shape = {x.shape}, y shape = {y.shape}')

    for i, s in enumerate(pairs[0]):
        assert(s[0] < len(_node_value[0]))
        # node_v = _node_value[0][s[0]].cpu().item()
        
        for j, t in enumerate(pairs[1][0]):
            assert(t < len(_node_value[0]))
            # m += (_node_value[0][s[0]].cpu().item() * d[i][j])
            # target_node_sum += (_node_value[0][t].cpu().item())
            target_node_sum = 0.
            
            path = nx.all_shortest_paths(_dirG, source=s[0], target=t)
            p_l = []
            for p in path:
                p_l.append(p)
            
            tmp = 0.
            for k in range(len(p_l[0]) - 1):
                # n = p_l[0][k+1] # next node
                # n_v = _node_value[0][n].cpu().item() # next node value
                # # if n in _nbr_dict.keys():
                # #     neighbors = _nbr_dict[n]
                # # else:
                # #     neighbors = list(_Gk.iterInNeighbors(n)) # in going neighbors of next node
                # #     _nbr_dict[n] = neighbors
                
                # neighbors = list(_Gk.iterInNeighbors(n)) # in going neighbors of next node
                # sum_after = 0.
                
                # for nbr in neighbors:
                #     if (nbr != p_l[0][k]):
                #         sum_after += _node_value[0][nbr].cpu().item() * _dirG[nbr][n]['weight']
                        
                # sum_after = activation(torch.tensor(sum_after))
                # c = np.abs(n_v - sum_after.item())
                
                # print(_apsp[p_l[0][k]][p_l[0][k+1]])
                target_node_sum += _node_value[0][p_l[0][k]].cpu().item()
                # m += (_node_value[0][p_l[0][k]].cpu().item() * _apsp[p_l[0][k]][p_l[0][k+1]])
                # m += ((_node_value[0][p_l[0][k]].cpu().item() * _apsp[p_l[0][k]][p_l[0][k+1]])/_node_value[0][p_l[0][k+1]].cpu().item())
            total_flow += target_node_sum
            m +=  (target_node_sum * d[i][j])
            
            # m += (node_v * d[i][j])
            # target_node_sum += _node_value[0][t].cpu().item()
            
    M = m/total_flow
    return M



def _compute_ricci_curvature_single_edge(source, target):

    # logger.debug("EDGE:%s,%s"%(source,target))
    assert source != target, "Self loop is not allowed."  # to prevent self loop

    # If the weight of edge is too small, return 0 instead.
    if _Gk.weight(source, target) < EPSILON:
        logger.trace("Zero weight edge detected for edge (%s,%s), return Ricci Curvature as 0 instead." %
                       (source, target))
        return {(source, target): 0}

    # compute transportation distance
    m = 1  # assign an initial cost
    assert _method in ["OTD", "ATD", "Sinkhorn", "OTDSinkhornMix", "custom"], \
        'Method %s not found, support method:["OTD", "ATD", "Sinkhorn", "OTDSinkhornMix", "custom"]' % _method
    if _method == "OTD":
        x, y, d, source_topknbr, target_topknbr = _distribute_densities(source, target)
        m = _optimal_transportation_distance(x, y, d)
    elif _method == "OTDSinkhornMix":
        x, y, d, source_topknbr, target_topknbr = _distribute_densities(source, target)
        # When x and y are small (usually around 2000 to 3000), ot.emd2 is way faster than ot.sinkhorn2
        # So we only do sinkhorn when both x and y are too large for ot.emd2
        m = _optimal_transportation_distance(x, y, d)
    
    # self-defined method
    elif _method == "custom":
        assert(_node_value != None)
        x, y, d, source_topknbr, target_topknbr = _distribute_densities(source, target)
        m = _new_transportation_distance(x, y, d, source_topknbr, target_topknbr)
        
    # compute Ricci curvature: k=1-(m_{x,y})/d(x,y)
    result = 1 - (m / _Gk.weight(source, target))  # Divided by the length of d(i, j)
    logger.debug("Ricci curvature (%s,%s) = %f" % (source, target, result))

    return {(source, target): result}


def _wrap_compute_single_edge(stuff):
    """Wrapper for args in multiprocessing."""
    return _compute_ricci_curvature_single_edge(*stuff)


def _compute_ricci_curvature_edges(G: nx.Graph, weight="weight", edge_list=[],
                                   alpha=0.5, method="OTDSinkhornMix",
                                   base=math.e, exp_power=2, proc=mp.cpu_count(), chunksize=None, cache_maxsize=1000000,
                                   shortest_path="all_pairs", nbr_topk=3000, nodes_v = None):

    logger.trace("Number of nodes: %d" % G.number_of_nodes())
    logger.trace("Number of edges: %d" % G.number_of_edges())

    if not nx.get_edge_attributes(G, weight):
        logger.info('Edge weight not detected in graph, use "weight" as default edge weight.')
        for (v1, v2) in G.edges():
            G[v1][v2][weight] = 1.0

    # ---set to global variable for multiprocessing used.---
    global _Gk
    global _alpha
    global _weight
    global _method
    global _base
    global _exp_power
    global _proc
    global _cache_maxsize
    global _shortest_path
    global _nbr_topk
    global _apsp
    global _node_value
    global _dirG
    # -------------------------------------------------------

    _Gk = nk.nxadapter.nx2nk(G, weightAttr=weight)
    _dirG = G
    _nbr_dict = dict()
    _alpha = alpha
    _weight = weight
    _method = method
    _base = base
    _exp_power = exp_power
    _proc = proc
    _cache_maxsize = cache_maxsize
    _shortest_path = shortest_path
    _nbr_topk = nbr_topk
    _node_value = nodes_v

    # Construct nx to nk dictionary
    nx2nk_ndict, nk2nx_ndict = {}, {}
    for idx, n in enumerate(G.nodes()):
        nx2nk_ndict[n] = idx
        nk2nx_ndict[idx] = n

    if _shortest_path == "all_pairs":
        # Construct the all pair shortest path dictionary
        # if not _apsp:
        _apsp = _get_all_pairs_shortest_path()

    if edge_list:
        args = [(nx2nk_ndict[source], nx2nk_ndict[target]) for source, target in edge_list]
    else:
        args = [(nx2nk_ndict[source], nx2nk_ndict[target]) for source, target in G.edges()]

    # Start compute edge Ricci curvature
    t0 = time.time()

    with mp.get_context('fork').Pool(processes=_proc) as pool:
        # WARNING: Now only fork works, spawn will hang.

        # Decide chunksize following method in map_async
        if chunksize is None:
            chunksize, extra = divmod(len(args), proc * 4)
            if extra:
                chunksize += 1

        # Compute Ricci curvature for edges
        result = pool.imap_unordered(_wrap_compute_single_edge, args, chunksize=chunksize)
        pool.close()
        pool.join()

    # Convert edge index from nk back to nx for final output
    output = {}
    for rc in result:
        for k in list(rc.keys()):
            output[(nk2nx_ndict[k[0]], nk2nx_ndict[k[1]])] = rc[k]

    logger.info("%8f secs for Ricci curvature computation." % (time.time() - t0))

    return output


def _compute_ricci_curvature(G: nx.Graph, weight="weight", **kwargs):

    # compute Ricci curvature for all edges
    edge_ricci = _compute_ricci_curvature_edges(G, weight=weight, **kwargs)

    # Assign edge Ricci curvature from result to graph G
    nx.set_edge_attributes(G, edge_ricci, "ricciCurvature")

    # Compute node Ricci curvature
    for n in G.nodes():
        rc_sum = 0  # sum of the neighbor Ricci curvature
        if G.degree(n) != 0:
            for nbr in G.neighbors(n):
                if 'ricciCurvature' in G[n][nbr]:
                    rc_sum += G[n][nbr]['ricciCurvature']

            # Assign the node Ricci curvature to be the average of node's adjacency edges
            G.nodes[n]['ricciCurvature'] = rc_sum / G.degree(n)
            logger.debug("node %s, Ricci Curvature = %f" % (n, G.nodes[n]['ricciCurvature']))

    return G



class OllivierRicci:
    """A class to compute Ollivier-Ricci curvature for all nodes and edges in G.
    Node Ricci curvature is defined as the average of all it's adjacency edge.

    """

    def __init__(self, G: nx.Graph, weight="weight", alpha=0.5, method="OTDSinkhornMix",
                 base=math.e, exp_power=2, proc=mp.cpu_count(), chunksize=None, shortest_path="all_pairs",
                 cache_maxsize=1000000,
                 nbr_topk=3000, verbose="ERROR", nodes_v = None):

        self.G = G.copy()
        self.alpha = alpha
        self.weight = weight
        self.method = method
        self.base = base
        self.exp_power = exp_power
        self.proc = proc
        self.chunksize = chunksize
        self.cache_maxsize = cache_maxsize
        self.shortest_path = shortest_path
        self.nbr_topk = nbr_topk
        self.nodes_v = nodes_v

        self.set_verbose(verbose)
        self.lengths = {}  # all pair shortest path dictionary
        self.densities = {}  # density distribution dictionary

        assert util.find_spec("ot"), \
            "Package POT: Python Optimal Transport is required for Sinkhorn distance."

        if not nx.get_edge_attributes(self.G, weight):
            logger.info('Edge weight not detected in graph, use "weight" as default edge weight.')
            for (v1, v2) in self.G.edges():
                self.G[v1][v2][weight] = 1.0

        self_loop_edges = list(nx.selfloop_edges(self.G))
        if self_loop_edges:
            logger.info('Self-loop edge detected. Removing %d self-loop edges.' % len(self_loop_edges))
            self.G.remove_edges_from(self_loop_edges)

    def set_verbose(self, verbose):
        set_verbose(verbose)


    def compute_ricci_curvature(self):
        self.G = _compute_ricci_curvature(G=self.G, weight=self.weight,
                                          alpha=self.alpha, method=self.method,
                                          base=self.base, exp_power=self.exp_power,
                                          proc=self.proc, chunksize=self.chunksize, cache_maxsize=self.cache_maxsize,
                                          shortest_path=self.shortest_path,
                                          nbr_topk=self.nbr_topk, nodes_v = self.nodes_v)
        return self.G



    def recal_graph_weight(self, nodes, weight="weight"):
        Gk = nk.nxadapter.nx2nk(self.G, weightAttr=weight)
        
        for n in list(self.G.nodes()):
            neighbors = list(Gk.iterInNeighbors(n))
            
            if not neighbors:
                continue
            
            sum = 0.
            pos_sum = 0.
            
            for nbr in neighbors:
                w = nodes[0][nbr].cpu().item()*self.G[nbr][n]['weight']
                sum += w
                if w > 0:
                    pos_sum += w
                    
            if sum > 0:  
                for nbr in neighbors:
                    if self.G[nbr][n]['weight'] < 0:
                        self.G.remove_edge(nbr, n)
                    else:
                        self.G[nbr][n]['weight'] = 1./(self.G[nbr][n]['weight'] * (sum/pos_sum))
                        # self.G[nbr][n]['weight'] = 1./(self.G[nbr][n]['weight'])
                        
            else:
                for nbr in neighbors:
                    self.G.remove_edge(nbr, n)
                    
                    
                    

    