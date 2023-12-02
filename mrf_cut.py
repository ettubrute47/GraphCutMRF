from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.flow import shortest_augmenting_path
from networkx.algorithms.flow import boykov_kolmogorov


def default_unary_potential(node_value, index):
    return node_value, 1 - node_value


def get_unary_log_odds(values):
    source_cost, sink_cost = np.clip(values, 0.01, 1)
    log_odds = np.log(source_cost) - np.log(sink_cost)
    return log_odds


def default_pairwise_potential(value1, value2, idx1, idx2):
    return 1


def get_pos(G, shape):
    pos = {}
    height, width = shape

    for node in G.nodes():
        if node == "source":
            pos[node] = (-1, height / 2)
        elif node == "sink":
            pos[node] = (width, height / 2)
        else:
            i, j = node
            pos[node] = (j, height - i - 1)
    return pos


def create_pairwise_mrf(image, unary_potential_fnc=None, pairwise_potential_fnc=None):
    image = np.asarray(image)

    if unary_potential_fnc is None:
        unary_potential_fnc = default_unary_potential
    if pairwise_potential_fnc is None:
        pairwise_potential_fnc = default_pairwise_potential

    G = nx.Graph(shape=image.shape, pos={})

    G.add_nodes_from(["source", "sink"])

    for i, j in product(range(image.shape[0]), range(image.shape[1])):
        intensity = image[i][j]
        node = (i, j)
        log_odds = get_unary_log_odds(unary_potential_fnc(intensity, node))
        G.add_node(node, value=intensity, log_odds=log_odds)

        # add t-links, for exact binary you can just add an edge to source if log odds > 0, else add to sink
        if log_odds > 0:
            G.add_edge("source", node, capacity=log_odds)
        else:
            G.add_edge(node, "sink", capacity=-log_odds)

    # add n-links between pixels
    for i, j in product(range(image.shape[0]), range(image.shape[1])):
        src = (i, j)
        value = G.nodes[src]["value"]
        if i > 0:  # add edge left
            tgt = (i - 1, j)
            pairwise_value = pairwise_potential_fnc(value, image[tgt], src, tgt)
            G.add_edge(src, tgt, capacity=pairwise_value)
        if j > 0:  # add edge below
            tgt = (i, j - 1)
            pairwise_value = pairwise_potential_fnc(value, image[tgt], src, tgt)
            G.add_edge(src, tgt, capacity=pairwise_value)

    # save positioning
    G.graph["pos"] = get_pos(G, image.shape)
    G.graph["image"] = np.array(image)

    return G


def graph_cut_mrf(mrf: nx.Graph):
    # minimum_cut/maximum_flow partition nodes also minimizes energy of mrf
    _, cut = nx.minimum_cut(mrf, "source", "sink", flow_func=boykov_kolmogorov)
    image = np.zeros(mrf.graph["shape"])
    for node in cut[0]:
        if node != "source":
            image[node] = 1
    return image


def segment_image(image, unary_potential_fnc=None, pairwise_potential_fnc=None):
    G = create_pairwise_mrf(image, unary_potential_fnc, pairwise_potential_fnc)
    return graph_cut_mrf(G)
