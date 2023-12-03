import numpy as np
import maxflow
from itertools import product

# Rest of your provided functions remains unchanged...


def default_unary_potential(node_value, index):
    return node_value, 1 - node_value


def get_unary_log_odds(values):
    source_cost, sink_cost = np.clip(values, 0.01, 1)
    log_odds = np.log(source_cost) - np.log(sink_cost)
    return log_odds


def default_pairwise_potential(value1, value2, idx1, idx2):
    return 0.2
    return 1 - abs(value1 - value2)


def graph_cut_mrf_maxflow(image, unary_potential_fnc, pairwise_potential_fnc):
    height, width = image.shape
    g = maxflow.Graph[float](height, width)
    nodeids = g.add_grid_nodes(image.shape)

    # Add unary relationships (t-links)
    for i, j in product(range(height), range(width)):
        node = (i, j)
        intensity = image[i][j]
        log_odds = get_unary_log_odds(unary_potential_fnc(intensity, node))

        if log_odds > 0:
            g.add_tedge(nodeids[i, j], log_odds, 0)
        else:
            g.add_tedge(nodeids[i, j], 0, -log_odds)

    # Add pairwise relationships (n-links)
    for i, j in product(range(height), range(width)):
        if i > 0:  # Link to the pixel above
            tgt = (i - 1, j)
            pairwise_value = pairwise_potential_fnc(
                image[i, j], image[tgt], (i, j), tgt
            )
            g.add_edge(nodeids[i, j], nodeids[tgt], pairwise_value, pairwise_value)

        if j > 0:  # Link to the pixel to the left
            tgt = (i, j - 1)
            pairwise_value = pairwise_potential_fnc(
                image[i, j], image[tgt], (i, j), tgt
            )
            g.add_edge(nodeids[i, j], nodeids[tgt], pairwise_value, pairwise_value)

    # Compute the maxflow
    g.maxflow()

    # Get the segments
    sgm = g.get_grid_segments(nodeids)

    # The result is a binary image
    return np.int_(np.logical_not(sgm))


def segment_image_maxflow(image, unary_potential_fnc=None, pairwise_potential_fnc=None):
    if unary_potential_fnc is None:
        unary_potential_fnc = default_unary_potential
    if pairwise_potential_fnc is None:
        pairwise_potential_fnc = default_pairwise_potential

    return graph_cut_mrf_maxflow(image, unary_potential_fnc, pairwise_potential_fnc)


# Usage example:
# image = ...  # Your binary image data here
# segmented_image = segment_image_maxflow(image)
