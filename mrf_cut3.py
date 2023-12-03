import numpy as np
import gco
from itertools import product

# The rest of your functions (default_unary_potential, get_unary_log_odds, etc.) remain unchanged...


def default_unary_potential(node_value, index):
    return node_value, 1 - node_value


def get_unary_log_odds(values):
    source_cost, sink_cost = np.clip(values, 0.01, 1)
    log_odds = np.log(source_cost) - np.log(sink_cost)
    return log_odds


def default_pairwise_potential(value1, value2, idx1, idx2):
    return 0.2
    return 1 - abs(value1 - value2)


def graph_cut_mrf_gco(image, unary_potential_fnc, pairwise_potential_fnc):
    height, width = image.shape
    num_labels = 2

    # Create the graph
    gc = gco.GCO()
    gc.create_general_graph(height * width, num_labels, True)

    # Unary potentials
    print("Getting unary potentials")
    unary_potentials = np.zeros((height * width, num_labels), dtype=np.float32)
    for i, j in product(range(height), range(width)):
        node = i * width + j
        intensity = image[i, j]
        unary_potentials[node] = unary_potential_fnc(intensity, (i, j))
    print("Done getting unary potentials")

    # Apply unary potentials to the graph
    gc.set_data_cost(unary_potentials)

    # Pairwise potentials
    pairwise = np.zeros((num_labels, num_labels), dtype=np.float32)
    for label1 in range(num_labels):
        for label2 in range(num_labels):
            pairwise[label1, label2] = pairwise_potential_fnc(
                label1, label2, None, None
            )

    # Apply pairwise potentials to the graph
    gc.set_smooth_cost(pairwise)

    # Add edges between pixels with the given weights
    for i, j in product(range(height), range(width)):
        node = i * width + j
        if i + 1 < height:
            down_node = (i + 1) * width + j
            weight = pairwise_potential_fnc(
                image[i, j], image[i + 1, j], (i, j), (i + 1, j)
            )
            gc.set_neighbor_pair(node, down_node, weight)
        if j + 1 < width:
            right_node = i * width + (j + 1)
            weight = pairwise_potential_fnc(
                image[i, j], image[i, j + 1], (i, j), (i, j + 1)
            )
            gc.set_neighbor_pair(node, right_node, weight)
    print("Getting ready to cut")
    # Compute the cut
    gc.expansion(1)  # Number of iterations can be adjusted

    # Get the labels and reshape to the image shape
    labels = gc.get_labels()
    segmentation = labels.reshape((height, width))

    return segmentation


def segment_image_gco(image, unary_potential_fnc=None, pairwise_potential_fnc=None):
    if unary_potential_fnc is None:
        unary_potential_fnc = default_unary_potential
    if pairwise_potential_fnc is None:
        pairwise_potential_fnc = default_pairwise_potential

    return graph_cut_mrf_gco(image, unary_potential_fnc, pairwise_potential_fnc)


# Usage example:
# image = ...  # Your binary image data here
# segmented_image = segment_image_gco(image)
