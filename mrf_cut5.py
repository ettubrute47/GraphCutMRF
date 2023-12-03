from itertools import product
import numpy as np
from numba import jit
import igraph as ig
import matplotlib.pyplot as plt
import time
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)


def default_unary_potential(node_value):
    return node_value, 1 - node_value


def get_unary_log_odds(values):
    source_cost, sink_cost = np.clip(values, 0.01, 1)
    log_odds = np.log(source_cost) - np.log(sink_cost)
    return log_odds


def default_pairwise_potential(value1, value2, idx1, idx2):
    return 0.2
    return 1 - abs(value1 - value2)


def create_pairwise_mrf_igraph(
    image, unary_potential_fnc=None, pairwise_potential_fnc=None
):
    image = np.asarray(image)

    if unary_potential_fnc is None:
        unary_potential_fnc = default_unary_potential
    if pairwise_potential_fnc is None:
        pairwise_potential_fnc = default_pairwise_potential

    height, width = image.shape
    num_pixels = height * width
    num_nodes = num_pixels + 2  # +2 for source and sink

    # Create a new graph
    g = ig.Graph(n=num_nodes, directed=True)
    node_names = ["source", "sink"] + [
        (i, j) for i in range(height) for j in range(width)
    ]
    g.vs["name"] = node_names

    # Preallocate edge and capacity lists
    edges = []
    capacities = []

    # Add source and sink edges with capacities
    for index, (i, j) in enumerate(product(range(height), range(width))):
        node_index = index + 2  # account for source and sink
        intensity = image[i, j]
        log_odds = get_unary_log_odds(unary_potential_fnc(intensity, (i, j)))

        if log_odds > 0:
            edges.append(("source", node_index))
            capacities.append(log_odds)
        else:
            edges.append((node_index, "sink"))
            capacities.append(-log_odds)

    # Add n-links with capacities
    for i in range(height):
        for j in range(width):
            src_index = i * width + j + 2
            value = image[i, j]

            if i > 0:
                tgt_index = (i - 1) * width + j + 2
                pairwise_value = pairwise_potential_fnc(
                    value, image[i - 1, j], (i, j), (i - 1, j)
                )
                edges.append((src_index, tgt_index))
                capacities.append(pairwise_value)

            if j > 0:
                tgt_index = i * width + (j - 1) + 2
                pairwise_value = pairwise_potential_fnc(
                    value, image[i, j - 1], (i, j), (i, j - 1)
                )
                edges.append((src_index, tgt_index))
                capacities.append(pairwise_value)

    # Add all edges and capacities at once
    g.add_edges(edges)
    g.es["capacity"] = capacities

    return g


@jit(nopython=True)
def build_edges_and_capacities(height, width, unary_potentials, pairwise_potentials):
    edges = []
    capacities = []

    # Add edges and capacities for t-links (unary potentials)
    for idx, log_odds in enumerate(unary_potentials):
        if log_odds > 0:
            edges.append((0, idx + 2))  # Edge from source
            capacities.append(log_odds)
        else:
            edges.append((idx + 2, 1))  # Edge to sink
            capacities.append(-log_odds)

    # Add edges and capacities for n-links (pairwise potentials)
    for i in range(height):
        for j in range(width):
            idx = i * width + j + 2
            if i > 0:  # edge to pixel above
                edges.append((idx, idx - width))
                capacities.append(pairwise_potentials[idx - 2][0])
            if j > 0:  # edge to pixel to the left
                edges.append((idx, idx - 1))
                capacities.append(pairwise_potentials[idx - 2][1])

    return edges, capacities


def create_pairwise_mrf_igraph(
    image, unary_potential_fnc=None, pairwise_potential_fnc=None
):
    image = np.asarray(image)
    height, width = image.shape
    num_nodes = width * height + 2  # +2 for source and sink

    if unary_potential_fnc is None:
        unary_potential_fnc = default_unary_potential
    if pairwise_potential_fnc is None:
        pairwise_potential_fnc = default_pairwise_potential
        # print("USING DEFAULT PAIRWISE?!")

    # print("Computing potentials")
    start = time.perf_counter()
    # Pre-compute all unary potentials and pairwise potentials
    unary_potentials = get_unary_log_odds(unary_potential_fnc(image)).flatten()
    # unary_potentials = get_unary_log_odds(
    #     np.array(
    #         [
    #             unary_potential_fnc(image[i, j], (i, j))
    #             for i in range(height)
    #             for j in range(width)
    #         ]
    #     )
    # )
    # print("Done computing unary ", time.perf_counter() - start)
    pairwise_potentials = np.array(
        [
            [
                pairwise_potential_fnc(image[i, j], image[i - 1, j], (i, j), (i - 1, j))
                if i > 0
                else 0,
                pairwise_potential_fnc(image[i, j], image[i, j], (i, j), (i, j - 1))
                if j > 0
                else 0,
            ]
            for i in range(height)
            for j in range(width)
        ]
    )
    # print("Done computing potentials ", time.perf_counter() - start)

    # Use numba to build edges and capacities
    edges, capacities = build_edges_and_capacities(
        height, width, unary_potentials, pairwise_potentials
    )

    # Create the graph with the pre-built edge list and capacity list
    g = ig.Graph(num_nodes, directed=False)
    g.add_edges(edges, {"capacity": capacities})
    # g.es["capacity"] = capacities
    g.vs["name"] = ["source", "sink"] + [
        (i, j) for i in range(height) for j in range(width)
    ]
    g["shape"] = image.shape
    return g


@jit(nopython=True)
def construct_segmentation(image_shape, partition_indices):
    segmentation = np.zeros(image_shape, dtype=np.uint8)
    height, width = image_shape
    for idx in partition_indices:
        i, j = divmod(idx, width)
        segmentation[i, j] = 1
    return segmentation


def graph_cut_mrf_igraph(mrf: ig.Graph):
    # Compute the minimum s-t cut
    # print("Finding indices")
    srcidx = 0
    sinkidx = 1
    # print("Preparing min cut")
    start = time.perf_counter()

    mincut = mrf.st_mincut(srcidx, sinkidx, "capacity")
    # print("Done min cut in ", time.perf_counter() - start)
    partition = mincut.partition[0]
    image_shape = mrf["shape"]

    # print("Building partition to return")
    # Convert partition to a numpy array of indices
    # partition_indices = np.array(
    #     [mrf.vs["name"].index(vertex) for vertex in partition if vertex != "source"]
    # )
    # # Adjust indices to account for source and sink not being part of the image
    # partition_indices -= 2
    # image_shape = mrf["shape"]
    # Use the JIT-compiled function to construct the segmentation
    segmentation = construct_segmentation(image_shape, partition)
    return segmentation


def plot_igraph(g):
    # layout = g.layout("fr")  # Use a grid layout; 'fr' stands for Fruchterman-Reingold

    # Define visual style
    visual_style = {}
    visual_style["vertex_size"] = 20
    visual_style["vertex_label"] = g.vs["name"]
    width, height = g["shape"]
    num_nodes = width * height + 2  # +2 for source and sink

    # Create a grid layout for the image nodes
    grid_layout = [(j, height - i - 1) for i in range(height) for j in range(width)]

    # Add positions for the source and sink nodes
    # Place the source at the top middle and the sink at the bottom middle
    grid_layout = [(width // 2, height)] + [(width // 2, -1)] + grid_layout

    # Set 'grid_layout' as the layout
    visual_style["layout"] = grid_layout
    visual_style["bbox"] = (600, 600)  # Size of the plot
    visual_style["margin"] = 20  # Margin around the plot

    # If the graph is too large, you might want to skip the labels to avoid clutter
    if len(g.vs) > 100:
        del visual_style["vertex_label"]
    # Define edge colors based on capacity
    max_capacity = max(g.es["capacity"]) if g.es["capacity"] else 1
    # g.es["color"] = [
    #     "rgb(0,0,255)" if cap > max_capacity / 2 else "rgb(255,0,0)"
    #     for cap in g.es["capacity"]
    # ]
    # g.es["width"] = [cap / max_capacity * 10 for cap in g.es["capacity"]]

    # # Update visual style
    # visual_style["edge_color"] = g.es["color"]
    # visual_style["edge_width"] = g.es["width"]

    # Draw the graph with the updated style
    fig, ax = plt.subplots()
    ig.plot(g, target=ax, **visual_style)
    plt.show()


def segment_image_igraph(image, unary_potential_fnc=None, pairwise_potential_fnc=None):
    # print("Construcint graph")
    G = create_pairwise_mrf_igraph(image, unary_potential_fnc, pairwise_potential_fnc)
    # plot_igraph(G)
    # print("Done construcint graph")
    G["shape"] = image.shape  # Store shape as graph attribute
    return graph_cut_mrf_igraph(G)


if __name__ == "__main__":
    img = np.random.rand(3, 3)
    guess = segment_image_igraph(img, pairwise_potential_fnc=lambda *x: 100)
    print(guess)
