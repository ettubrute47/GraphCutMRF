from typing import Callable
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.offline import iplot
from skimage.transform import resize


def get_node_labels(G):
    node_labels = {}
    for node in G.nodes():
        if node == "source":
            node_labels[node] = "source"
        elif node == "sink":
            node_labels[node] = "sink"
        else:
            node_labels[node] = "{:.2f}".format(G.nodes[node]["value"])
    return node_labels


def get_edge_labels(G):
    edge_labels = {}
    for node1, node2, data in G.edges(data=True):
        if "capacity" in data:
            edge_labels[(node1, node2)] = "{:.2f}".format(data["capacity"])
    return edge_labels


def resolve_maybe_fnc(fnc, *args, default=None):
    if fnc is None:
        fnc = default
    return fnc(*args) if isinstance(fnc, Callable) else fnc


def visualize_graph(
    G: nx.Graph,
    node_label_fnc=None,
    edge_label_fnc=None,
    node_color_fnc=None,
    edge_color_fnc=None,
    title=None,
):
    pos = G.graph["pos"]

    resolve_maybe_fnc(node_color_fnc, G, default="skyblue")
    node_color = resolve_maybe_fnc(node_color_fnc, G, default="skyblue")
    nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=500)

    edge_color = resolve_maybe_fnc(edge_color_fnc, G, default="black")
    nx.draw_networkx_edges(G, pos, edge_color=edge_color)

    node_labels = resolve_maybe_fnc(node_label_fnc, G, default=get_node_labels)
    nx.draw_networkx_labels(
        G, pos, labels=node_labels, font_size=10, verticalalignment="bottom"
    )

    edge_labels = resolve_maybe_fnc(edge_label_fnc, G, default=get_edge_labels)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    title = title or "MRF"
    plt.title(title)
    plt.axis("off")
    plt.show()


def get_edge_traces(G: nx.Graph, pos):
    traces = []
    for u, v, data in G.edges(data=True):
        x, y, z = zip(pos[u], pos[v])
        edge_trace = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            line=dict(width=data["capacity"] * 5, color="#888"),
            opacity=0.4,
            mode="lines",
        )
        traces.append(edge_trace)
        edge_trace = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            line=dict(width=data["capacity"] * 2, color="#FF0000"),
            mode="lines",
        )
        traces.append(edge_trace)
    return traces


def viz_graph_3d(G: nx.Graph):
    # 3D spring layout
    # pos is the same except source and sink are at different depths

    pos = dict()
    for node, (x, y) in G.graph["pos"].items():
        z = {"source": 1, "sink": -1}.get(node, 0)
        if z != 0:
            w, h = G.graph["shape"]
            pos[node] = (z, w // 2, h // 2)
        else:
            pos[node] = (z, x, y)

    edge_traces = get_edge_traces(G, pos)

    node_x = []
    node_y = []
    node_z = []
    for node in G.nodes():
        x, y, z = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)

    node_trace = go.Scatter3d(
        x=node_x,
        y=node_y,
        z=node_z,
        mode="markers",
        hoverinfo="text",
        marker=dict(size=10),
    )

    image = np.random.rand(3, 3)
    image = np.ones((3, 3))
    image[:, 1] = 0

    # Normalize the image array if it isn't already (i.e., scale from 0 to 1)
    image_normalized = (image - np.min(image)) / (np.max(image) - np.min(image))

    # Create a colorscale based on the grayscale image array
    colorscale = []
    for i, row in enumerate(image_normalized):
        for j, value in enumerate(row):
            # Convert the grayscale value to a color
            grayscale_color = str(int(value * 255))
            color = "rgb({0}, {0}, {0})".format(grayscale_color)

            # The position of the color in the colorscale
            position = (i * 3 + j) / (len(image_normalized) * len(row) - 1)
            colorscale.append([position, color])

    # Create the x and y grid
    N = 30
    image = resize(
        G.graph["image"],
        (N, N),
        preserve_range=True,
        anti_aliasing=False,
        mode="edge",
        order=0,
    )

    plt.imshow(image)
    plt.show()
    x_range = np.linspace(min(node_x) - 0.5, max(node_x) + 0.5, num=N)
    y_range = np.linspace(min(node_y) - 0.5, max(node_y) + 0.5, num=N)
    z_range = np.linspace(min(node_z) - 0.5, max(node_z) + 0.5, num=N)
    Y, Z = np.meshgrid(y_range, z_range)

    # Assuming all your nodes lie on the same plane, you can use one z-value for all
    # If they are not on the same plane, you would need to interpolate the z-values for the grid
    x_value = np.mean(node_x)  # This is a simplification, adjust accordingly
    X = np.full_like(Y, x_value)

    colorscale = [[0, "rgb(0,0,0)"], [1, "rgb(255,255,255)"]]
    # how do I upscale a 3x3 image?

    # colorscale = [
    #     [0, "rgb(0,0,0)"],
    #     [0.49, "rgb(0,0,0)"],
    #     [0.5, "rgb(255,255,255)"],
    #     [1, "rgb(255,255,255)"],
    # ]

    image_trace = go.Surface(x=X, y=Y, z=Z, colorscale=colorscale, surfacecolor=image)

    fig = go.Figure(
        data=[*edge_traces, node_trace, image_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            scene=dict(
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            ),
        ),
    )

    iplot(fig)
