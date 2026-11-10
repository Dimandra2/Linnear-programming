import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def draw_network(drivers, trips, costs, solution, filename="results/network.png"):
    os.makedirs("results", exist_ok=True)
    G = nx.DiGraph()
    for d in drivers: G.add_node(d[0])
    for t in trips: G.add_node(t[0])
    for (i, j), c in costs.items(): G.add_edge(drivers[i][0], trips[j][0])

    pos = {d[0]: (-1, -i) for i, d in enumerate(drivers)}
    pos.update({t[0]: (2, -j) for j, t in enumerate(trips)})

    plt.figure(figsize=(12, 8))
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=2000, node_shape='s')
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    nx.draw_networkx_edges(G, pos, edge_color='lightgray', width=1, arrowsize=15, alpha=0.6)

    if solution:
        selected_edges = [(drivers[i][0], trips[j][0]) for (i, j), k in solution.items() if round(k) == 1]
        nx.draw_networkx_edges(G, pos, edgelist=selected_edges, edge_color='red', width=3, arrowsize=20)
        edge_labels = {(drivers[i][0], trips[j][0]): f"{costs[(i, j)]} у.е."
                       for (i, j) in costs if round(solution.get((i, j), 0)) == 1}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10,
                                     font_color='red', label_pos=0.3,
                                     bbox=dict(alpha=0.8, edgecolor='none', facecolor='white'))
    plt.title("Оптимальное назначение")
    plt.savefig(filename, dpi=300)
    plt.close()


def draw_heatmap(drivers, trips, costs, filename="results/heatmap.png"):
    os.makedirs("results", exist_ok=True)
    n_d, n_t = len(drivers), len(trips)
    matrix = np.full((n_d, n_t), np.nan)
    for (i, j), c in costs.items():
        matrix[i, j] = c

    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt=".0f", cmap="YlGnBu", cbar_kws={'label': 'Стоимость (у.е.)'})
    plt.title("Матрица стоимостей назначений")
    plt.ylabel("Водители")
    plt.xlabel("Рейсы")
    plt.xticks(np.arange(n_t) + 0.5, [t[0] for t in trips])
    plt.yticks(np.arange(n_d) + 0.5, [d[0] for d in drivers], rotation=0)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()