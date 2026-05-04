import torch
import matplotlib.pyplot as plt
import networkx as nx

from dataloader import get_dataloaders
from baseline import ErdosRenyiBaseline


def adjacency_to_networkx(A):
    """Convert adjacency matrix to NetworkX graph"""
    return nx.from_numpy_array(A.numpy())


def main():
    # =========================================================
    # Load data
    # =========================================================
    train_loader, val_loader, test_loader = get_dataloaders()

    # ⚠️ Bitno: treba ti dataset, ne loader
    train_dataset = train_loader.dataset

    # =========================================================
    # Init model
    # =========================================================
    model = ErdosRenyiBaseline(train_dataset)

    # =========================================================
    # Sample graphs
    # =========================================================
    num_samples = 5

    print("\n=== Sampling graphs ===\n")

    samples = []

    for i in range(num_samples):
        A = model.sample()
        samples.append(A)

        N = A.shape[0]
        E = int(A.sum().item() / 2)  # undirected
        max_E = N * (N - 1) / 2
        density = E / max_E if max_E > 0 else 0

        print(f"Graph {i+1}")
        print(f"  Nodes: {N}")
        print(f"  Edges: {E}")
        print(f"  Density: {density:.3f}\n")

    # =========================================================
    # Visualize one graph
    # =========================================================
    A = samples[0]
    G = adjacency_to_networkx(A)

    plt.figure(figsize=(5, 5))
    nx.draw(G, node_size=100, with_labels=False)
    plt.title("Erdős–Rényi Sample")

    # Save instead of show
    plt.savefig("er_sample.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()