import torch
import matplotlib
matplotlib.use("Agg")  # HPC safe

import matplotlib.pyplot as plt
import os

from dataloader import get_dataloaders
from baseline import ErdosRenyiBaseline


# =========================================================
# Helpers
# =========================================================
def compute_stats_from_dataset(dataset):
    node_counts = []
    densities = []

    for data in dataset:
        N = data.num_nodes
        E = data.num_edges // 2

        max_E = N * (N - 1) / 2
        density = E / max_E if max_E > 0 else 0

        node_counts.append(N)
        densities.append(density)

    return node_counts, densities


def compute_stats_from_generated(samples):
    node_counts = []
    densities = []

    for A in samples:
        N = A.shape[0]
        E = int(A.sum().item() / 2)

        max_E = N * (N - 1) / 2
        density = E / max_E if max_E > 0 else 0

        node_counts.append(N)
        densities.append(density)

    return node_counts, densities


def plot_histogram(train_vals, gen_vals, title, filename):
    plt.figure()

    plt.hist(train_vals, bins=20, alpha=0.6, label="Train", density=True)
    plt.hist(gen_vals, bins=20, alpha=0.6, label="Generated", density=True)

    plt.legend()
    plt.title(title)

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# Main
# =========================================================
def main():
    os.makedirs("plots", exist_ok=True)

    # Load data
    train_loader, _, _ = get_dataloaders()
    train_dataset = train_loader.dataset

    # Init model
    model = ErdosRenyiBaseline(train_dataset)

    # =====================================================
    # Compute TRAIN statistics
    # =====================================================
    train_nodes, train_density = compute_stats_from_dataset(train_dataset)

    # =====================================================
    # Generate samples
    # =====================================================
    num_samples = 1000

    print(f"Generating {num_samples} graphs...")

    samples = [model.sample() for _ in range(num_samples)]

    gen_nodes, gen_density = compute_stats_from_generated(samples)

    # =====================================================
    # Plot comparisons
    # =====================================================
    print("Saving plots...")

    plot_histogram(
        train_nodes,
        gen_nodes,
        "Node Count Distribution",
        "plots/node_distribution.png",
    )

    plot_histogram(
        train_density,
        gen_density,
        "Density Distribution",
        "plots/density_distribution.png",
    )

    print("Done! Plots saved in ./plots/")


if __name__ == "__main__":
    main()