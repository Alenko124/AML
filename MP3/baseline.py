import torch
from collections import defaultdict
from torch_geometric.data import Data


class ErdosRenyiBaseline:
    """
    Erdős–Rényi baseline model G(N, r).

    Steps:
    1. Sample N from empirical distribution of training graphs.
    2. Estimate edge probability r (graph density) conditioned on N.
    3. Sample random graph with N nodes and edge probability r.
    """

    def __init__(self, train_dataset):
        self.node_counts, self.density_per_N, self.global_density = \
            self._compute_training_statistics(train_dataset)

    # =========================================================
    # STEP 1: Compute statistics from training data
    # =========================================================
    def _compute_training_statistics(self, dataset):
        node_counts = []
        densities_per_N = defaultdict(list)

        for data in dataset:
            N = data.num_nodes

            # PyG stores undirected edges twice → divide by 2
            E = data.num_edges // 2

            max_edges = N * (N - 1) / 2
            density = E / max_edges if max_edges > 0 else 0.0

            node_counts.append(N)
            densities_per_N[N].append(density)

        node_counts = torch.tensor(node_counts)

        # Average density per N
        density_per_N = {
            N: sum(vals) / len(vals)
            for N, vals in densities_per_N.items()
        }

        # Global fallback density
        all_densities = [d for vals in densities_per_N.values() for d in vals]
        global_density = sum(all_densities) / len(all_densities)

        return node_counts, density_per_N, global_density

    # =========================================================
    # STEP 2: Sample N
    # =========================================================
    def _sample_N(self):
        idx = torch.randint(len(self.node_counts), (1,))
        return int(self.node_counts[idx])

    # =========================================================
    # STEP 3: Get density r
    # =========================================================
    def _get_density(self, N):
        if N in self.density_per_N:
            return self.density_per_N[N]
        else:
            return self.global_density  # fallback

    # =========================================================
    # STEP 4: Sample graph G(N, r)
    # =========================================================
    def _sample_graph(self, N, r):
        """
        Sample adjacency matrix A ~ G(N, r)

        Returns
        -------
        A : torch.Tensor (N x N)
            Symmetric adjacency matrix with zeros on diagonal
        """

        # Initialize empty adjacency matrix
        A = torch.zeros((N, N), dtype=torch.float32)

        # Fill upper triangle
        for i in range(N):
            for j in range(i + 1, N):
                if torch.rand(1).item() < r:
                    A[i, j] = 1.0
                    A[j, i] = 1.0  # ensure symmetry

        return A

    # =========================================================
    # PUBLIC API
    # =========================================================
    def sample(self):
        """
        Sample one graph from the model.
        """
        N = self._sample_N()
        r = self._get_density(N)
        return self._sample_graph(N, r)

    def sample_batch(self, num_graphs):
        """
        Sample multiple graphs.
        """
        return [self.sample() for _ in range(num_graphs)]