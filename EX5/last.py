import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt


def G1(x):
    """
    Evaluate the metric G(x) = (1 + |x|^2) I for torch tensor x of shape Nx2.
    The result has shape Nx2x2
    """
    N, D = x.shape
    I = torch.eye(D, D).reshape(1, D, D).repeat(N, 1, 1)  # NxDxD
    alpha = 1 + torch.sum(x**2, dim=1)  # N
    G = alpha.reshape(N, 1, 1) * I  # NxDxD
    return G


def G2(x, data, sigma=0.1):
    """
    Evaluate the metric 1/p(x) * I, where p(x) is a Gaussian kernel density estimate.
    The result has shape Nx2x2
    """
    N, D = x.shape
    M, D = data.shape
    sigma2 = sigma**2
    normalization = (2 * 3.14159) ** (D / 2) * sigma**D  # scalar
    I = torch.eye(D, D).reshape(1, D, D)  # 1xDxD

    Gs = []
    for n in range(N):
        xn = x[n].reshape(1, D)  # 1xD
        delta = xn - data  # MxD
        K = torch.exp(-0.5 * torch.sum(delta**2, dim=1) / sigma2) / normalization  # M
        pn = K.sum() / M  # scalar
        Gs.append(I / (pn + 1e-4))

    G = torch.concatenate(Gs, dim=0)  # NxDxD
    return G


def plot_metric(metric, range):
    X, Y = torch.meshgrid(range, range, indexing="ij")
    XY = torch.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1)), dim=1)
    G = metric(XY)  # NxDxD
    trG = G[:, 0, 0] + G[:, 1, 1]  # N

    plt.imshow(
        trG.reshape(X.shape).detach().numpy().T,
        extent=(range[0], range[-1], range[0], range[-1]),
        origin="lower",
    )


class PLcurve:
    def __init__(self, x0, x1, N):
        """
        Represent the piecewise linear curve connecting x0 to x1 using N nodes
        """
        super().__init__()
        self.x0 = x0.reshape(1, -1)
        self.x1 = x1.reshape(1, -1)
        self.N = N

        t = torch.linspace(0, 1, N).reshape(N, 1)
        c = (1 - t) @ self.x0 + t @ self.x1
        self.params = c[1:-1]
        self.params.requires_grad = True

    def points(self):
        return torch.concatenate((self.x0, self.params, self.x1), dim=0)

    def plot(self):
        c = self.points().detach().numpy()
        plt.plot(c[:, 0], c[:, 1])


def curve_energy(metric, curve):
    G = metric(curve[:-1])  # (N-1)xDxD
    delta = curve[1:] - curve[:-1]  # (N-1)xD
    tmp = torch.bmm(G, delta.unsqueeze(-1)).squeeze(-1)
    energy = torch.sum(delta * tmp)
    return energy


def connecting_geodesic(metric, curve):
    opt = optim.LBFGS([curve.params], lr=0.5)

    def closure():
        opt.zero_grad()
        energy = curve_energy(metric, curve.points())
        energy.backward()
        return energy

    for _ in range(1000):
        opt.step(closure)


# -------- MAIN --------
if __name__ == "__main__":
    metric_type = "density"  # or "density"

    if metric_type == "quadratic":
        r = 5
        plot_metric(G1, torch.linspace(-r, r, 100))

        N = 20
        for _ in range(5):
            c = PLcurve(
                2 * r * (torch.rand(2) - 0.5),
                2 * r * (torch.rand(2) - 0.5),
                N,
            )

            print("Energy before:", curve_energy(G1, c.points()).item())
            connecting_geodesic(G1, c)
            print("Energy after:", curve_energy(G1, c.points()).item())

            c.plot()

        plt.axis((-r, r, -r, r))
        plt.savefig("geodesic_quadratic.png")

    elif metric_type == "density":
        data = torch.from_numpy(np.load("toybanana.npy")).float()

        r = 3
        G = lambda x: G2(x, data)

        plot_metric(G, torch.linspace(-r, r, 100))
        plt.scatter(data[:, 0], data[:, 1], s=1)

        T = 20
        for _ in range(10):
            idx = torch.randint(data.shape[0], (2,))
            c = PLcurve(data[idx[0]], data[idx[1]], T)

            print("Energy before:", curve_energy(G, c.points()).item())
            connecting_geodesic(G, c)
            print("Energy after:", curve_energy(G, c.points()).item())

            c.plot()

        plt.axis((-r, r, -r, r))
        plt.savefig("geodesic_density.png")