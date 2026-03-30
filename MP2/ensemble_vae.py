# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-01-27)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
#
# Significant extension by Søren Hauberg, 2024

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from tqdm import tqdm
from copy import deepcopy
import os
import math
import matplotlib.pyplot as plt

class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int]
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)


class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)


class GaussianDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(GaussianDecoder, self).__init__()
        self.decoder_net = decoder_net
        # self.std = nn.Parameter(torch.ones(28, 28) * 0.5, requires_grad=True) # In case you want to learn the std of the gaussian.

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor]
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        means = self.decoder_net(z)
        return td.Independent(td.Normal(loc=means, scale=1e-1), 3)


class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """

    def __init__(self, prior, decoder, encoder):
        """
        Parameters:
        prior: [torch.nn.Module]
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """

        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        q = self.encoder(x)
        z = q.rsample()

        log_qz = q.log_prob(z)
        log_pz = self.prior().log_prob(z)

        recons = self.decoder(z)  # lista

        recon_logprob = 0
        for recon in recons:
            recon_logprob += recon.log_prob(x)

        recon_logprob = recon_logprob / len(recons)

        elbo = torch.mean(recon_logprob - log_qz + log_pz)

        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model.

        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()

    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x)


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """

    num_steps = len(data_loader) * epochs
    epoch = 0

    def noise(x, std=0.05):
        eps = std * torch.randn_like(x)
        return torch.clamp(x + eps, min=0.0, max=1.0)

    with tqdm(range(num_steps)) as pbar:
        for step in pbar:
            try:
                x = next(iter(data_loader))[0]
                x = noise(x.to(device))
                model = model
                optimizer.zero_grad()
                # from IPython import embed; embed()
                loss = model(x)
                loss.backward()
                optimizer.step()

                # Report
                if step % 5 == 0:
                    loss = loss.detach().cpu()
                    pbar.set_description(
                        f"total epochs ={epoch}, step={step}, loss={loss:.1f}"
                    )

                if (step + 1) % len(data_loader) == 0:
                    epoch += 1
            except KeyboardInterrupt:
                print(
                    f"Stopping training at total epoch {epoch} and current loss: {loss:.1f}"
                )
                break



# =========================
# ENERGY FUNCTION (ENSEMBLE)
# =========================
def ensemble_energy(curve, decoders, num_samples=5):
    energy = 0

    for i in range(len(curve) - 1):
        z_i = curve[i:i+1]
        z_next = curve[i+1:i+2]

        for _ in range(num_samples):
            f_l = random.choice(decoders)
            f_k = random.choice(decoders)

            x_i = f_l(z_i).mean
            x_next = f_k(z_next).mean

            energy += ((x_i - x_next) ** 2).sum()

    return energy / num_samples

def geodesic_distance(decoders, z_start, z_end):


    # initialize straight line
    N = 20
    t = torch.linspace(0, 1, N).to(device).unsqueeze(1)

    curve = (1 - t) * z_start + t * z_end
    curve = curve.clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam([curve], lr=1e-2)

    # optimize curve
    for _ in range(100):
        loss = ensemble_energy(curve, decoders)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            curve[0] = z_start
            curve[-1] = z_end

    # final distance = energy
    with torch.no_grad():
        dist = ensemble_energy(curve, decoders)

    return dist.item()

class EnsembleDecoder(nn.Module):
    def __init__(self, num_decoders, decoder_fn):
        super().__init__()
        self.decoders = nn.ModuleList(
            [GaussianDecoder(decoder_fn()) for _ in range(num_decoders)]
        )

    def forward(self, z):
        return [decoder(z) for decoder in self.decoders]
    

if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        type=str,
        default="train",
        choices=["train", "sample", "eval", "geodesics", "save", "cov"],
        help="what to do when running the script (default: %(default)s)",
    )
    parser.add_argument(
        "--experiment-folder",
        type=str,
        default="experiment",
        help="folder to save and load experiment results in (default: %(default)s)",
    )
    parser.add_argument(
        "--samples",
        type=str,
        default="samples.png",
        help="file to save samples in (default: %(default)s)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda", "mps"],
        help="torch device (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="batch size for training (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs-per-decoder",
        type=int,
        default=50,
        metavar="N",
        help="number of training epochs per each decoder (default: %(default)s)",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=2,
        metavar="N",
        help="dimension of latent variable (default: %(default)s)",
    )
    parser.add_argument(
        "--num-decoders",
        type=int,
        default=6,
        metavar="N",
        help="number of decoders in the ensemble (default: %(default)s)",
    )
    parser.add_argument(
        "--num-reruns",
        type=int,
        default=10,
        metavar="N",
        help="number of reruns (default: %(default)s)",
    )
    parser.add_argument(
        "--num-curves",
        type=int,
        default=3,
        metavar="N",
        help="number of geodesics to plot (default: %(default)s)",
    )
    parser.add_argument(
        "--num-t",  # number of points along the curve
        type=int,
        default=20,
        metavar="N",
        help="number of points along the curve (default: %(default)s)",
    )

    args = parser.parse_args()
    print("# Options")
    for key, value in sorted(vars(args).items()):
        print(key, "=", value)

    device = args.device

    # Load a subset of MNIST and create data loaders
    def subsample(data, targets, num_data, num_classes):
        idx = targets < num_classes
        new_data = data[idx][:num_data].unsqueeze(1).to(torch.float32) / 255
        new_targets = targets[idx][:num_data]

        return torch.utils.data.TensorDataset(new_data, new_targets)

    num_train_data = 2048
    num_classes = 3
    train_tensors = datasets.MNIST(
        "data/",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    test_tensors = datasets.MNIST(
        "data/",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    train_data = subsample(
        train_tensors.data, train_tensors.targets, num_train_data, num_classes
    )
    test_data = subsample(
        test_tensors.data, test_tensors.targets, num_train_data, num_classes
    )

    mnist_train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True
    )
    mnist_test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False
    )

    # Define prior distribution
    M = args.latent_dim

    def new_encoder():
        encoder_net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(512, 2 * M),
        )
        return encoder_net

    def new_decoder():
        decoder_net = nn.Sequential(
            nn.Linear(M, 512),
            nn.Unflatten(-1, (32, 4, 4)),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
        )
        return decoder_net
    # Choose mode to run
    if args.mode == "train":

        experiments_folder = args.experiment_folder
        os.makedirs(experiments_folder, exist_ok=True)

        total_epochs = args.epochs_per_decoder * args.num_decoders
        num_models = 10 # npr. 10

        print(f"Training {num_models} models with {args.num_decoders} decoders each")
        print(f"Total epochs per model: {total_epochs}")

        for m in range(num_models):

            print(f"\n=== Model {m+1}/{num_models} ===")

            #reset seed
            torch.manual_seed(m)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(m)

            model = VAE(
                GaussianPrior(M),
                EnsembleDecoder(args.num_decoders, new_decoder),
                GaussianEncoder(new_encoder()),
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            train(
                model,
                optimizer,
                mnist_train_loader,
                total_epochs,
                args.device,
            )

            save_path = f"{experiments_folder}/model_{m}.pt"
            torch.save(model.state_dict(), save_path)

            print(f"Saved model to {save_path}")
    elif args.mode == "sample":
        model = VAE(
            GaussianPrior(M),
            EnsembleDecoder(args.num_decoders, new_decoder),
            GaussianEncoder(new_encoder()),
        ).to(device)

        model.load_state_dict(torch.load(args.experiment_folder + "/model.pt"))
        model.eval()

        # uzmi listu dekodera
        decoders = model.decoder.decoders

        with torch.no_grad():

            # =========================
            # SAMPLE IZ PRIORA
            # =========================
            z = model.prior().sample((64,)).to(device)

            # koristi prvi decoder (ili random)
            x = decoders[0](z).mean

            save_image(x.view(64, 1, 28, 28), args.samples)

            # =========================
            # REKONSTRUKCIJA
            # =========================
            data = next(iter(mnist_test_loader))[0].to(device)

            z = model.encoder(data).mean

            # opet koristi jedan decoder
            recon = decoders[0](z).mean

            save_image(
                torch.cat([data.cpu(), recon.cpu()], dim=0),
                "reconstruction_means.png"
            )
    elif args.mode == "eval":
        # Load trained model
        model = VAE(
            GaussianPrior(M),
            EnsembleDecoder(args.num_decoders, new_decoder),
            GaussianEncoder(new_encoder()),
        ).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + "/model.pt"))
        model.eval()

        elbos = []
        with torch.no_grad():
            for x, y in mnist_test_loader:
                x = x.to(device)
                elbo = model.elbo(x)
                elbos.append(elbo)
        mean_elbo = torch.tensor(elbos).mean()
        print("Print mean test elbo:", mean_elbo)

    elif args.mode == "geodesics":

        import random
        import torch.optim as optim
        import matplotlib.pyplot as plt
        from tqdm import tqdm

        # =========================
        # LOAD ONE ENSEMBLE MODEL
        # =========================
        model = VAE(
            GaussianPrior(M),
            EnsembleDecoder(args.num_decoders, new_decoder),
            GaussianEncoder(new_encoder()),
        ).to(device)

        model.load_state_dict(torch.load(args.experiment_folder + "/model_0.pt"))
        model.eval()

        encoder = model.encoder
        decoders = model.decoder.decoders  # list of decoders

        # =========================
        # ENCODE DATA + LABELS
        # =========================
        zs = []
        labels = []

        for x, y in mnist_test_loader:
            x = x.to(device)

            with torch.no_grad():
                z = encoder(x).mean

            zs.append(z)
            labels.append(y)

        zs = torch.cat(zs, dim=0)
        labels = torch.cat(labels, dim=0)

        zs_np = zs.cpu().numpy()
        labels_np = labels.cpu().numpy()

        # =========================
        # PLOT LATENT SPACE
        # =========================
        plt.figure(figsize=(7, 7))

        scatter = plt.scatter(
            zs_np[:, 0], zs_np[:, 1],
            c=labels_np,
            cmap='tab10',
            alpha=0.3,
            s=5
        )

        plt.colorbar(scatter, label='Class')

        # =========================
        # COMPUTE MULTIPLE GEODESICS
        # =========================
        num_curves = args.num_curves

        for k in tqdm(range(num_curves), desc="Curves"):

            # randomly sample two latent points
            idx1 = torch.randint(0, len(zs), (1,))
            idx2 = torch.randint(0, len(zs), (1,))

            z_start = zs[idx1].to(device)
            z_end   = zs[idx2].to(device)

            # initialize linear interpolation
            N = args.num_t
            t = torch.linspace(0, 1, N).to(device).unsqueeze(1)

            curve = (1 - t) * z_start + t * z_end
            curve = curve.clone().detach().requires_grad_(True)

            optimizer = torch.optim.Adam([curve], lr=1e-2)

            # =========================
            # OPTIMIZE CURVE (GEODESIC)
            # =========================
            for step in tqdm(range(200), desc=f"Optimizing curve {k}", leave=False):

                loss = ensemble_energy(curve, decoders, num_samples=2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # print progress occasionally
                if step % 20 == 0:
                    print(f"[curve {k}] step {step} | loss = {loss.item():.4f}")

                # enforce boundary conditions
                with torch.no_grad():
                    curve[0] = z_start
                    curve[-1] = z_end

            # =========================
            # PLOT GEODESIC AND LINEAR PATH
            # =========================
            curve_np = curve.detach().cpu().numpy()

            # optimized geodesic
            plt.plot(curve_np[:, 0], curve_np[:, 1], '-', linewidth=2)

            # linear interpolation (baseline)
            linear_curve = (1 - t) * z_start + t * z_end
            linear_np = linear_curve.detach().cpu().numpy()

            plt.plot(linear_np[:, 0], linear_np[:, 1], '--', linewidth=1)

            # start and end points
            plt.scatter(curve_np[0, 0], curve_np[0, 1], c='green', s=30)
            plt.scatter(curve_np[-1, 0], curve_np[-1, 1], c='red', s=30)

        # =========================
        # FINALIZE PLOT
        # =========================
        plt.title("Latent space with ensemble geodesics")
        plt.axis('equal')

        plt.savefig(args.samples)
        plt.close()
    elif args.mode == "save":

        import os
        import torch

        os.makedirs(args.experiment_folder, exist_ok=True)

        num_pairs = 10

        print(f"Generating {num_pairs} fixed pairs...")

        # =========================
        # LOAD DATA SIZE
        # =========================
        data = []
        for x, _ in mnist_test_loader:
            data.append(x)

        data = torch.cat(data, dim=0)
        N = len(data)

        # =========================
        # GENERATE PAIRS
        # =========================
        pairs = []

        for _ in range(num_pairs):
            i = torch.randint(0, N, (1,)).item()
            j = torch.randint(0, N, (1,)).item()

            # optional: avoid identical pairs
            while j == i:
                j = torch.randint(0, N, (1,)).item()

            pairs.append([i, j])

        pairs = torch.tensor(pairs)

        # =========================
        # SAVE
        # =========================
        save_path = f"{args.experiment_folder}/pairs.pt"
        torch.save(pairs, save_path)

        print(f"Saved pairs to {save_path}")
    elif args.mode == "cov":

        import numpy as np
        from tqdm import tqdm
        import random
        num_models = 10

        print(f"Computing CoV with {num_models} models")

        # =========================
        # LOAD DATA
        # =========================
        data = []
        for x, _ in mnist_test_loader:
            data.append(x)

        data = torch.cat(data, dim=0).to(device)

        # =========================
        # LOAD FIXED PAIRS
        # =========================
        pairs = torch.load(f"{args.experiment_folder}/pairs.pt")
        pairs = pairs.tolist()

        print(f"Loaded {len(pairs)} fixed pairs")

        # =========================
        # LOAD ALL MODELS
        # =========================
        models = []

        for m in range(num_models):
            model = VAE(
                GaussianPrior(M),
                EnsembleDecoder(args.num_decoders, new_decoder),
                GaussianEncoder(new_encoder()),
            ).to(device)

            model.load_state_dict(
                torch.load(f"{args.experiment_folder}/model_{m}.pt")
            )
            model.eval()
            models.append(model)

        # =========================
        # CoV FUNCTION
        # =========================
        def compute_cov(values):
            v = torch.tensor(values)
            return (v.std() / v.mean()).item()

        # =========================
        # MAIN LOOP OVER K
        # =========================
        Ks = [1, 2, 3, 4, 5, 6]

        results = {}

        for K in Ks:

            print(f"\n===== K = {K} decoders =====")

            cov_euc_all = []
            cov_geo_all = []

            for (i, j) in tqdm(pairs, desc=f"Pairs (K={K})"):

                x_i = data[i:i+1]
                x_j = data[j:j+1]

                euc_dists = []
                geo_dists = []

                for model in models:

                    encoder = model.encoder

                    with torch.no_grad():
                        z_i = encoder(x_i).mean
                        z_j = encoder(x_j).mean

                    # Euclidean distance (same for all K)
                    d_euc = torch.norm(z_i - z_j)
                    euc_dists.append(d_euc.item())

                    # use only K decoders
                    decoders = model.decoder.decoders[:K]

                    # Geodesic distance
                    d_geo = geodesic_distance(decoders, z_i, z_j)
                    geo_dists.append(d_geo)

                cov_euc_all.append(compute_cov(euc_dists))
                cov_geo_all.append(compute_cov(geo_dists))

            mean_cov_euc = np.mean(cov_euc_all)
            mean_cov_geo = np.mean(cov_geo_all)

            results[K] = (mean_cov_euc, mean_cov_geo)

            print(f"Euclidean CoV: {mean_cov_euc:.4f}")
            print(f"Geodesic  CoV: {mean_cov_geo:.4f}")


        # =========================
        # FINAL SUMMARY
        # =========================
        print("\n===== FINAL SUMMARY =====")

        for K in Ks:
            euc, geo = results[K]
            print(f"K={K} | Euclidean: {euc:.4f} | Geodesic: {geo:.4f}")