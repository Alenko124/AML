# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.2 (2024-02-06)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torch.nn import functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
from ddpm import DDPM, FcNetwork
from fid import compute_fid

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



class MoGPrior(nn.Module):
    def __init__(self, M, K=10):
        """
        Define a Mixture of Gaussians prior.

        Parameters:
        M: [int]
           Dimension of the latent space.
        K: [int]
           Number of mixture components.
        """
        super(MoGPrior, self).__init__()

        self.M = M
        self.K = K

        # Learnable means (K, M)
        self.means = nn.Parameter(torch.randn(K, M))

        # Learnable log stds (K, M)
        self.log_stds = nn.Parameter(torch.zeros(K, M))

        # Learnable mixture logits (K,)
        self.logits = nn.Parameter(torch.zeros(K))

    def forward(self):
        """
        Return the mixture prior distribution.
        """

        # Categorical over components
        mix = td.Categorical(logits=self.logits)

        # Component Gaussians (K, M)
        comp = td.Independent(
            td.Normal(
                loc=self.means,
                scale=torch.exp(self.log_stds)
            ),
            1
        )

        # Mixture distribution
        prior = td.MixtureSameFamily(mix, comp)

        return prior



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


class BernoulliDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters: 
        encoder_net: [torch.nn.Module]             
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(BernoulliDecoder, self).__init__()
        self.decoder_net = decoder_net
        self.std = nn.Parameter(torch.ones(28, 28)*0.5, requires_grad=True)

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor] 
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        logits = self.decoder_net(z)
        return td.Independent(td.Bernoulli(logits=logits), 2)

class GaussianDecoder(nn.Module):
    def __init__(self, decoder_net):
        super(GaussianDecoder, self).__init__()
        self.decoder_net = decoder_net

    def forward(self, z):
        out = self.decoder_net(z)  # (B, 784*2)

        mean, log_std = torch.chunk(out, 2, dim=-1)  # (B, 784), (B, 784)

        mean = mean.view(-1, 28, 28)
        log_std = log_std.view(-1, 28, 28)

        std = torch.exp(log_std) + 1e-6

        return td.Independent(td.Normal(loc=mean, scale=std), 2)


class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """
    def __init__(self, prior, decoder, encoder, beta=1.0):
        """
        Parameters:
        prior: [torch.nn.Module] 
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        beta: [float]
            The beta parameter for the ELBO.
        """
            
        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder
        self.beta = beta
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

        log_px = self.decoder(z).log_prob(x)
        log_pz = self.prior().log_prob(z)
        log_qz = q.log_prob(z)

        kl = log_qz - log_pz

        elbo = torch.mean(log_px - self.beta * kl)
        
        #elbo = torch.mean(self.decoder(z).log_prob(x) - td.kl_divergence(q, self.prior()), dim=0)
        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model.
        
        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        #return self.decoder(z).sample()
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
    model.train()

    total_steps = len(data_loader)*epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for x in data_iter:
            x = x[0].to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()

if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image, make_grid
    import glob

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'post', 'sample-ddpm', 'fid'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=64, metavar='N', help='dimension of latent variable (default: %(default)s)')
    parser.add_argument("--prior", type=str, default="gaussian", choices=["gaussian", "mog", "flow"], help="Type of prior distribution")
    parser.add_argument("--n_components", type=int, default=10, help="Number of mixture components (only used if prior=mog)")
    parser.add_argument("--beta", type=float, default=1.0, help="Beta parameter for ELBO (default: %(default)s)")

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device
    """
    # Load MNIST as binarized at 'thresshold' and create data loaders
    thresshold = 0.5
    mnist_train_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=True, download=True,
                                                                    transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                    batch_size=args.batch_size, shuffle=True)
    mnist_test_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=False, download=True,
                                                                transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                    batch_size=args.batch_size, shuffle=True)
    """

    if args.prior == "gaussian":
        model_path = "models/model_gaussian.pt"
    elif args.prior == "mog":
        model_path = "models/model_mog.pt"
    elif args.prior == "flow":
        model_path = "models/model_flow.pt"

    if args.beta == 1.0:
        model_path = model_path.replace(".pt", "_beta1.pt")
    elif args.beta == 0.1:
        model_path = model_path.replace(".pt", "_beta0.1.pt")
    elif args.beta == 0.001:
        model_path = model_path.replace(".pt", "_beta0.001.pt")
    elif args.beta == 0.00001:
        model_path = model_path.replace(".pt", "_beta0.00001.pt")    
    # -------------------------------------------------
    # Load MNIST (binarized or continuous)
    # -------------------------------------------------

    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.squeeze())])
    mnist_train_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=True, download=True, transform=transform), batch_size=args.batch_size, shuffle=True)
    mnist_test_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=False, download=True, transform=transform), batch_size=args.batch_size, shuffle=False)

    # Define prior distribution
    M = args.latent_dim
    if args.prior == "gaussian":
        prior = GaussianPrior(M)
    elif args.prior == "mog":
        from vae_bernoulli import MoGPrior
        prior = MoGPrior(M, K=args.n_components)
    else:
        raise ValueError("Unknown prior type")
    # Define encoder and decoder networks
    encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, M*2),
    )

    decoder_net = nn.Sequential(
        nn.Linear(M, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 2 * 784),
        #nn.Unflatten(-1, (28, 28))  # for binarized
    )
    decoder = GaussianDecoder(decoder_net)
    encoder = GaussianEncoder(encoder_net)
    model = VAE(prior, decoder, encoder, beta=args.beta).to(device)


    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train model
        train(model, optimizer, mnist_train_loader, args.epochs, args.device)

        # Save model
        torch.save(model.state_dict(), model_path)

    elif args.mode == 'sample':
        # Load ddpm model

        latent_path = f"latents/latents_beta_{args.beta:.0e}.pt"
        if not os.path.exists(latent_path):
            raise FileNotFoundError(f"Latent file not found: {latent_path}")
        z_data = torch.load(latent_path)
        print(f"Loaded {z_data.shape[0]} latent vectors")
        print(f"Latent dimension: {z_data.shape[1]}")
        D = args.latent_dim

        # Define the network
        num_hiddena = 256
        network = FcNetwork(D, num_hiddena)

        # Set the number of steps in the diffusion process
        T = 300

        # Define model
        ddpm = DDPM(network, T=T).to(args.device)
        model.load_state_dict(torch.load(model_path, map_location=torch.device(args.device)))
        model.eval()
        # -------------------------------------------------
        # Aggregate posterior sampling
        # -------------------------------------------------
        all_z = []

        model.eval()

        with torch.no_grad():
            for x, _ in mnist_test_loader:
                x = x.to(device)
                q = model.encoder(x)
                z = q.rsample()
                all_z.append(z.cpu())

        Z = torch.cat(all_z, dim=0)

        # -------------------------------------------------
        # Sample from prior (general solution)
        # -------------------------------------------------
        num_samples = Z.shape[0]

        with torch.no_grad():
            prior_dist = model.prior  # distribution object
            Z_prior = prior_dist().sample(torch.Size([num_samples]))

        Z_prior = Z_prior.cpu().numpy()
        # load DDPM samples
        latent_dim = args.latent_dim
        load_path = f"ddpm/ddpm_beta_{args.beta:.0e}.pt"
        ddpm.load_state_dict(torch.load(load_path, map_location=device))
        ddpm.eval()

        with torch.no_grad():
            Z_ddpm = ddpm.sample((num_samples, latent_dim)).cpu().numpy()

        # -------------------------------------------------
        # PCA
        # -------------------------------------------------
        if M > 2:
            print("Applying PCA...")
            pca = PCA(n_components=2)
            Z = pca.fit_transform(Z)
            Z_prior = pca.transform(Z_prior)
        if M > 2:
            Z_ddpm = pca.transform(Z_ddpm)    
        # -------------------------------------------------
        # Compute shared axis limits
        # -------------------------------------------------
        x_min = min(Z[:, 0].min(), Z_prior[:, 0].min(), Z_ddpm[:, 0].min())
        x_max = max(Z[:, 0].max(), Z_prior[:, 0].max(), Z_ddpm[:, 0].max())
        y_min = min(Z[:, 1].min(), Z_prior[:, 1].min(), Z_ddpm[:, 1].min())
        y_max = max(Z[:, 1].max(), Z_prior[:, 1].max(), Z_ddpm[:, 1].max())

        # Add small padding
        padding = 0.05
        x_range = x_max - x_min
        y_range = y_max - y_min

        x_min -= padding * x_range
        x_max += padding * x_range
        y_min -= padding * y_range
        y_max += padding * y_range

        # -------------------------------------------------
        # Plot all three side-by-side
        # -------------------------------------------------
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # ---- 1️⃣ Aggregate Posterior ----
        axes[0].scatter(
            Z[:, 0], Z[:, 1],
            s=5,
            alpha=0.4,
            c="black"
        )
        axes[0].set_xlim(x_min, x_max)
        axes[0].set_ylim(y_min, y_max)
        axes[0].set_aspect('equal', 'box')
        axes[0].set_title("Aggregate Posterior")
        axes[0].set_xlabel("Latent Dim 1")
        axes[0].set_ylabel("Latent Dim 2")

        # ---- 2️⃣ Prior ----
        axes[1].scatter(
            Z_prior[:, 0], Z_prior[:, 1],
            s=5,
            alpha=0.4,
            c="black"
        )
        axes[1].set_xlim(x_min, x_max)
        axes[1].set_ylim(y_min, y_max)
        axes[1].set_aspect('equal', 'box')
        axes[1].set_title("Prior Samples")
        axes[1].set_xlabel("Latent Dim 1")

        # ---- 3️⃣ Latent DDPM ----
        axes[2].scatter(
            Z_ddpm[:, 0], Z_ddpm[:, 1],
            s=5,
            alpha=0.4,
            c="black"
        )
        axes[2].set_xlim(x_min, x_max)
        axes[2].set_ylim(y_min, y_max)
        axes[2].set_aspect('equal', 'box')
        axes[2].set_title("Latent DDPM")
        axes[2].set_xlabel("Latent Dim 1")

        plt.tight_layout()

        filename_all = f"figures/latent_comparison_beta_{args.beta}.png"
        plt.savefig(filename_all, dpi=300)
        plt.close()

        print(f"Saved combined figure to {filename_all}")
    elif args.mode == 'post':
        model.load_state_dict(torch.load(model_path, map_location=torch.device(args.device)))
        # Generate latent samples
        model.eval()
        os.makedirs("latents", exist_ok=True)

        all_z = []

        with torch.no_grad():
            for x, _ in mnist_train_loader:
                x = x.to(args.device)

                q = model.encoder(x)

               
                z = q.base_dist.loc

                all_z.append(z.cpu())

        all_z = torch.cat(all_z, dim=0)

        save_path = f"latents/latents_beta_{args.beta:.0e}.pt"
        torch.save(all_z, save_path)

        print(f"Saved {all_z.shape[0]} latent vectors to {save_path}")
        print(f"Latent dimension: {all_z.shape[1]}")
    elif args.mode == 'sample-ddpm':
        # Load ddpm model

        latent_path = f"latents/latents_beta_{args.beta:.0e}.pt"
        if not os.path.exists(latent_path):
            raise FileNotFoundError(f"Latent file not found: {latent_path}")
        z_data = torch.load(latent_path)
        print(f"Loaded {z_data.shape[0]} latent vectors")
        print(f"Latent dimension: {z_data.shape[1]}")
        D = args.latent_dim

        # Define the network
        num_hiddena = 256
        network = FcNetwork(D, num_hiddena)

        # Set the number of steps in the diffusion process
        T = 300

        # Define model
        ddpm = DDPM(network, T=T).to(args.device)
        model.load_state_dict(torch.load(model_path, map_location=torch.device(args.device)))
        model.load_state_dict(torch.load(model_path, map_location=torch.device(args.device)))
        # Generate latent ddpm samples
        model.eval()

        load_path = f"ddpm/ddpm_beta_{args.beta:.0e}.pt"
        ddpm.load_state_dict(torch.load(load_path, map_location=torch.device(args.device)))
        print(f"Loaded model from {load_path}")
        ddpm.eval()
        os.makedirs("figures", exist_ok=True)

        n_samples = 4

        with torch.no_grad():

            # ---- DDPM ----
            z_samples = ddpm.sample((n_samples, D)).to(args.device)
            x_ddpm = model.decoder(z_samples).mean
            x_ddpm = torch.clamp(x_ddpm, 0, 1)
            x_ddpm = x_ddpm.view(n_samples, 1, 28, 28)

            # ---- VAE ----
            x_vae = model.sample(n_samples)
            x_vae = torch.clamp(x_vae, 0, 1)
            x_vae = x_vae.view(n_samples, 1, 28, 28)

            # ---- Spoji u jedan tensor ----
            # Prvo DDPM red, pa VAE red
            combined = torch.cat([x_ddpm, x_vae], dim=0)

            save_path = f"figures/ddpm_vs_vae_beta_{args.beta:.0e}.png"

            save_image(
                combined,
                save_path,
                nrow=8,      # 8 po redu → 2 reda ukupno
                normalize=True
            )

        print(f"Saved comparison image to {save_path}")
                # -------------------------------------------------
        # 2x2 Representative Samples (DDPM only)
        # -------------------------------------------------
        rep_ddpm = x_ddpm.detach().cpu()

        rep_path_ddpm = f"figures/ddpm_2x2_beta_{args.beta:.0e}.png"

        save_image(
            rep_ddpm,
            rep_path_ddpm,
            nrow=2,          # 2 po redu → 2x2
            normalize=True
        )

        print(f"Saved 2x2 DDPM representative samples to {rep_path_ddpm}")
    elif args.mode == 'fid':
        # Load ddpm model

        latent_path = f"latents/latents_beta_{args.beta:.0e}.pt"
        if not os.path.exists(latent_path):
            raise FileNotFoundError(f"Latent file not found: {latent_path}")
        z_data = torch.load(latent_path)
        print(f"Loaded {z_data.shape[0]} latent vectors")
        print(f"Latent dimension: {z_data.shape[1]}")
        D = args.latent_dim

        # Define the network
        num_hiddena = 256
        network = FcNetwork(D, num_hiddena)

        # Set the number of steps in the diffusion process
        T = 300

        # Define model
        ddpm = DDPM(network, T=T).to(args.device)
        from fid import compute_fid
        import os
        import time

        # -------------------------------------------------
        # Load VAE
        # -------------------------------------------------
        model.load_state_dict(
            torch.load(model_path, map_location=torch.device(args.device))
        )
        model.eval()

        # -------------------------------------------------
        # Load latent DDPM
        # -------------------------------------------------
        load_path = f"ddpm/ddpm_beta_{args.beta:.0e}.pt"
        ddpm.load_state_dict(
            torch.load(load_path, map_location=torch.device(args.device))
        )
        ddpm.eval()
        print(f"Loaded model from {load_path}")

        # -------------------------------------------------
        # Collect REAL images from test loader
        # -------------------------------------------------
        real_images = []

        with torch.no_grad():
            for x, _ in mnist_test_loader:
                real_images.append(x)

        real_images = torch.cat(real_images, dim=0).to(args.device)
        n_samples = real_images.shape[0]

        print(f"Number of real test samples: {n_samples}")

        # -------------------------------------------------
        # Generate samples from latent DDPM
        # -------------------------------------------------
        #latent_dim = real_images.shape[1] * 0 + D  # koristi već definisan D

        start = time.time()

        with torch.no_grad():
            z_samples = ddpm.sample((n_samples, D)).to(args.device)
            x_gen = model.decoder(z_samples).mean

        end = time.time()

        samples_per_sec = n_samples / (end - start)
        print(f"Sampling speed: {samples_per_sec:.2f} samples/sec")

        # -------------------------------------------------
        # Rescale to [-1, 1] for FID
        # -------------------------------------------------
        #real_images = 2 * real_images - 1
        #x_gen = 2 * x_gen - 1
        real_images = real_images.unsqueeze(1)
        x_gen = x_gen.unsqueeze(1)
        print(real_images.min(), real_images.max())
        x_gen = torch.clamp(x_gen, -1, 1)
        real_images = torch.clamp(real_images, -1, 1)
        print(x_gen.min(), x_gen.max())
        # -------------------------------------------------
        # Compute FID
        # -------------------------------------------------
        fid_score = compute_fid(
            real_images,
            x_gen,
            device=args.device,
            classifier_ckpt="mnist_classifier.pth"
        )
        with torch.no_grad():
            x_genvae = model.sample(n_samples)
        x_genvae = x_genvae.to(args.device)
        x_genvae = x_genvae.unsqueeze(1)
        x_genvae = torch.clamp(x_genvae, -1, 1)
        fid_scorevae = compute_fid(
            real_images,
            x_genvae,
            device=args.device,
            classifier_ckpt="mnist_classifier.pth"
        )
        print(f"FID (latent DDPM, beta={args.beta}): {fid_score:.4f}")
        print(f"FID (betaVAE, beta={args.beta}): {fid_scorevae:.4f}")
        num_show = 5

        real_plot = real_images[:num_show].detach().cpu()
        ddpm_plot = x_gen[:num_show].detach().cpu()
        vae_plot = x_genvae[:num_show].detach().cpu()

        # Ako su u [-1,1], prebaci u [0,1] za prikaz
        #real_plot = (real_plot + 1) / 2
        #ddpm_plot = (ddpm_plot + 1) / 2
        #vae_plot = (vae_plot + 1) / 2
        real_plot=torch.clamp(real_plot, 0, 1)
        ddpm_plot=torch.clamp(ddpm_plot, 0, 1)
        vae_plot=torch.clamp(vae_plot, 0, 1)
        fig, axes = plt.subplots(3, num_show, figsize=(12, 6))

        for i in range(num_show):
            axes[0, i].imshow(real_plot[i, 0], cmap="gray")
            axes[0, i].axis("off")
            
            axes[1, i].imshow(ddpm_plot[i, 0], cmap="gray")
            axes[1, i].axis("off")
            
            axes[2, i].imshow(vae_plot[i, 0], cmap="gray")
            axes[2, i].axis("off")

        axes[0, 0].set_ylabel("Real", fontsize=12)
        axes[1, 0].set_ylabel("Latent DDPM", fontsize=12)
        axes[2, 0].set_ylabel("β-VAE", fontsize=12)

        plt.tight_layout()
        plt.savefig(f"figures/comparison_beta_{args.beta}.png", dpi=300)
        plt.close()

        print(f"Saved comparison plot to figures/comparison_beta_{args.beta}.png")