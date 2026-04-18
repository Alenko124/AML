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

        std = torch.exp(log_std)

        return td.Independent(td.Normal(loc=mean, scale=std), 2)


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

        log_px = self.decoder(z).log_prob(x)
        log_pz = self.prior().log_prob(z)
        log_qz = q.log_prob(z)

        elbo = torch.mean(log_px + log_pz - log_qz)
        
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
        return self.decoder(z).mean
    
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
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample'], help='what to do when running the script (default: %(default)s)')
    #parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=32, metavar='N', help='dimension of latent variable (default: %(default)s)')
    parser.add_argument("--prior", type=str, default="gaussian", choices=["gaussian", "mog"], help="Type of prior distribution")
    parser.add_argument("--n_components", type=int, default=10, help="Number of mixture components (only used if prior=mog)")
    parser.add_argument("--data-type", type=str, default="binarized", choices=["binarized", "continuous"], help="Use binarized or continuous MNIST")

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
        model_path = "model_gaussian.pt"
    elif args.prior == "mog":
        model_path = "model_mog.pt"
    # -------------------------------------------------
    # Load MNIST (binarized or continuous)
    # -------------------------------------------------

    if args.data_type == "binarized":
        threshold = 0.5
        transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (x > threshold).float().squeeze())])
    else:  # continuous
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
        nn.Linear(512, 784),
        nn.Unflatten(-1, (28, 28))  # za binarized
    )


    # Define VAE model
    if args.data_type == "binarized":
        decoder = BernoulliDecoder(decoder_net)
    else:  # continuous
        decoder = GaussianDecoder(decoder_net)
    encoder = GaussianEncoder(encoder_net)
    model = VAE(prior, decoder, encoder).to(device)

    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train model
        train(model, optimizer, mnist_train_loader, args.epochs, args.device)

        # Save model
        torch.save(model.state_dict(), model_path)

    elif args.mode == 'sample':
        model.load_state_dict(torch.load(model_path, map_location=torch.device(args.device)))

        # Generate samples
        model.eval()
        with torch.no_grad():
            samples = (model.sample(64)).cpu() 
            save_image(samples.view(64, 1, 28, 28), args.samples)
        total_elbo = 0.0
        total_samples = 0

        with torch.no_grad():
            for x, _ in mnist_test_loader:
                x = x.to(device)
                elbo = model.elbo(x)
                total_elbo += elbo.item() * x.size(0)
                total_samples += x.size(0)

        test_elbo = total_elbo / total_samples
        print(f"Test ELBO: {test_elbo:.4f}")
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
            prior_dist = model.prior()  # distribution object
            Z_prior = prior_dist.sample(torch.Size([num_samples]))

        Z_prior = Z_prior.cpu().numpy()

        # -------------------------------------------------
        # PCA (fit ONLY on posterior)
        # -------------------------------------------------
        if M > 2:
            print("Applying PCA...")
            pca = PCA(n_components=2)
            Z = pca.fit_transform(Z)
            Z_prior = pca.transform(Z_prior)

        # -------------------------------------------------
        # Compute shared axis limits  (FER comparison)
        # -------------------------------------------------
        x_min = min(Z[:, 0].min(), Z_prior[:, 0].min())
        x_max = max(Z[:, 0].max(), Z_prior[:, 0].max())
        y_min = min(Z[:, 1].min(), Z_prior[:, 1].min())
        y_max = max(Z[:, 1].max(), Z_prior[:, 1].max())

        # Add small padding
        padding = 0.05
        x_range = x_max - x_min
        y_range = y_max - y_min

        x_min -= padding * x_range
        x_max += padding * x_range
        y_min -= padding * y_range
        y_max += padding * y_range

        # -------------------------------------------------
        # Plot 1: Aggregate Posterior
        # -------------------------------------------------
        plt.figure(figsize=(6, 6))
        plt.scatter(
            Z[:, 0], Z[:, 1],
            s=5,
            alpha=0.4,
            c="black"
        )

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.gca().set_aspect('equal', 'box')

        plt.title(f"Aggregate Posterior ({args.prior})")
        plt.xlabel("Latent Dimension 1")
        plt.ylabel("Latent Dimension 2")
        plt.tight_layout()

        filename_post = f"aggregate_posterior_{args.prior}.png"
        plt.savefig(filename_post, dpi=300)
        plt.close()

        # -------------------------------------------------
        # Plot 2: Prior
        # -------------------------------------------------
        plt.figure(figsize=(6, 6))
        plt.scatter(
            Z_prior[:, 0], Z_prior[:, 1],
            s=5,
            alpha=0.4,
            c="black"
        )

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.gca().set_aspect('equal', 'box')

        plt.title(f"Prior Samples ({args.prior})")
        plt.xlabel("Latent Dimension 1")
        plt.ylabel("Latent Dimension 2")
        plt.tight_layout()

        filename_prior = f"prior_samples_{args.prior}.png"
        plt.savefig(filename_prior, dpi=300)
        plt.close()

        print("Figures saved:")
        print(filename_post)
        print(filename_prior)