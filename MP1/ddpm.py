# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-02-11)

import torch
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as F
from tqdm import tqdm


class DDPM(nn.Module):
    def __init__(self, network, beta_1=1e-4, beta_T=2e-2, T=100):
        """
        Initialize a DDPM model.

        Parameters:
        network: [nn.Module]
            The network to use for the diffusion process.
        beta_1: [float]
            The noise at the first step of the diffusion process.
        beta_T: [float]
            The noise at the last step of the diffusion process.
        T: [int]
            The number of steps in the diffusion process.
        """
        super(DDPM, self).__init__()
        self.network = network
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.T = T

        self.beta = nn.Parameter(torch.linspace(beta_1, beta_T, T), requires_grad=False)
        self.alpha = nn.Parameter(1 - self.beta, requires_grad=False)
        self.alpha_cumprod = nn.Parameter(self.alpha.cumprod(dim=0), requires_grad=False)
    
    def negative_elbo(self, x):
        """
        Evaluate the DDPM negative ELBO on a batch of data.

        Parameters:
        x: [torch.Tensor]
            A batch of data (x) of dimension `(batch_size, *)`.
        Returns:
        [torch.Tensor]
            The negative ELBO of the batch of dimension `(batch_size,)`.
        """

        ### Implement Algorithm 1 here ###
        batch_size = x.shape[0]
        device = x.device

        # 1. Sample t uniformly from {0,...,T-1}
        t = torch.randint(0, self.T, (batch_size,), device=device)

        # 2. Sample epsilon ~ N(0, I)
        epsilon = torch.randn_like(x)

        # 3. Get alpha_bar_t
        alpha_bar_t = self.alpha_cumprod[t]
        alpha_bar_t = alpha_bar_t.view(batch_size, *([1] * (x.dim() - 1)))

        # 4. Forward diffusion: q(x_t | x_0)
        x_t = torch.sqrt(alpha_bar_t) * x + \
            torch.sqrt(1 - alpha_bar_t) * epsilon

        # 5. Predict noise
        t_input = t.unsqueeze(1).float() / (self.T - 1) # Normalize time input for FC network
        epsilon_theta = self.network(x_t, t_input)

        # 6. Compute MSE per sample (negative ELBO)
        neg_elbo = torch.mean(
            (epsilon - epsilon_theta) ** 2,
            dim=tuple(range(1, x.dim()))
        )

        return neg_elbo

    def sample(self, shape):
        """
        Sample from the model.

        Parameters:
        shape: [tuple]
            The shape of the samples to generate.
        Returns:
        [torch.Tensor]
            The generated samples.
        """
        # Sample x_t for t=T (i.e., Gaussian noise)
        device = self.alpha.device

        # x_T ~ N(0,I)
        x_t = torch.randn(shape, device=device)


        # Sample x_t given x_{t+1} until x_0 is sampled
        for t in range(self.T-1, -1, -1):
            ### Implement the remaining of Algorithm 2 here ###
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)

            beta_t = self.beta[t]
            alpha_t = self.alpha[t]
            alpha_bar_t = self.alpha_cumprod[t]

            # Predict noise
            t_tensor = torch.full((shape[0],), t, device=device) #za FC network
            t_tensor = t_tensor.unsqueeze(1).float() / (self.T - 1) # Normalize time input for FC network
            epsilon_theta = self.network(x_t, t_tensor)

            # Compute mean of p_theta(x_{t-1} | x_t)
            mean = (1 / torch.sqrt(alpha_t)) * (
                x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * epsilon_theta
            )

            if t > 0:
                z = torch.randn_like(x_t)
                sigma_t = torch.sqrt(beta_t)
                x_t = mean + sigma_t * z
            else:
                # at t=0 no noise
                x_t = mean


        return x_t

    def loss(self, x):
        """
        Evaluate the DDPM loss on a batch of data.

        Parameters:
        x: [torch.Tensor]
            A batch of data (x) of dimension `(batch_size, *)`.
        Returns:
        [torch.Tensor]
            The loss for the batch.
        """
        return self.negative_elbo(x).mean()


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a Flow model.

    Parameters:
    model: [Flow]
       The model to train.
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
            if isinstance(x, (list, tuple)):
                x = x[0]
            x = x.to(device)
            optimizer.zero_grad()
            loss = model.loss(x)
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()


class FcNetwork(nn.Module):
    def __init__(self, input_dim, num_hidden):
        """
        Initialize a fully connected network for the DDPM, where the forward function also take time as an argument.
        
        parameters:
        input_dim: [int]
            The dimension of the input data.
        num_hidden: [int]
            The number of hidden units in the network.
        """
        super(FcNetwork, self).__init__()
        self.network = nn.Sequential(nn.Linear(input_dim+1, num_hidden), nn.ReLU(), 
                                     nn.Linear(num_hidden, num_hidden), nn.ReLU(), 
                                     nn.Linear(num_hidden, input_dim))
        self.network = nn.Sequential(
            nn.Linear(input_dim + 1, num_hidden),
            nn.SiLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.SiLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.SiLU(),
            nn.Linear(num_hidden, input_dim),
        )

    def forward(self, x, t):
        """"
        Forward function for the network.
        
        parameters:
        x: [torch.Tensor]
            The input data of dimension `(batch_size, input_dim)`
        t: [torch.Tensor]
            The time steps to use for the forward pass of dimension `(batch_size, 1)`
        """
        x_input = torch.cat([x, t], dim=1)
        h = self.network[:-1](x_input)
        return self.network[-1](h) + x


if __name__ == "__main__":
    import torch.utils.data
    from torchvision import datasets, transforms
    from torchvision.utils import save_image

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'test'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--data', type=str, default='tg', choices=['tg', 'cb', 'mnist', 'latent'], help='dataset to use {tg: two Gaussians, cb: chequerboard} (default: %(default)s)')
    parser.add_argument('--model', type=str, default='FC.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='V', help='learning rate for training (default: %(default)s)')
    parser.add_argument("--beta", type=float, default=1.0, help="Beta parameter for ELBO (default: %(default)s)")

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    import os

    latent_path = f"latents/latents_beta_{args.beta:.0e}.pt"

    if not os.path.exists(latent_path):
        raise FileNotFoundError(f"Latent file not found: {latent_path}")

    z_data = torch.load(latent_path)

    print(f"Loaded {z_data.shape[0]} latent vectors")
    print(f"Latent dimension: {z_data.shape[1]}")

    class LatentDataset(torch.utils.data.Dataset):
        def __init__(self, z):
            self.z = z

        def __len__(self):
            return self.z.shape[0]

        def __getitem__(self, idx):
            return self.z[idx]

    latent_dataset = LatentDataset(z_data)

    train_loader = torch.utils.data.DataLoader(
        latent_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    # Get the dimension of the dataset
    batch = next(iter(train_loader))
    if isinstance(batch, (list, tuple)):
        D = batch[0].shape[1]
    else:
        D = batch.shape[1]

    # Define the network
    num_hidden = 256
    network = FcNetwork(D, num_hidden)

    # Set the number of steps in the diffusion process
    T = 300

    # Define model
    model = DDPM(network, T=T).to(args.device)

    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Train model
        train(model, optimizer, train_loader, args.epochs, args.device)

        # Save model
        import os

        os.makedirs("ddpm", exist_ok=True)

        save_path = f"ddpm/ddpm_beta_{args.beta:.0e}.pt"

        torch.save(model.state_dict(), save_path)
        
        print(f"Model saved to {save_path}")

    elif args.mode == 'sample':
        import matplotlib.pyplot as plt
        import numpy as np

        # Load model
        model.load_state_dict(
            torch.load(args.model, map_location=torch.device(args.device))
        )

        model.eval()
        with torch.no_grad():
            samples = model.sample((100, D)).cpu()

        # Transform back from [-1,1] to [0,1]
        samples = samples / 2 + 0.5
        samples = torch.clamp(samples, 0.0, 1.0)

        if args.data in ['tg', 'cb']:
            # ---- TOY DATA PLOT ----
            coordinates = [[[x,y] for x in np.linspace(*toy.xlim, 1000)]
                        for y in np.linspace(*toy.ylim, 1000)]
            prob = torch.exp(toy().log_prob(torch.tensor(coordinates)))

            fig, ax = plt.subplots(1, 1, figsize=(7, 5))
            im = ax.imshow(prob,
                        extent=[toy.xlim[0], toy.xlim[1],
                                toy.ylim[0], toy.ylim[1]],
                        origin='lower',
                        cmap='YlOrRd')

            ax.scatter(samples[:, 0], samples[:, 1],
                    s=3, c='black', alpha=0.5)

            ax.set_xlim(toy.xlim)
            ax.set_ylim(toy.ylim)
            ax.set_aspect('equal')
            fig.colorbar(im)

            plt.savefig(args.samples)
            plt.close()

        elif args.data == "mnist":
            # ---- MNIST GRID PLOT ----
            samples = samples.view(-1, 1, 28, 28)

            grid_size = 10
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))

            for i in range(grid_size):
                for j in range(grid_size):
                    idx = i * grid_size + j
                    axes[i, j].imshow(samples[idx, 0], cmap='gray')
                    axes[i, j].axis('off')

            plt.tight_layout()
            plt.savefig(args.samples)
            plt.close()