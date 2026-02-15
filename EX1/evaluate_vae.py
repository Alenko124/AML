import torch
import torch.nn as nn
import torch.utils.data
import torch.distributions as td
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
import argparse

from vae_bernoulli import VAE, GaussianPrior, GaussianEncoder, BernoulliDecoder, MoGPrior, GaussianDecoder


# -------------------------------------------------
# Argument parser
# -------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--latent_dim", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=128)
#parser.add_argument("--model_path", type=str, default="model.pt")
parser.add_argument("--prior", type=str, default="gaussian", choices=["gaussian", "mog"], help="Type of prior distribution")
parser.add_argument("--n_components", type=int, default=10, help="Number of mixture components (only used if prior=mog)")
parser.add_argument("--data-type", type=str, default="binarized", choices=["binarized", "continuous"], help="Use binarized or continuous MNIST")
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------
# Load binarized MNIST
# -------------------------------------------------
if args.data_type == "binarized":
        threshold = 0.5
        transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (x > threshold).float().squeeze())])
else:  # continuous
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.squeeze())])

mnist_train_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=True, download=True, transform=transform), batch_size=args.batch_size, shuffle=True)

mnist_test_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=False, download=True, transform=transform), batch_size=args.batch_size, shuffle=False)



if args.prior == "gaussian":
    model_path = "model_gaussian.pt"
elif args.prior == "mog":
    model_path = "model_mog.pt"
# -------------------------------------------------
# Define model architecture
# -------------------------------------------------
M = args.latent_dim
if args.prior == "gaussian":
    prior = GaussianPrior(M)

elif args.prior == "mog":
    from vae_bernoulli import MoGPrior
    prior = MoGPrior(M, K=args.n_components)

else:
    raise ValueError("Unknown prior type")


encoder_net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, M * 2),
)

decoder_net = nn.Sequential(
    nn.Linear(M, 512),
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, 2*784),
    #nn.Unflatten(-1, (28, 28)) za binarized
)

encoder = GaussianEncoder(encoder_net)
# Define VAE model
if args.data_type == "binarized":
    decoder = BernoulliDecoder(decoder_net)
else:  # continuous
    decoder = GaussianDecoder(decoder_net)

model = VAE(prior, decoder, encoder).to(device)


# -------------------------------------------------
# Load trained model
# -------------------------------------------------
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print("Model loaded successfully.")


# -------------------------------------------------
# Evaluate ELBO on test set
# -------------------------------------------------
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
all_labels = []

with torch.no_grad():
    for x, y in mnist_test_loader:
        x = x.to(device)
        q = model.encoder(x)
        z = q.rsample()  # sample from q(z|x)
        all_z.append(z.cpu())
        all_labels.append(y)

Z = torch.cat(all_z, dim=0).numpy()
labels = torch.cat(all_labels, dim=0).numpy()


# -------------------------------------------------
# PCA if latent dimension > 2
# -------------------------------------------------
if M > 2:
    print("Applying PCA...")
    pca = PCA(n_components=2)
    Z = pca.fit_transform(Z)


# -------------------------------------------------
# Plot aggregate posterior
# -------------------------------------------------
plt.figure(figsize=(8, 6))
scatter = plt.scatter(Z[:, 0], Z[:, 1], c=labels, cmap="tab10", s=5)
plt.colorbar(scatter)
plt.title("Aggregate Posterior (Test Set)")
plt.xlabel("Latent Dimension 1")
plt.ylabel("Latent Dimension 2")
plt.tight_layout()

filename = f"aggregate_posterior_{args.prior}.png"
plt.savefig(filename, dpi=300)
print(f"Figure saved as {filename}")

