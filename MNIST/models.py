import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim=400, output_dim=784):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            # nn.Linear(300, 300),
            # nn.ELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def decode(self, z, y=None):
        zy = z if y is None else torch.cat((z, y), dim=1)
        return self.net(zy)


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=400, latent_dim=20):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            # nn.Linear(400, 400),
            # nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )

    def encode(self, x, y=None):
        xy = x if y is None else torch.cat((x, y), dim=1)
        h = self.net(xy)
        mu, logvar = torch.split(h, self.latent_dim, dim=1)
        return mu, logvar


class CVAE(nn.Module):
    def __init__(self, input_dim=784, label_dim=10, latent_dim=20):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.label_dim = label_dim

        self.encoder = Encoder(input_dim+label_dim,
                               hidden_dim=400, latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim+label_dim,
                               hidden_dim=400, output_dim=input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def loss(self, x, y):
        mu, logvar = self.encoder.encode(x, y)
        z = self.reparameterize(mu, logvar)
        recon_x = torch.sigmoid(self.decoder.decode(z, y))
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        return BCE + KLD, BCE, KLD

    def sample_x_given_y(self, y, num_samples):
        device = next(self.parameters()).device
        # Generate random latent vectors
        z = torch.randn(num_samples, self.latent_dim, device=device)

        # Create labels all equal to y
        y_tensor = torch.full(
            (num_samples,), y, dtype=torch.long, device=device)
        y_onehot = F.one_hot(y_tensor, num_classes=self.label_dim).float()

        # Generate samples
        logits = self.decoder.decode(z, y_onehot)

        return torch.sigmoid(logits)
