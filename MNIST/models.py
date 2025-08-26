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


class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, latent_dim=20):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )

    def encode(self, x, y=None):
        xy = x if y is None else torch.cat([x, y], dim=1)
        h = self.net(xy)
        mu, logvar = torch.split(h, self.latent_dim, dim=1)
        return mu, logvar


class MLPDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim=512, output_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, output_dim)  # logits
        )

    def decode(self, z, y=None):
        zy = z if y is None else torch.cat([z, y], dim=1)
        return self.net(zy)  # (B,784) logits


class ConvEncoder2D(nn.Module):
    """expects x flat (B,784); returns (mu, logvar)"""

    def __init__(self, input_dim, latent_dim=20, label_dim=10):
        super().__init__()
        assert input_dim >= 784, "This encoder is for 28x28 images"
        self.latent_dim = latent_dim
        self.label_dim = label_dim

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.GELU(),   # 28->14
            nn.Conv2d(32, 64, 4, 2, 1), nn.GELU(),  # 14->7
        )
        self.fc = nn.Linear(64 * 7 * 7 + self.label_dim, 2 * latent_dim)

    def encode(self, x, y=None):
        B = x.size(0)
        x_img = x.view(B, 1, 28, 28)
        h = self.conv(x_img).view(B, -1)
        if y is not None:
            h = torch.cat([h, y], dim=1)
        h = self.fc(h)
        mu, logvar = torch.split(h, self.latent_dim, dim=1)
        return mu, logvar


class ConvDecoder2D(nn.Module):
    """decode(z,y) -> logits flat (B,784)"""

    def __init__(self, latent_dim, output_dim=784):
        super().__init__()
        assert output_dim == 784, "This decoder is for 28x28 images"
        self.output_dim = output_dim
        self.latent_dim = latent_dim

        self.fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GELU(),  # 7->14
            nn.ConvTranspose2d(32, 1, 4, 2, 1),              # 14->28
            # logits out
        )

    def decode(self, z, y=None):
        h = z if y is None else torch.cat([z, y], dim=1)
        h = self.fc(h)
        h = h.view(z.size(0), 64, 7, 7)
        logits = self.deconv(h)                 # (B,1,28,28)
        return logits.view(z.size(0), self.output_dim)      # flat logits


class CVAE(nn.Module):
    def __init__(self, input_dim=784, label_dim=10, latent_dim=20, name="", arch="base"):
        super(CVAE, self).__init__()
        self.name = name
        self.latent_dim = latent_dim
        self.label_dim = label_dim
        self.arch = arch
        if arch == "base":
            self.encoder = Encoder(input_dim=input_dim+label_dim,
                                   hidden_dim=400, latent_dim=latent_dim)
            self.decoder = Decoder(latent_dim=latent_dim+label_dim,
                                   hidden_dim=400, output_dim=input_dim)
        elif arch == "mlp":
            self.encoder = MLPEncoder(input_dim=input_dim+label_dim,
                                      hidden_dim=512, latent_dim=latent_dim)
            self.decoder = MLPDecoder(latent_dim=latent_dim+label_dim,
                                      hidden_dim=512, output_dim=input_dim)
        elif arch == "conv":
            self.encoder = ConvEncoder2D(input_dim=input_dim+label_dim,
                                         latent_dim=latent_dim, label_dim=label_dim)
            self.decoder = ConvDecoder2D(latent_dim=latent_dim+label_dim,
                                         output_dim=input_dim)
        else:
            raise ValueError(f"Invalid architecture: {arch}")

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def loss(self, x, y):
        """
        Compute CVAE loss.

        Args:
            x: Input images [batch_size, 784] (flattened)
            y: Labels [batch_size, label_dim] (one-hot encoded)

        Returns:
            (total_loss, summary_dict) where summary_dict contains:
            - 'recon': reconstruction loss (binary cross entropy)
            - 'kld': KL divergence loss            
        """
        mu, logvar = self.encoder.encode(x, y)
        z = self.reparameterize(mu, logvar)
        recon_logits = self.decoder.decode(z, y)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        BCE = F.binary_cross_entropy_with_logits(
            recon_logits, x, reduction='sum')

        total_loss = BCE + KLD

        # Return loss and model-specific summary (normalized per sample)
        batch_size = x.size(0)
        summary = {
            'recon': BCE.item(),  # Per-sample reconstruction,
            'kld': KLD.item(),  # Per-sample KLD
        }

        return total_loss, summary

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
        prob = torch.sigmoid(logits)

        return torch.bernoulli(prob)

    def get_name(self):
        return self.name

    def __str__(self):
        return f"CVAE(name={self.name}, latent_dim={self.latent_dim})"

    def __repr__(self):
        return f"CVAE(name={self.name}, latent_dim={self.latent_dim})"


class SyntheticDiscriminator(nn.Module):
    """
    A flexible discriminator that can be trained with the same training function as CVAE.
    Works with flattened images (784 dimensions) like CVAE.
    """

    def __init__(self, input_dim=784):
        super().__init__()
        self.label_dim = 2  # Binary classification: 0 (fake) or 1 (real)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)  # Keep [batch_size, 1] shape for logits

    def score(self, x):
        """
        Score images to get discriminator predictions.

        Args:
            x: Input images [batch_size, 784] (flattened) or [batch_size, 1, 28, 28]

        Returns:
            Scores [batch_size, 1] where higher values indicate more real-like images
        """
        # Ensure input is flattened
        if x.dim() == 4 and x.shape[1] == 1:
            # Input is [batch_size, 1, 28, 28], flatten to [batch_size, 784]
            x = x.view(x.size(0), -1)
        elif x.dim() != 2 or x.shape[1] != 784:
            raise ValueError(
                f"Expected input shape [batch_size, 784] or [batch_size, 1, 28, 28], got {x.shape}")

        # Forward pass and apply sigmoid
        logits = self.forward(x)  # [batch_size, 1]
        scores = torch.sigmoid(logits)  # [batch_size, 1]

        return scores

    def loss(self, x, y):
        """
        Compute discriminator loss.

        Args:
            x: Input images [batch_size, 784] (flattened like CVAE)
            y: Labels [batch_size, 2] (one-hot: [1,0] for fake, [0,1] for real)
               OR [batch_size] (binary: 0 for fake, 1 for real)

        Returns:
            (total_loss, summary_dict) where summary_dict contains:
            - 'accuracy': number of correct predictions
            - 'total_predictions': total number of predictions
        """
        # Forward pass (x is already flattened)
        x = (x > 0.5).float()
        logits = self.forward(x)  # [batch_size, 1]

        # Debug: print shapes to understand the issue
        # print(
        #     f"DEBUG - x shape: {x.shape}, y shape: {y.shape}, logits shape: {logits.shape}")

        # Handle both one-hot and binary labels
        if y.dim() == 2 and y.size(1) == 2:
            # One-hot format: [1,0] = fake, [0,1] = real
            y_binary = y[:, 1]  # Take second column: fake=0, real=1
        else:
            # Binary format: already 0 or 1
            y_binary = y.view(-1).float()

        # print(f"DEBUG - y_binary shape before processing: {y_binary.shape}")
        # Ensure y_binary has the same shape as logits [batch_size, 1]
        y_binary = y_binary.view(-1, 1)

        # print(f"DEBUG - y_binary shape after processing: {y_binary.shape}")

        # Binary Cross Entropy Loss
        total_loss = F.binary_cross_entropy_with_logits(
            logits, y_binary, reduction='sum')

        # Calculate accuracy
        predictions = (logits > 0).float()  # [batch_size, 1]
        # Ensure both tensors have the same shape for comparison
        correct = (predictions.squeeze(-1) == y_binary.squeeze(-1)).sum()
        # Return loss and model-specific summary
        summary = {
            'accuracy': correct.item(),
        }

        return total_loss, summary
