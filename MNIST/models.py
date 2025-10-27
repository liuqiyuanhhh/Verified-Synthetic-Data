from torch.nn.utils.parametrizations import spectral_norm
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

    def _ensure_label_format(self, y):
        """
        Ensure y is in the correct format for CVAE.

        Args:
            y: Labels [batch_size] (integers 0-9) or [batch_size, label_dim] (one-hot encoded)

        Returns:
            y: Labels [batch_size, label_dim] (one-hot encoded)

        Raises:
            ValueError: If y format is not supported
        """
        if y.dim() == 1:
            # y is integers, convert to one-hot
            if not torch.all((y >= 0) & (y < self.label_dim)):
                raise ValueError(
                    f"CVAE expects integer labels in range [0, {self.label_dim-1}], got values outside this range")
            return F.one_hot(y, num_classes=self.label_dim).float()
        elif y.dim() == 2 and y.size(1) == self.label_dim:
            # y is already one-hot
            return y.float()
        else:
            raise ValueError(
                f"CVAE expects y shape [batch_size] (integers) or [batch_size, {self.label_dim}] (one-hot), got {y.shape}")

    def loss(self, x, y):
        """
        Compute CVAE loss.

        Args:
            x: Input images [batch_size, 784] (flattened)
            y: Labels [batch_size] (integers 0-9) or [batch_size, label_dim] (one-hot encoded)

        Returns:
            (total_loss, summary_dict) where summary_dict contains:
            - 'recon': reconstruction loss (binary cross entropy)
            - 'kld': KL divergence loss            
        """
        y = self._ensure_label_format(y)
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

    def sample_x_given_y(self, y, num_samples, binary_format: bool = True):
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

        if binary_format:
            return torch.bernoulli(prob)
        else:
            return prob

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

    def score(self, x, y_digit):
        """
        Score images to get discriminator predictions.

        Args:
            x: Input images [batch_size, 784] (flattened) or [batch_size, 1, 28, 28]
            y_digit: Digit Labels [batch_size, 10] (one-hot encoding of digits 0-9), but not used in this discriminator
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

    def _ensure_label_format(self, y):
        """
        Ensure y is in the correct format for SyntheticDiscriminator.

        Args:
            y: Labels [batch_size] (integers 0 or 1) or [batch_size, 2] (one-hot: [1,0] for fake, [0,1] for real)

        Returns:
            y_binary: Labels [batch_size, 1] (binary: 0 for fake, 1 for real)

        Raises:
            ValueError: If y format is not supported
        """
        if y.dim() == 1:
            # y is integers (0=fake, 1=real)
            if not torch.all((y == 0) | (y == 1)):
                raise ValueError(
                    "SyntheticDiscriminator expects integer labels 0 (fake) or 1 (real)")
            return y.float().view(-1, 1)
        elif y.dim() == 2 and y.size(1) == 2:
            # One-hot format: [1,0] = fake, [0,1] = real
            if not torch.all((y == 0) | (y == 1)):
                raise ValueError(
                    "SyntheticDiscriminator one-hot format should contain only 0s and 1s")
            if not torch.all(y.sum(dim=1) == 1):
                raise ValueError(
                    "SyntheticDiscriminator one-hot format should have exactly one 1 per row")
            # Take second column: fake=0, real=1
            return y[:, 1].float().view(-1, 1)
        else:
            raise ValueError(
                f"SyntheticDiscriminator expects y shape [batch_size] (integers 0,1) or [batch_size, 2] (one-hot), got {y.shape}")

    def loss(self, x, y):
        """
        Compute discriminator loss.

        Args:
            x: Input images [batch_size, 784] (flattened like CVAE)
            y: Labels [batch_size] (integers 0 or 1) or [batch_size, 2] (one-hot: [1,0] for fake, [0,1] for real)

        Returns:
            (total_loss, summary_dict) where summary_dict contains:
            - 'accuracy': number of correct predictions
        """
        # Forward pass (x is already flattened)
        x = (x > 0.5).float()
        logits = self.forward(x)  # [batch_size, 1]

        # Ensure y is in correct format
        y_binary = self._ensure_label_format(y)

        # Binary Cross Entropy Loss
        total_loss = F.binary_cross_entropy_with_logits(
            logits, y_binary, reduction='sum')

        # Calculate accuracy
        predictions = (logits > 0).float()  # [batch_size, 1]
        correct = predictions.eq(y_binary).sum()

        summary = {
            'accuracy': correct.item(),
        }

        return total_loss, summary


class ConditionalDiscriminator(nn.Module):
    """
    A flexible conditional discriminator that can work with both MLP and Conv architectures.
    Supports class-conditional discrimination with digit labels.
    Compatible with existing training functions in train_helper.
    """

    def __init__(self, input_dim=784, name="", arch='mlp',
                 base_channels=64, dropout=0.1, use_spectral_norm=True):
        """
        Args:
            input_dim: Input dimension (784 for flattened MNIST)
            arch: Architecture type ('mlp' or 'conv')
            base_channels: Base channels for Conv architecture
            dropout: Dropout rate
            use_spectral_norm: Whether to use spectral normalization
        """
        super().__init__()
        self.name = name
        self.input_dim = input_dim
        self.num_classes = 10  # Number of digit classes
        self.arch = arch

        # Define spectral normalization transform
        self.spectral_norm_transform = lambda m: spectral_norm(
            m) if use_spectral_norm else m

        if arch == 'mlp':
            self._build_mlp_architecture(dropout)
        elif arch == 'conv':
            self._build_conv_architecture(
                base_channels, dropout)
        else:
            raise ValueError(
                f"Unknown architecture: {arch}. Choose 'mlp' or 'conv'")

    def _build_mlp_architecture(self, dropout):
        """Build MLP architecture similar to your current discriminator."""
        hidden_dims = [512, 256, 128]

        # Input: image (784) + digit_onehot (10) = 794
        dims = [self.input_dim + self.num_classes, *hidden_dims, 1]

        layers = []
        for i in range(len(dims) - 2):
            lin = self.spectral_norm_transform(nn.Linear(dims[i], dims[i+1]))
            layers.extend([lin, nn.LeakyReLU(0.2, inplace=True)])
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))

        # Output layer
        lin_out = self.spectral_norm_transform(nn.Linear(dims[-2], dims[-1]))
        layers.append(lin_out)

        self.net = nn.Sequential(*layers)

    def _build_conv_architecture(self, base_channels, dropout):
        """Build Conv architecture with spatial broadcasting."""
        # Input channels: 1 image + 10 digit conditions = 11
        C_in = 1 + self.num_classes

        self.features = nn.Sequential(
            self.spectral_norm_transform(
                # 28 -> 14
                nn.Conv2d(C_in, base_channels, 3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            self.spectral_norm_transform(
                # 14 -> 7
                nn.Conv2d(base_channels, base_channels*2, 3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            self.spectral_norm_transform(
                nn.Conv2d(base_channels*2, base_channels*4, 3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Build head layer with optional dropout
        linear_layer = self.spectral_norm_transform(
            nn.Linear(base_channels*4*7*7, 1))
        if dropout > 0:
            self.head = nn.Sequential(nn.Dropout(p=dropout), linear_layer)
        else:
            self.head = linear_layer

    def _flatten_image(self, x):
        """Convert image to flattened format if needed."""
        if x.dim() == 4 and x.shape[1] == 1:
            return x.view(x.size(0), -1)
        elif x.dim() == 2 and x.shape[1] == self.input_dim:
            return x
        else:
            raise ValueError(
                f"Expected x as [B,1,28,28] or [B,{self.input_dim}], got {x.shape}")

    def _ensure_image_format(self, x):
        """Convert to image format for conv architecture."""
        if x.dim() == 2 and x.shape[1] == self.input_dim:
            return x.view(x.size(0), 1, 28, 28)
        elif x.dim() == 4 and x.shape[1] == 1:
            return x
        else:
            raise ValueError(
                f"Expected x as [B,{self.input_dim}] or [B,1,28,28], got {x.shape}")

    def forward(self, x, y_digit):
        """
        Forward pass with conditional input.

        Args:
            x: Input images [batch_size, 784] or [batch_size, 1, 28, 28]
            y_digit: Digit labels [batch_size, 10] (one-hot encoding of digits 0-9)

        Returns:
            Logits [batch_size, 1]
        """
        if self.arch == 'mlp':
            x = self._flatten_image(x).float()
            y_digit = y_digit.float()
            # Concatenate image + digit one-hot
            inp = torch.cat([x, y_digit], dim=1)  # [batch_size, 784+10]
            return self.net(inp)

        elif self.arch == 'conv':
            x = self._ensure_image_format(x).float()  # [batch_size, 1, 28, 28]
            B, _, H, W = x.shape

            # Spatial broadcast of digit conditions
            y_digit = y_digit.float().view(
                B, -1, 1, 1)  # [batch_size, 10, 1, 1]
            y_digit = y_digit.expand(-1, -1, H, W)  # [batch_size, 10, 28, 28]

            # Concatenate image + digit conditions
            inp = torch.cat([x, y_digit], dim=1)  # [batch_size, 11, 28, 28]

            f = self.features(inp)
            f = f.view(B, -1)
            return self.head(f)

    def score(self, x, y_digit):
        """
        Score images to get discriminator predictions.

        Args:
            x: Input images [batch_size, 784] or [batch_size, 1, 28, 28]
            y_digit: Digit labels [batch_size, 10] (one-hot encoding of digits 0-9)

        Returns:
            Scores [batch_size, 1] where higher values indicate more real-like
        """
        logits = self.forward(x, y_digit)
        return torch.sigmoid(logits)

    def _ensure_label_format(self, y):
        """
        Ensure y is in the correct format for ConditionalDiscriminator.

        Args:
            y: Labels [batch_size, 11] (first 10: digit one-hot, last 1: real/fake binary)

        Returns:
            y: Labels [batch_size, 11] (validated format)

        Raises:
            ValueError: If y format is not supported
        """
        if y.dim() != 2 or y.size(1) != 11:
            raise ValueError(
                f"ConditionalDiscriminator expects y shape [batch_size, 11], got {y.shape}")

        # Validate digit one-hot part (first 10 columns)
        digit_part = y[:, :self.num_classes]
        if not torch.all((digit_part == 0) | (digit_part == 1)):
            raise ValueError(
                "ConditionalDiscriminator digit one-hot part should contain only 0s and 1s")
        if not torch.all(digit_part.sum(dim=1) == 1):
            raise ValueError(
                "ConditionalDiscriminator digit one-hot part should have exactly one 1 per row")

        # Validate real/fake part (last column)
        real_fake_part = y[:, self.num_classes:]
        if not torch.all((real_fake_part == 0) | (real_fake_part == 1)):
            raise ValueError(
                "ConditionalDiscriminator real/fake part should contain only 0s and 1s")

        return y.float()

    def loss(self, x, y):
        """
        Compute discriminator loss compatible with train_helper functions.

        Args:
            x: Input images [batch_size, 784] or [batch_size, 1, 28, 28]
            y: Labels [batch_size, 11] (first 10: digit one-hot, last 1: real/fake binary)

        Returns:
            (total_loss, summary_dict) compatible with existing training loops
        """
        # Ensure y is in correct format
        y = self._ensure_label_format(y)

        # Binarize input images (similar to your current discriminator)
        x = (x > 0.5).float()

        # Extract digit labels (first 10) and real/fake target (last 1)
        y_digit = y[:, :self.num_classes]  # [batch_size, 10]
        target = y[:, self.num_classes:]  # [batch_size, 1]

        # Forward pass
        logits = self.forward(x, y_digit)  # [batch_size, 1]

        # Binary Cross Entropy Loss
        total_loss = F.binary_cross_entropy_with_logits(
            logits, target, reduction='sum')

        # Calculate accuracy
        predictions = (logits > 0).float()
        correct = predictions.eq(target).sum()

        summary = {
            'accuracy': correct.item(),
        }

        return total_loss, summary

    def get_name(self):
        return self.name

    def __str__(self):
        return f"CondiDisc(name={self.name}, arch={self.arch})"

    def __repr__(self):
        return f"CondiDisc(name={self.name}, arch={self.arch})"
