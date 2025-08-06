import sys
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

sample_size = int(sys.argv[1])
filter_threshold = float(sys.argv[2])

############################ real data training ############################
sys.path.append("/home/qiyuanliu/data_filter/Verified-Synthetic-Data/MNIST")

import torch.nn.functional as F
from cvae_model import CVAE, cvae_loss
from torch.utils.data import Subset

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
latent_dim = 20
label_dim = 10
batch_size = 128
epochs = 200
lr = 1e-3
# Load MNIST
transform = transforms.ToTensor()
full_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_dataset = Subset(full_dataset, range(sample_size))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize model
model = CVAE(latent_dim=latent_dim, label_dim=label_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# One-hot encoding helper
def one_hot(labels, num_classes=10):
    return F.one_hot(labels, num_classes).float()

best_val_loss = float('inf')
patience = 5
trigger_times = 0

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for x, y in train_loader:
        x = x.view(-1, 784).to(device)
        y = one_hot(y).to(device)

        optimizer.zero_grad()
        recon_x, mu, logvar = model(x, y)
        loss = cvae_loss(recon_x, x, mu, logvar)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_loss:.4f}")

    # Early stopping based on training loss
    if avg_loss < best_train_loss:
        best_train_loss = avg_loss
        trigger_times = 0
        torch.save(model.state_dict(), "best_model.pt")
    else:
        trigger_times += 1
        print(f"EarlyStopping counter: {trigger_times} out of {patience}")
        if trigger_times >= patience:
            print("Early stopping triggered.")
            break

# save the model to model_saved folder
import os
os.chdir("/home/qiyuanliu/data_filter/Verified-Synthetic-Data/MNIST")  
torch.save(model.state_dict(), f"model_saved/cvae_mnist_{sample_size}.pth")
print(f"Model saved to model_saved/cvae_mnist_{sample_size}.pth")

############################ generate synthetic data ############################

sys.path.append("/home/qiyuanliu/data_filter/Verified-Synthetic-Data/MNIST")
import torch
from cvae_model import CVAE 
import os
latent_dim = 20
label_dim = 10
model = CVAE(latent_dim=latent_dim, label_dim=label_dim)
os.chdir("/home/qiyuanliu/data_filter/Verified-Synthetic-Data/MNIST")  
model.load_state_dict(torch.load(f"model_saved/cvae_mnist_{sample_size}.pth"))
model.eval()

def generate_images_in_batches(model, total_samples, latent_dim, num_classes, batch_size=10000, device='cuda'):
    model.eval()
    generated_images = []
    all_labels = []

    for start in range(0, total_samples, batch_size):
        end = min(start + batch_size, total_samples)
        batch_size_actual = end - start

        # Generate z and y
        z = torch.randn(batch_size_actual, latent_dim).to(device)
        y = torch.arange(num_classes).repeat_interleave(total_samples // num_classes)[start:end]
        y_onehot = F.one_hot(y, num_classes=num_classes).float().to(device)

        with torch.no_grad():
            imgs = model.decode(z, y_onehot).view(-1, 1, 28, 28).cpu()
            generated_images.append(imgs)
            all_labels.append(y)

    images = torch.cat(generated_images, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return images, labels

# large sample size for training
latent_dim = model.latent_dim
device = next(model.parameters()).device
gen_imgs,y = generate_images_in_batches(
    model=model,
    total_samples=6000000,
    latent_dim=latent_dim,
    num_classes=10,
    batch_size=10000,
    device=device
)

os.chdir("/home/qiyuanliu/data_filter/Verified-Synthetic-Data/MNIST")  
save_path = f"data_saved/synthetic_mnist_cvae_{sample_size}.pt"
torch.save({
    'images': gen_imgs,    # Tensor [6000000, 1, 28, 28]
    'labels': y            # Tensor [6000000]
}, save_path)

# smaller sample size for evaluation
gen_imgs,y = generate_images_in_batches(
    model=model,
    total_samples=6000,
    latent_dim=latent_dim,
    num_classes=10,
    batch_size=10000,
    device=device
)

os.chdir("/home/qiyuanliu/data_filter/Verified-Synthetic-Data/MNIST")  
save_path = f"data_saved/synthetic_mnist_cvae_{sample_size}_2.pt"
torch.save({
    'images': gen_imgs,    # Tensor [6000, 1, 28, 28]
    'labels': y            # Tensor [6000]
}, save_path)

############################ filter synthetic data ############################

import torch
from torch.utils.data import DataLoader
from discriminator import Discriminator
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D = Discriminator().to(device)
os.chdir("/home/qiyuanliu/data_filter/Verified-Synthetic-Data/MNIST")
D.load_state_dict(torch.load("model_saved/discriminator_mnist_cvae_2.pth"))
D.eval()

data = torch.load(f"data_saved/synthetic_mnist_cvae_{sample_size}.pt")
synthetic_images = data['images']  

synthetic_loader = DataLoader(synthetic_images, batch_size=512)

all_probs = []

with torch.no_grad():
    for batch in synthetic_loader:
        batch = batch.to(device)
        probs = D(batch)  # [batch_size, 1], already sigmoid activated
        all_probs.append(probs.cpu())

all_probs = torch.cat(all_probs, dim=0)
# Flatten probs to shape [N]
probs = all_probs.squeeze(1)

# Load images and labels
images = data['images']      # [N, 1, 28, 28]
labels = data['labels']      # [N]

# Create mask for p > 0.5
mask = probs > filter_threshold

# Apply mask
filtered_images = images[mask]
filtered_labels = labels[mask]

print(f"Selected {filtered_images.shape[0]} samples with p > 0.4")
# Save to file
torch.save({
    'images': filtered_images,
    'labels': filtered_labels
}, f"data_saved/synthetic_mnist_filtered_pgt{filter_threshold}_{sample_size}.pt")


############################ synthetic data retraining ############################
from torch.utils.data import TensorDataset

images = filtered_images  # shape: [N, 1, 28, 28]
labels = filtered_labels  # shape: [N]

print(f"Loaded {images.shape[0]} filtered synthetic samples")

# Preprocess: flatten images and convert labels to one-hot
images = images.view(-1, 784)  # flatten to [N, 784]
labels_onehot = F.one_hot(labels, num_classes=label_dim).float()

# Create dataset and dataloader
dataset = TensorDataset(images, labels_onehot)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model
model = CVAE(latent_dim=latent_dim, label_dim=label_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for x, y in train_loader:
        x = x.view(-1, 784).to(device)
        y = one_hot(y).to(device)

        optimizer.zero_grad()
        recon_x, mu, logvar = model(x, y)
        loss = cvae_loss(recon_x, x, mu, logvar)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_loss:.4f}")

    # Early stopping based on training loss
    if avg_loss < best_train_loss:
        best_train_loss = avg_loss
        trigger_times = 0
        torch.save(model.state_dict(), "best_model.pt")
    else:
        trigger_times += 1
        print(f"EarlyStopping counter: {trigger_times} out of {patience}")
        if trigger_times >= patience:
            print("Early stopping triggered.")
            break

# save the model to model_saved folder
os.chdir("/home/qiyuanliu/data_filter/Verified-Synthetic-Data/MNIST")
torch.save(model.state_dict(), f"model_saved/cvae_mnist_filtered_synthetic_data_{sample_size}.pth")

# use the new model to generate synthetic data for evaluation
model.eval()

n_per_class = 60000
num_classes = 10
total_samples = n_per_class * num_classes
latent_dim = model.latent_dim
device = next(model.parameters()).device

z = torch.randn(total_samples, latent_dim).to(device)
y = torch.arange(num_classes).repeat_interleave(n_per_class)
y_onehot = F.one_hot(y, num_classes=num_classes).float().to(device)

with torch.no_grad():
    gen_imgs = model.decode(z, y_onehot).view(-1, 1, 28, 28).cpu()  # shape: [60000, 1, 28, 28]

# save the generated images and labels
os.chdir("/home/qiyuanliu/data_filter/Verified-Synthetic-Data/MNIST")
save_path = f"data_saved/synthetic_mnist_cvae_filtered_synthetic_model_generated_data_{sample_size}.pt"
torch.save({"images": gen_imgs, "labels": y}, save_path)

############################ Model Evaluation ############################

# FID

from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import TensorDataset
import os
import sys
import torch

sys.path.append("/home/qiyuanliu/data_filter/Verified-Synthetic-Data/MNIST")

from FID import calculate_fid_score

transform = transforms.ToTensor()

real_ds = MNIST(root='./data', train=False, download=True, transform=transform)
os.chdir("/home/qiyuanliu/data_filter/Verified-Synthetic-Data/MNIST")
### Synthetic dataset
synthetic = torch.load(f"data_saved/synthetic_mnist_cvae_{sample_size}_2.pt")
synthetic_ds = TensorDataset(synthetic['images'], torch.zeros(len(synthetic['images'])))
fid_value = calculate_fid_score(real_ds, synthetic_ds)
print(f"FID Score(real data and synthetic data): {fid_value:.2f}")

## filtered synthetic data
synthetic = torch.load(f"data_saved/synthetic_mnist_filtered_pgt{filter_threshold}_{sample_size}.pt")
synthetic_ds = TensorDataset(synthetic['images'], torch.zeros(len(synthetic['images'])))
fid_value = calculate_fid_score(real_ds, synthetic_ds)
print(f"FID Score(real data and filtered synthetic data): {fid_value:.2f}")

# Synthetic dataset
synthetic = torch.load(f"data_saved/synthetic_mnist_cvae_filtered_synthetic_model_generated_data_{sample_size}.pt")
synthetic_ds = TensorDataset(synthetic['images'], torch.zeros(len(synthetic['images'])))
fid_value = calculate_fid_score(real_ds, synthetic_ds)
print(f"FID Score(real data and model 2 synthetic data): {fid_value:.2f}")

# Reconstruction Loss

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from cvae_model import CVAE, cvae_loss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load model
model = CVAE(latent_dim=20, label_dim=10).to(device)
model.load_state_dict(torch.load(f"model_saved/cvae_mnist_{sample_size}.pth"))
model.eval()

# Load test set
test_dataset = datasets.MNIST(root="./data", train=False, transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Evaluation
total_loss = 0
total_recon_loss = 0
total_kl = 0
num_samples = 0

with torch.no_grad():
    for x, y in test_loader:
        x = x.view(-1, 784).to(device)
        y = F.one_hot(y, num_classes=10).float().to(device)

        recon_x, mu, logvar = model(x, y)
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = BCE + KLD

        total_loss += loss.item()
        total_recon_loss += BCE.item()
        total_kl += KLD.item()
        num_samples += x.size(0)

print(f"Test Set Results(model1):")
print(f"  Avg CVAE Loss: {total_loss / num_samples:.4f}")
print(f"  Avg Reconstruction (BCE) Loss: {total_recon_loss / num_samples:.4f}")
print(f"  Avg KL Divergence: {total_kl / num_samples:.4f}")

# get the loss of synthetic model
import sys
sys.path.append("/home/qiyuanliu/data_filter/Verified-Synthetic-Data/MNIST")
import torch
from cvae_model import CVAE 
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

latent_dim = 20
label_dim = 10
model = CVAE(latent_dim=latent_dim, label_dim=label_dim).to(device)
#os.chdir("/home/qiyuanliu/data_filter/Verified-Synthetic-Data/MNIST")  
os.chdir("/home/qiyuanliu/data_filter/Verified-Synthetic-Data/MNIST")
model.load_state_dict(torch.load(f"model_saved/cvae_mnist_filtered_synthetic_data_{sample_size}.pth"))
model.eval()


# Load test set
test_dataset = datasets.MNIST(root="./data", train=False, transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Evaluation
total_loss = 0
total_recon_loss = 0
total_kl = 0
num_samples = 0

with torch.no_grad():
    for x, y in test_loader:
        x = x.view(-1, 784).to(device)
        y = F.one_hot(y, num_classes=10).float().to(device)

        recon_x, mu, logvar = model(x, y)
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = BCE + KLD

        total_loss += loss.item()
        total_recon_loss += BCE.item()
        total_kl += KLD.item()
        num_samples += x.size(0)

print(f"Test Set Results(model2):")
print(f"  Avg CVAE Loss: {total_loss / num_samples:.4f}")
print(f"  Avg Reconstruction (BCE) Loss: {total_recon_loss / num_samples:.4f}")
print(f"  Avg KL Divergence: {total_kl / num_samples:.4f}")
