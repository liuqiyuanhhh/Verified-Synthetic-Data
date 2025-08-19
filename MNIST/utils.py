import torch
import matplotlib.pyplot as plt

# Plot n random samples for each digit
def plot_samples_per_digit(num_samples, model):
    fig, axes = plt.subplots(10, num_samples, figsize=(20, 20))
    fig.suptitle("10 Random Samples for Each Digit (0-9)", fontsize=16)
    
    for digit in range(10):
        # Find indices for this digit
        samples = model.sample_x_given_y(digit, num_samples)
            
        for i in range(num_samples):
            img = samples[i].view(28,28).cpu().detach().numpy()
            axes[digit, i].imshow(img, cmap='gray')
            axes[digit, i].set_title(f'Digit {digit}')
            axes[digit, i].axis('off')
    
    plt.tight_layout()
    plt.show()
