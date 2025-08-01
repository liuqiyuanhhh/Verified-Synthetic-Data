### Conditional VAE  
#### MNIST Data

We begin with ~60,000 real MNIST images. From this dataset, we use 40,000 samples to train the CVAE model. After training, we generate 60,000 synthetic images(6,000 samples per digit) by sampling from the model.

To evaluate the quality of the generated data, we train a discriminator to classify whether an image is real or synthetic, using all 60,000 real images and the 60,000 generated samples. (Serve as a stronger verifier that trained on larger dataset)

Among the synthetic samples, only about **1%** receive a discriminator probability greater than 0.5, indicating they are classified as highly realistic. Specifically, we find:
| Digit | Count | Proportion (%) |
|-------|-------|----------------|
|   0   |  98   |     16.3%      |
|   1   |  95   |     15.8%      |
| **2** |  22   |     **3.6%**   |
|   3   |  52   |      8.6%      |
|   4   |  54   |      9.0%      |
| **5** |   9   |     **1.5%**   |
|   6   |  67   |     11.1%      |
|   7   |  74   |     12.3%      |
| **8** |  20   |     **3.3%**   |
|   9   | 112   |     18.6%      |

Next, we generate 6,000,000 synthetic images (600,000 samples per digit) using the conditional VAE. We then filter the synthetic data by retaining only samples with discriminator confidence $p>0.5$, resulting in **75,242** high-quality synthetic samples. When training the CVAE on 40,000 real MNIST images, we obtain a Fréchet Inception Distance (FID) score of **0.02**. In comparison, training the same model on the filtered synthetic data yields an FID score of **0.08**.

<p align="center">
  <img src="https://github.com/user-attachments/assets/8d0d63da-760e-4a7f-92e5-be349b33dae6" width="300" alt="Model 1"/>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://github.com/user-attachments/assets/31fc28b9-b06b-4a20-8f7f-404873bd8ad1" width="300" alt="Model 2"/>
</p>

<p align="center">
  <b>Model 1</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>Model 2</b>
</p>


Visually, Model 2 produces more realistic results compared to Model 1. **Why the FID is larger?**


#### Thoughts
1. FID measures the dissimilarity between the distribution of generated data and real data, capturing both the quality and diversity of the generated samples.
2. Are there any other criteria to evaluate the new model?




