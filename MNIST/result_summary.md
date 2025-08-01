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

Next, we generate 6,000,000 synthetic images (600,000 samples per digit) using the conditional VAE. We then filter the synthetic data by retaining only samples with discriminator confidence $p>0.5$, resulting in **75,242** high-quality synthetic samples. When training the CVAE on 40,000 real MNIST images, we obtain a Fréchet Inception Distance (FID) score of **0.02**. The filtered data and the real data's FID is **0.03**. In comparison, training the same model on the filtered synthetic data yields an FID score of **0.08**.
<table align="center">
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/f8497b06-315a-487c-af31-0962671b48c1" width="300" alt="Real Data"/>
      <div><b>Real Data</b></div>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/3390d731-5725-4185-bc02-9079c2eb1e19" width="300" alt="Model 1"/>
      <div><b>Model 1</b></div>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/e9c77de7-7ec0-4ba5-8624-60f50252f25d" width="300" alt="Filtered Model 1"/>
      <div><b>Filtered Model 1</b></div>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/06692d9f-18df-42da-809a-8cd01dacaa24" width="300" alt="Model 2"/>
      <div><b>Model 2</b></div>
    </td>
  </tr>
</table>

Visually, Model 2 produces more realistic results compared to Model 1. **Why the FID is larger?**


#### Thoughts
1. FID measures the dissimilarity between the distribution of generated data and real data, capturing both the quality and diversity of the generated samples.
2. Are there any other criteria to evaluate the new model?




