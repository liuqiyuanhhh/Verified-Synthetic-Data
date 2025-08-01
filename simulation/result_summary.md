### Conditional VAE  
#### MNIST Data

We begin with ~60,000 real MNIST images. From this dataset, we use 40,000 samples to train the CVAE model. After training, we generate 60,000 synthetic images by sampling from the model.

To evaluate the quality of the generated data, we train a discriminator to classify whether an image is real or synthetic, using all 60,000 real images and the 60,000 generated samples. (Serve as a stronger verifier that trained on larger data)

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







