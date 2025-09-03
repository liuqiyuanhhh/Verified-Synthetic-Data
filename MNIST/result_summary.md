# Conditional VAE  


## Experiment Setup

- **Model 0**: A CVAE trained on 5,000 real MNIST samples.  
- **Discriminator**: Trained using 60,000 real samples and 60,000 synthetic samples.  
- **Model 1 (with filtering)**: Generate 700,000 synthetic samples; for each digit, select the top 1.5% according to the discriminator score (10,500 samples retained) and use them for training.  
- **Model 1 (without filtering)**: Generate 10,500 synthetic samples at random and use them directly for training.  


## Visualize

| Real | Filter | No Filter |
|------|--------|-----------|
| <img width="2400" height="3000" alt="model_00_real_grid" src="https://github.com/user-attachments/assets/815e7f30-9b9f-4131-adb1-c6831f21b686" /> | <img width="2400" height="3000" alt="model_01_grid" src="https://github.com/user-attachments/assets/96ed0539-dbb2-46e4-b655-e5efd0cdc686" /> | <img width="2400" height="3000" alt="no_filter" src="https://github.com/user-attachments/assets/9785e97c-c14b-48c8-9d7a-d7b742fe4498" /> |

## FID

<img width="688" height="490" alt="compare" src="https://github.com/user-attachments/assets/bf550b89-0d76-4ba8-be28-8d2bf518c2e4" />
