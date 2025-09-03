# Conditional VAE  

<!-- 
#### MNIST Data
Model 0: Train CVAE with 5000 real samples.  
Discriminator: Train the discriminator with 60,000 real samples and 60,000 synthetic samples.  
Model 1 with filtered data: generate 700,000 synthetic samples. Each digit select top 1.5%. 10,500 data remains, using it to train.  
Model 1 without filtering: generate 10,500 data and train.  
-->

## Visualize

| Real | Filter | No Filter |
|------|--------|-----------|
| <img width="2400" height="3000" alt="model_00_real_grid" src="https://github.com/user-attachments/assets/815e7f30-9b9f-4131-adb1-c6831f21b686" /> | <img width="2400" height="3000" alt="model_01_grid" src="https://github.com/user-attachments/assets/96ed0539-dbb2-46e4-b655-e5efd0cdc686" /> | <img width="2400" height="3000" alt="no_filter" src="https://github.com/user-attachments/assets/9785e97c-c14b-48c8-9d7a-d7b742fe4498" /> |

## FID

<img width="688" height="490" alt="compare" src="https://github.com/user-attachments/assets/bf550b89-0d76-4ba8-be28-8d2bf518c2e4" />
