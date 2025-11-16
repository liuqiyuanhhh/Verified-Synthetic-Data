# Experimental Setup

The experiment combines initial GAN training, ResNet-based feature filtering, and iterative retraining rounds to progressively improve synthetic image quality.

---

# High-Level Verifier Overview

1. **Feature Extraction (ResNet18)**
   - Use an ImageNet-pretrained ResNet18 as a fixed feature extractor.
   - Replace the final FC layer with `Identity` so the output is a 512-D embedding.
   - For each image (real or synthetic):
     - Input format: `[-1, 1]` range, shape `(3, H, W)`.
     - Rescale to `[0, 1]` and resize to `224 × 224`.
     - Normalize with ImageNet mean and std.
     - Run through ResNet18 and L2-normalize the 512-D output.
   - Result: each image → a unit-norm 512-D feature vector.

2. **Build Real Feature Reference**
   - Sample `verifier_samples` real CelebA images.
   - Compute their 512-D ResNet18 features.
   - Concatenate into a matrix `real_feats` with shape `(N_real, 512)`.
   - This matrix is the **reference feature bank** representing the real data manifold.

3. **Scoring Synthetic Images**
   - Generate a large batch of synthetic images using the current Generator `G`.
   - For each synthetic image:
     - Compute its 512-D feature `f_fake` via the same ResNet18 pipeline.
     - Compute L2 distances to all real features:
       \[
       d_i = \| f_\text{fake} - f_{\text{real}, i} \|_2
       \]
     - Take the **minimum** distance:
       \[
       d_{\min} = \min_i d_i
       \]
     - Define the **score** as the negative distance:
       \[
       \text{score} = -d_{\min}
       \]
       - Higher score = closer to some real image = more “real-like”.

4. **Chunked Processing (Memory-Friendly)**
   - Total synthetic samples `synth_total` are split into `verifier_chunks`.
   - For each chunk:
     - Generate `chunk_total` fake images.
     - Compute scores for all images in that chunk.
     - Keep the top `keep_ratio` fraction (e.g., best 10% by score).
   - Concatenate all kept images from all chunks to form the **filtered synthetic dataset**.

5. **Verifier-Filtered Retraining**
   - Train the GAN initially on a real subset (no verifier).
   - For each round:
     1. Generate `synth_total` synthetic images with the current generator.
     2. Use the ResNet18 verifier to score and keep only the top `keep_ratio` fraction.
     3. Retrain the GAN **only** on this filtered synthetic dataset.
