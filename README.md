# Fashion-MNIST Variational Autoencoder (VAE) Project

This project explores Variational Autoencoders (VAEs) for image generation and reconstruction on the Fashion-MNIST dataset. It covers data preparation, VAE implementation (both fully connected and convolutional), training, evaluation using FID score, and latent space exploration.

## Table of Contents
1.  [Project Setup](#project-setup)
2.  [Phase 1: Data Management & EDA](#phase-1-data-management--eda)
3.  [Phase 2: Base VAE Implementation & Evaluation](#phase-2-base-vae-implementation--evaluation)
4.  [Phase 3: VAE Improvements & Hyperparameter Tuning](#phase-3-vae-improvements--hyperparameter-tuning)
5.  [Key Components](#key-components)
6.  [Metrics: Fréchet Inception Distance (FID)](#metrics-fréchet-inception-distance-fid)
7.  [Results Summary](#results-summary)
8.  [Usage](#usage)

## Project Setup

To run this notebook, you will need a Python environment with the following libraries:
-   `torch`
-   `torchvision`
-   `numpy`
-   `matplotlib`
-   `seaborn`
-   `scipy`

It's recommended to use Google Colab, as the necessary `fashion_resnet18_classifier.pt` model (for FID calculation) is typically provided or can be easily uploaded.

## Phase 1: Data Management & EDA

This phase focuses on setting up the Fashion-MNIST dataset and performing initial exploratory data analysis.

**`FashionMNISTManager` Class:**
-   Handles downloading, splitting (train, validation, test), and loading the Fashion-MNIST dataset.
-   Applies `ToTensor` transformations.
-   Provides methods for reporting data statistics (`report_data_stats`), sanity checks (visualizing a batch of images `sanity_check`), and comprehensive EDA (`perform_eda`) including class distribution plots and pixel intensity histograms.

## Phase 2: Base VAE Implementation & Evaluation

This phase implements a basic Variational Autoencoder (VAE) and establishes a baseline performance.

**`BaseVAE` Class:**
-   A fully connected VAE architecture with an encoder, reparameterization trick, and a decoder.
-   Takes a `latent_dim` as input (default 2).

**`VAETrainer` Class:**
-   Manages the training and evaluation loop for VAE models.
-   Implements the VAE loss function (BCE + `beta` * KLD).
-   Provides `train` and `evaluate` methods.

**`VAEVisualizer` Class:**
-   Offers visualization tools for VAEs, including `show_reconstructions` (original vs. reconstructed images) and `generate_samples` (sampling from the latent space).

**Baseline Model Training:**
-   A `BaseVAE` with `latent_dim=2` is trained for 30 epochs with `beta=1`.
-   Reconstructions and generated samples are visualized.
-   Test metrics (Total Loss, Reconstruction Loss, KLD) are reported.

## Phase 3: VAE Improvements & Hyperparameter Tuning

This phase explores improvements to the VAE model by adjusting hyperparameters and trying a convolutional architecture.

**Improvement 1: Change Regularization Intensity (`beta`)**
-   The `BaseVAE` (latent_dim=2) is retrained with `beta=0.5` to observe the effect on reconstruction quality and sample diversity. Visualizations and metrics are compared.

**Improvement 2: Change Latent Dimension**
-   The `BaseVAE` is trained with `latent_dim=10` (using `beta=1`) to investigate how a higher-dimensional latent space impacts learning. Visualizations and metrics are provided.

**Improvement 3: Convolutional Architecture (`ConvVAE`)**
-   A `ConvVAE` model is introduced, leveraging convolutional layers for encoding and transposed convolutional layers for decoding, which are generally more effective for image data.
-   **`ConvVAE` Class:**
    -   Encoder: `Conv2d` layers followed by `Flatten`.
    -   Decoder: `Linear` layer, `Reshape`, then `ConvTranspose2d` layers.
-   The `ConvVAE` (latent_dim=2) is trained (using `beta=0.5`). Visualizations and metrics are presented.

**Latent Traversal Study (Regularization Weight `beta` Impact):**
-   A deeper dive into the effect of `beta` with `latent_dim=10`.
-   Models are trained for `beta` values of `0.5`, `1.0`, and `4.0`.
-   For each `beta`, latent traversal grids are generated to visualize how varying individual latent dimensions affects the generated output, providing insight into the disentanglement capabilities of the VAE.

## Phase 4: Conditional VAE (label-controlled generation)
-   A `ConditionalVAE` (CVAE) is implemented, extending the VAE to incorporate class labels during both encoding and decoding.
-   This allows for controlled image generation based on specific Fashion-MNIST classes.
-   The CVAE is trained with `latent_dim=2` and `beta=1`.
-   Generated images for each class are displayed in a grid.
-   A pre-trained `FashionResNet18` classifier evaluates the classification accuracy of these generated images, demonstrating the CVAE's ability to produce class-specific outputs.

## Key Components

-   **`FashionMNISTManager`**: Data handling and EDA.
-   **`BaseVAE`**: Fully connected VAE model.
-   **`ConvVAE`**: Convolutional VAE model.
-   **`ConditionalVAE`**: VAE with class-conditional generation.
-   **`VAETrainer`**: Training and evaluation utility for `BaseVAE` and `ConvVAE`.
-   **`ConditionalVAETrainer`**: Training and evaluation utility for `ConditionalVAE`.
-   **`VAEVisualizer`**: Image reconstruction and generation visualization.
-   **`FashionResNet18`**: Pre-trained classifier used for FID calculation and CVAE evaluation.

## Metrics: Fréchet Inception Distance (FID)

The FID score is the primary metric used to evaluate the quality of generated images. It measures the distance between the feature distributions of real and generated images using a pre-trained Inception-v3 network (here, `FashionResNet18`).

$$FID = \|\mu_r - \mu_g\|^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2\sqrt{\Sigma_r \Sigma_g})$$

-   **`calculate_fid` function**: Computes the FID score.

## Results Summary

| Model                  | Latent Dim | Beta | Total Loss (Test) | Reconstruction Loss | KLD Loss | FID Score |
| :--------------------- | :--------- | :--- | :---------------- | :------------------ | :------- | :-------- |
| **BaseVAE (Baseline)** | 2          | 1.0  | 261.66            | 255.22              | 6.44     | 23.8946   |
| BaseVAE (Beta 0.5)     | 2          | 0.5  | 258.67            | 254.86              | 7.61     |           |
| BaseVAE (Latent Dim 10)| 10         | 1.0  | 241.50            | 225.89              | 15.61    |           |
| ConvVAE                | 2          | 0.5  | 262.40            | 258.90              | 7.01     |           |
| ConditionalVAE         | 2          | 1.0  | (Approx 257.2)    | (Approx 251.9)      | (Approx 5.3) |           |

*Note: FID scores for all improved models would typically be calculated and compared. For ConditionalVAE, training loss at Epoch 3 was used as an approximation for test metrics in this summary.* (Only baseline FID is available in the provided context).

## Usage

To run this project, simply execute the cells in the provided Jupyter/Colab notebook sequentially. Ensure that the `fashion_resnet18_classifier.pt` file is accessible in your environment for FID calculation.

