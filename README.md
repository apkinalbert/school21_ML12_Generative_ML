# Generative Models

This project explores image generation on MNIST with variational autoencoders, generative adversarial networks, and a diffusion model. It includes custom PyTorch models, conditional generation, latent space visualizations, and step-by-step diffusion sampling.

## Installation

Install all required dependencies:

```bash
pip3 install -r requirements.txt
```

The `-r` option installs the dependencies listed in `requirements.txt`.

Open the notebook and run its cells in order after installing the dependencies.

## What was done

* Prepared MNIST and split the data into **training, validation, and test sets**.
* Built and trained **VAEs** with different latent dimensions and visualized their latent spaces and generated samples.
* Implemented a **conditional VAE** with one-hot labels to control the generated digit.
* Trained a **GAN** with Fourier features and examined generator and discriminator losses.
* Implemented **gradient reversal** to train both GAN networks with a single backward pass.
* Built a **conditional GAN** and compared different digit labels using the same latent input.
* Implemented **DDPM with a custom U-Net** and visualized how an image develops from noise.
* Created a shared training framework with **validation, early stopping, checkpoint selection, and image callbacks**.

## Main techniques

`VAE` · `ELBO` · `KL Divergence` · `Reparameterization` · `GAN` · `Conditional Generation` · `One-Hot Encoding` · `Fourier Features` · `Gradient Reversal` · `DDPM` · `U-Net` · `BCE` · `MSE` · `AdamW` · `PyTorch`

## Dataset

* **MNIST** — grayscale images of handwritten digits from zero to nine, with a resolution of **28 × 28 pixels**. The dataset is downloaded through `torchvision`.
