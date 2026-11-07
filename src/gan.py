import torch
import torch.nn as nn
from models import GAN, ConditionalGAN
from configs import TrainConfig


def gan_grl_train_step(
    model: GAN,
    batch: tuple[torch.Tensor, torch.Tensor],
    optimizers: dict[str, torch.optim.Optimizer],
) -> dict[str, torch.Tensor]:
    images, labels = batch

    g_optimizer = optimizers["generator"]
    d_optimizer = optimizers["discriminator"]

    z = torch.randn(images.shape[0], model.latent_dim, device=images.device)

    criterion = nn.BCEWithLogitsLoss()

    d_optimizer.zero_grad(set_to_none=True)
    g_optimizer.zero_grad(set_to_none=True)

    fake_images = model.generate(z)

    fake_images_grl = 2 * fake_images.detach() - fake_images
    fake_logits = model.discriminate(fake_images_grl)

    real_logits = model.discriminate(images)

    ones = torch.ones_like(real_logits)
    zeros = torch.zeros_like(fake_logits)

    real_loss = criterion(real_logits, ones)
    fake_loss = criterion(fake_logits, zeros)

    loss = (real_loss + fake_loss) / 2

    loss.backward()
    d_optimizer.step()
    g_optimizer.step()

    return {
        "discriminator_loss": loss.detach(),
        "generator_loss": criterion(fake_logits.detach(), ones),
    }


def gan_train_step(
    model: GAN,
    batch: tuple[torch.Tensor, torch.Tensor],
    optimizers: dict[str, torch.optim.Optimizer],
) -> dict[str, torch.Tensor]:
    images, labels = batch

    g_optimizer = optimizers["generator"]
    d_optimizer = optimizers["discriminator"]

    z = torch.randn(images.shape[0], model.latent_dim, device=images.device)

    criterion = nn.BCEWithLogitsLoss()

    d_optimizer.zero_grad(set_to_none=True)

    fake_images = model.generate(z)

    real_logits = model.discriminate(images)
    fake_logits = model.discriminate(fake_images.detach())

    ones = torch.ones_like(real_logits)
    zeros = torch.zeros_like(fake_logits)

    real_loss = criterion(real_logits, ones)
    fake_loss = criterion(fake_logits, zeros)

    d_loss = (real_loss + fake_loss) / 2

    d_loss.backward()
    d_optimizer.step()

    g_optimizer.zero_grad(set_to_none=True)

    ones = torch.ones_like(real_logits)
    fake_logits = model.discriminate(fake_images)
    g_loss = criterion(fake_logits, ones)

    g_loss.backward()
    g_optimizer.step()

    return {
        "generator_loss": g_loss.detach(),
        "discriminator_loss": d_loss.detach(),
    }


def gan_validation_step(
    model: GAN,
    batch: tuple[torch.Tensor, torch.Tensor],
) -> dict[str, torch.Tensor]:
    images, labels = batch

    z = torch.randn(images.shape[0], model.latent_dim, device=images.device)

    criterion = nn.BCEWithLogitsLoss()

    fake_images = model.generate(z)

    real_logits = model.discriminate(images)
    fake_logits = model.discriminate(fake_images)

    ones = torch.ones_like(real_logits)
    zeros = torch.zeros_like(fake_logits)

    real_loss = criterion(real_logits, ones)
    fake_loss = criterion(fake_logits, zeros)

    d_loss = (real_loss + fake_loss) / 2
    g_loss = criterion(fake_logits, ones)

    return {
        "generator_loss": g_loss.detach(),
        "discriminator_loss": d_loss.detach(),
        "real_score": torch.sigmoid(real_logits).mean(),
        "fake_score": torch.sigmoid(fake_logits).mean(),
    }


def make_gan_optimizers(
    model: ConditionalGAN,
    config: TrainConfig,
) -> dict[str, torch.optim.Optimizer]:

    use_cuda = next(model.parameters()).is_cuda

    return {
        "generator": torch.optim.AdamW(
            model.generator.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            fused=use_cuda,
            betas=(0.5, 0.999),
        ),
        "discriminator": torch.optim.AdamW(
            model.discriminator.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            fused=use_cuda,
            betas=(0.5, 0.999),
        ),
    }


def conditional_gan_train_step(
    model: ConditionalGAN,
    batch: tuple[torch.Tensor, torch.Tensor],
    optimizers: dict[str, torch.optim.Optimizer],
) -> dict[str, torch.Tensor]:
    images, labels = batch

    g_optimizer = optimizers["generator"]
    d_optimizer = optimizers["discriminator"]

    z = torch.randn(images.shape[0], model.latent_dim, device=images.device)

    criterion = nn.BCEWithLogitsLoss()

    d_optimizer.zero_grad(set_to_none=True)

    fake_images = model.generate(z, labels)

    real_logits = model.discriminate(images, labels)
    fake_logits = model.discriminate(fake_images.detach(), labels)

    ones = torch.ones_like(real_logits)
    zeros = torch.zeros_like(fake_logits)

    real_loss = criterion(real_logits, ones)
    fake_loss = criterion(fake_logits, zeros)

    d_loss = (real_loss + fake_loss) / 2

    d_loss.backward()
    d_optimizer.step()

    g_optimizer.zero_grad(set_to_none=True)

    ones = torch.ones_like(real_logits)
    fake_logits = model.discriminate(fake_images, labels)
    g_loss = criterion(fake_logits, ones)

    g_loss.backward()
    g_optimizer.step()

    return {
        "generator_loss": g_loss.detach(),
        "discriminator_loss": d_loss.detach(),
    }


def conditional_gan_validation_step(
    model: GAN,
    batch: tuple[torch.Tensor, torch.Tensor],
) -> dict[str, torch.Tensor]:
    images, labels = batch

    z = torch.randn(images.shape[0], model.latent_dim, device=images.device)

    criterion = nn.BCEWithLogitsLoss()

    fake_images = model.generate(z, labels)

    real_logits = model.discriminate(images, labels)
    fake_logits = model.discriminate(fake_images, labels)

    ones = torch.ones_like(real_logits)
    zeros = torch.zeros_like(fake_logits)

    real_loss = criterion(real_logits, ones)
    fake_loss = criterion(fake_logits, zeros)

    d_loss = (real_loss + fake_loss) / 2
    g_loss = criterion(fake_logits, ones)

    return {
        "generator_loss": g_loss.detach(),
        "discriminator_loss": d_loss.detach(),
        "real_score": torch.sigmoid(real_logits).mean(),
        "fake_score": torch.sigmoid(fake_logits).mean(),
    }
