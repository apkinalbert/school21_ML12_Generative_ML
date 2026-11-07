import torch
import torch.nn as nn


def vae_loss(
    logits: torch.Tensor, images: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    batch_size = logits.shape[0]
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    reconstruction = criterion(logits, images) / batch_size
    kl = torch.sum(mu**2 + torch.exp(logvar) - 1 - logvar) / (2 * batch_size)
    loss = reconstruction + kl

    return loss, reconstruction, kl


def vae_train_step(
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
) -> dict[str, torch.Tensor]:
    images, labels = batch
    optimizer.zero_grad(set_to_none=True)

    logits, mu, logvar = model(images)

    loss, reconstruction, kl = vae_loss(
        logits=logits,
        images=images,
        mu=mu,
        logvar=logvar,
    )

    loss.backward()
    optimizer.step()

    return {
        "loss": loss.detach(),
        "reconstruction": reconstruction.detach(),
        "kl": kl.detach(),
    }


def vae_validation_step(
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
) -> dict[str, torch.Tensor]:
    images, labels = batch

    logits, mu, logvar = model(images)

    loss, reconstruction, kl = vae_loss(
        logits=logits,
        images=images,
        mu=mu,
        logvar=logvar,
    )

    return {
        "loss": loss.detach(),
        "reconstruction": reconstruction.detach(),
        "kl": kl.detach(),
    }


def conditional_vae_train_step(
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
) -> dict[str, torch.Tensor]:
    images, labels = batch
    optimizer.zero_grad(set_to_none=True)

    logits, mu, logvar = model(images, labels)

    loss, reconstruction, kl = vae_loss(
        logits=logits, images=images, mu=mu, logvar=logvar
    )
    loss.backward()

    optimizer.step()

    return {
        "loss": loss.detach(),
        "reconstruction": reconstruction.detach(),
        "kl": kl.detach(),
    }


def conditional_vae_validation_step(
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
) -> dict[str, torch.Tensor]:
    images, labels = batch
    logits, mu, logvar = model(images, labels)
    loss, reconstruction, kl = vae_loss(
        logits=logits, images=images, mu=mu, logvar=logvar
    )

    return {
        "loss": loss.detach(),
        "reconstruction": reconstruction.detach(),
        "kl": kl.detach(),
    }
