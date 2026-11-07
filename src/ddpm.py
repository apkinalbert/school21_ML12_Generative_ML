import torch
import torch.nn as nn
from models import DDPM


def ddpm_train_step(
    model: DDPM,
    batch: tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
) -> dict[str, torch.Tensor]:
    images, labels = batch
    images = images * 2 - 1

    criterion = nn.MSELoss()

    optimizer.zero_grad(set_to_none=True)

    timesteps = torch.randint(
        low=0,
        high=model.num_steps,
        size=(images.shape[0],),
        device=images.device,
    )

    noise = torch.randn_like(images)
    noisy_images = model.add_noise(images, timesteps, noise)

    predicted_noise = model(noisy_images, timesteps)
    loss = criterion(predicted_noise, noise)

    loss.backward()
    optimizer.step()

    return {"loss": loss.detach()}


def ddpm_validation_step(
    model: DDPM,
    batch: tuple[torch.Tensor, torch.Tensor],
) -> dict[str, torch.Tensor]:
    images, labels = batch
    images = images * 2 - 1

    criterion = nn.MSELoss()

    timesteps = torch.randint(
        low=0,
        high=model.num_steps,
        size=(images.shape[0],),
        device=images.device,
    )

    noise = torch.randn_like(images)
    noisy_images = model.add_noise(images, timesteps, noise)

    predicted_noise = model(noisy_images, timesteps)
    loss = criterion(predicted_noise, noise)

    return {"loss": loss.detach()}
