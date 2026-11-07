import torch


@torch.no_grad()
def sliced_wasserstein(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    directions: torch.Tensor,
) -> torch.Tensor:

    if real_images.shape != fake_images.shape:
        raise ValueError("Real and fake image shapes must match")

    real = real_images.flatten(start_dim=1).float()
    fake = fake_images.flatten(start_dim=1).float()

    real_projections = real @ directions
    fake_projections = fake @ directions

    real_sorted = real_projections.sort(dim=0).values
    fake_sorted = fake_projections.sort(dim=0).values

    return (real_sorted - fake_sorted).abs().mean()
