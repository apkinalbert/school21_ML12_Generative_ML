import torch.nn as nn
import torch

import math


class VAE(nn.Module):
    def __init__(
        self,
        hidden_dims: int | list[int] = 256,
        latent_dim: int = 2,
        image_shape: tuple[int, int, int] = (1, 28, 28),
    ) -> None:
        super().__init__()

        self.image_shape = image_shape
        self.input_dim = math.prod(image_shape)

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        self.encoder = self._make_encoder(hidden_dims=hidden_dims)

        in_features = hidden_dims[-1]

        self.mu = nn.Linear(in_features=in_features, out_features=latent_dim)
        self.logvar = nn.Linear(in_features=in_features, out_features=latent_dim)

        self.decoder = self._make_decoder(input_dim=latent_dim, hidden_dims=hidden_dims)

    def _make_encoder(self, hidden_dims: list[int]) -> nn.Sequential:
        encoder = [nn.Flatten(start_dim=1)]

        start_dim = self.input_dim

        for hidden_dim in hidden_dims:
            encoder.append(
                nn.Linear(in_features=start_dim, out_features=hidden_dim, bias=True)
            )
            encoder.append(nn.BatchNorm1d(num_features=hidden_dim))
            encoder.append(nn.ReLU())
            start_dim = hidden_dim

        return nn.Sequential(*encoder)

    def _make_decoder(self, input_dim: int, hidden_dims: list[int]) -> nn.Sequential:
        decoder = []

        start_dim = input_dim

        for hidden_dim in hidden_dims[::-1]:
            decoder.append(
                nn.Linear(in_features=start_dim, out_features=hidden_dim, bias=True)
            )
            decoder.append(nn.ReLU())
            start_dim = hidden_dim

        decoder.append(nn.Linear(in_features=start_dim, out_features=self.input_dim))

        return nn.Sequential(*decoder)

    def encode(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x)
        mu = self.mu(hidden)
        logvar = self.logvar(hidden)

        return mu, logvar

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)

        return mu + eps * std

    def decode(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        logits_flat = self.decoder(z)
        logits = logits_flat.view(-1, *self.image_shape)

        return logits

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)

        return logits, mu, logvar


class ConditionalVAE(VAE):
    def __init__(
        self,
        hidden_dims: int | list[int] = 256,
        latent_dim: int = 2,
        num_classes: int = 10,
        image_shape: tuple[int, int, int] = (1, 28, 28),
    ) -> None:
        super().__init__(
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            image_shape=image_shape,
        )

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        self.num_classes = num_classes

        self.decoder = self._make_decoder(
            input_dim=latent_dim + num_classes, hidden_dims=hidden_dims
        )

    def decode(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        ohe_labels = nn.functional.one_hot(labels, num_classes=self.num_classes).to(
            dtype=z.dtype
        )
        input_vector = torch.cat((z, ohe_labels), dim=1)
        logits_flat = self.decoder(input_vector)
        logits = logits_flat.view(-1, *self.image_shape)

        return logits

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z, labels)

        return logits, mu, logvar


class GAN(nn.Module):
    def __init__(
        self,
        generator_hidden_dims: int | list[int] = 256,
        discriminator_hidden_dims: int | list[int] = 256,
        latent_dim: int = 2,
        image_shape: tuple[int, int, int] = (1, 28, 28),
        n_freqs: int = 64,
        fourier_scale: float = 2.0,
    ) -> None:
        super().__init__()

        self.image_shape = image_shape
        self.input_dim = math.prod(image_shape)
        self.latent_dim = latent_dim

        self.register_buffer("B", torch.randn(latent_dim, n_freqs) * fourier_scale)
        self.gen_in_dim = latent_dim + 2 * n_freqs

        if isinstance(generator_hidden_dims, int):
            generator_hidden_dims = [generator_hidden_dims]

        if isinstance(discriminator_hidden_dims, int):
            discriminator_hidden_dims = [discriminator_hidden_dims]

        self.generator = self._make_generator(generator_hidden_dims)
        self.discriminator = self._make_discriminator(discriminator_hidden_dims)

    def _make_generator(self, generator_hidden_dims: list[int]) -> nn.Sequential:
        generator = []
        start_dim = self.gen_in_dim

        for hidden_dim in generator_hidden_dims:
            generator.append(
                nn.Linear(in_features=start_dim, out_features=hidden_dim, bias=True)
            )
            generator.append(nn.BatchNorm1d(num_features=hidden_dim))
            generator.append(nn.LeakyReLU(0.2))
            start_dim = hidden_dim

        generator.append(nn.Linear(in_features=start_dim, out_features=self.input_dim))
        generator.append(nn.Sigmoid())

        return nn.Sequential(*generator)

    def _make_discriminator(
        self, discriminator_hidden_dims: list[int]
    ) -> nn.Sequential:
        discriminator = [nn.Flatten()]

        start_dim = self.input_dim

        for hidden_dim in discriminator_hidden_dims:
            discriminator.append(
                nn.Linear(in_features=start_dim, out_features=hidden_dim, bias=True)
            )
            discriminator.append(nn.LeakyReLU(0.2))
            start_dim = hidden_dim

        discriminator.append(nn.Linear(in_features=start_dim, out_features=1))

        return nn.Sequential(*discriminator)

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        p = z @ self.B
        return torch.cat([z, torch.sin(p), torch.cos(p)], dim=-1)

    def generate(self, z: torch.Tensor) -> torch.Tensor:
        hidden = self.generator(self.encode(z))
        return hidden.view(-1, *self.image_shape)

    def discriminate(self, image: torch.Tensor) -> torch.Tensor:
        logits = self.discriminator(image)

        return logits

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.generate(z)


class ConditionalGAN(GAN):
    def __init__(
        self,
        generator_hidden_dims: int | list[int] = 256,
        discriminator_hidden_dims: int | list[int] = 256,
        latent_dim: int = 2,
        image_shape: tuple[int, int, int] = (1, 28, 28),
        n_freqs: int = 64,
        fourier_scale: float = 2.0,
        num_classes: int = 10,
    ) -> None:

        self.num_classes = num_classes

        super().__init__(
            generator_hidden_dims=generator_hidden_dims,
            discriminator_hidden_dims=discriminator_hidden_dims,
            latent_dim=latent_dim,
            image_shape=image_shape,
            n_freqs=n_freqs,
            fourier_scale=fourier_scale,
        )

    def _make_generator(self, generator_hidden_dims: list[int]) -> nn.Sequential:
        generator = []
        start_dim = self.gen_in_dim + self.num_classes

        for hidden_dim in generator_hidden_dims:
            generator.append(
                nn.Linear(in_features=start_dim, out_features=hidden_dim, bias=True)
            )
            generator.append(nn.BatchNorm1d(num_features=hidden_dim))
            generator.append(nn.LeakyReLU(0.2))
            start_dim = hidden_dim

        generator.append(
            nn.Linear(
                in_features=start_dim,
                out_features=self.input_dim,
            )
        )
        generator.append(nn.Sigmoid())

        return nn.Sequential(*generator)

    def _make_discriminator(
        self, discriminator_hidden_dims: list[int]
    ) -> nn.Sequential:
        discriminator = []

        start_dim = self.input_dim + self.num_classes

        for hidden_dim in discriminator_hidden_dims:
            discriminator.append(
                nn.Linear(in_features=start_dim, out_features=hidden_dim, bias=True)
            )
            discriminator.append(nn.LeakyReLU(0.2))
            start_dim = hidden_dim

        discriminator.append(nn.Linear(in_features=start_dim, out_features=1))

        return nn.Sequential(*discriminator)

    def generate(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ohe_labels = nn.functional.one_hot(labels, num_classes=self.num_classes).to(
            dtype=z.dtype
        )
        input_vector = torch.cat((self.encode(z), ohe_labels), dim=1)
        hidden = self.generator(input_vector)
        return hidden.view(-1, *self.image_shape)

    def discriminate(self, image: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        flat_image = image.flatten(start_dim=1)
        ohe_labels = nn.functional.one_hot(labels, num_classes=self.num_classes).to(
            dtype=flat_image.dtype
        )
        input_vector = torch.cat((flat_image, ohe_labels), dim=1)
        logits = self.discriminator(input_vector)

        return logits

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.generate(z, labels)


class DDPM(nn.Module):
    def __init__(
        self,
        image_shape: tuple[int, int, int] = (1, 28, 28),
        hidden_dims: int | list[int] = 32,
        num_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ) -> None:
        super().__init__()

        betas = torch.linspace(beta_start, beta_end, steps=num_steps)
        alpha = 1 - betas
        alpha_bars = torch.cumprod(alpha, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bars", alpha_bars)

        self.image_shape = image_shape
        self.num_steps = num_steps

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        self.network = self._make_network(hidden_dims)

    def _make_network(self, hidden_dims: list[int]) -> nn.ModuleDict:
        def make_block(in_channels: int, out_channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
            )

        encoders = []
        in_channels = self.image_shape[0] + 1

        for out_channels in hidden_dims:
            encoders.append(make_block(in_channels, out_channels))
            in_channels = out_channels

        decoders = []

        for out_channels in reversed(hidden_dims[:-1]):
            decoders.append(
                make_block(
                    in_channels + out_channels,
                    out_channels,
                )
            )
            in_channels = out_channels

        return nn.ModuleDict(
            {
                "encoders": nn.ModuleList(encoders),
                "decoders": nn.ModuleList(decoders),
                "output": nn.Conv2d(
                    in_channels,
                    self.image_shape[0],
                    kernel_size=1,
                ),
            }
        )

    def forward(
        self,
        noisy_images: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        time = timesteps.to(dtype=noisy_images.dtype) / (self.num_steps - 1)
        time = time.view(-1, 1, 1, 1).expand_as(noisy_images[:, :1])

        features = torch.cat((noisy_images, time), dim=1)
        skips = []

        for index, block in enumerate(self.network["encoders"]):
            features = block(features)

            if index < len(self.network["encoders"]) - 1:
                skips.append(features)
                features = nn.functional.max_pool2d(features, kernel_size=2)

        for block, skip in zip(self.network["decoders"], reversed(skips)):
            features = nn.functional.interpolate(
                features,
                size=skip.shape[-2:],
                mode="nearest",
            )

            features = torch.cat((features, skip), dim=1)
            features = block(features)

        return self.network["output"](features)

    def add_noise(
        self, images: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        alpha_bar = self.alpha_bars[timesteps].view(-1, 1, 1, 1)

        return torch.sqrt(alpha_bar) * images + torch.sqrt(1 - alpha_bar) * noise

    @torch.no_grad()
    def generate(self, noise: torch.Tensor) -> torch.Tensor:
        images = noise

        for step in reversed(range(self.num_steps)):
            timesteps = torch.full(
                size=(images.shape[0],),
                fill_value=step,
                dtype=torch.long,
                device=images.device,
            )

            predicted_noise = self(images, timesteps)

            alpha = self.alpha[step]
            alpha_bar = self.alpha_bars[step]
            beta = self.betas[step]

            images = (
                images - beta / torch.sqrt(1 - alpha_bar) * predicted_noise
            ) / torch.sqrt(alpha)

            if step > 0:
                images = images + torch.sqrt(beta) * torch.randn_like(images)

        return ((images + 1) / 2).clamp(0, 1)
