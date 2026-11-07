import matplotlib.pyplot as plt
import pandas as pd
import torch
from collections.abc import Callable
import torch.nn as nn
from IPython.display import display


def plot_history(
    history: pd.DataFrame,
    best_epoch: int | None,
    name: str = "",
) -> None:

    metrics = []

    for column in history.columns:
        if column.startswith("train_"):
            metric = column.removeprefix("train_")

            if f"valid_{metric}" in history.columns:
                metrics.append(metric)

    if not metrics:
        raise ValueError("No metrics to plot")

    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(5 * len(metrics), 4),
        squeeze=False,
    )

    epochs = history["epoch"]

    for ax, metric in zip(axes.flat, metrics):
        ax.plot(epochs, history[f"train_{metric}"], label="Train")
        ax.plot(epochs, history[f"valid_{metric}"], label="Validation")

        if best_epoch is not None:
            ax.axvline(
                best_epoch,
                color="gray",
                linestyle="--",
                label=f"Best epoch: {best_epoch}",
            )

        ax.set_title(metric.replace("_", " ").capitalize())
        ax.legend()

        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)

    if name:
        fig.suptitle(name)

    fig.tight_layout()
    plt.show()


def plot_image_grid(
    images: torch.Tensor,
    nrows: int,
    ncols: int,
) -> None:
    images = images.detach().cpu()

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols, nrows),
        squeeze=False,
    )

    for ax, image in zip(axes.flat, images):
        if image.shape[0] == 1:
            ax.imshow(
                image.squeeze(0),
                cmap="gray_r",
                vmin=0,
                vmax=1,
            )
        else:
            ax.imshow(image.permute(1, 2, 0))

        ax.axis("off")

    plt.show()


def make_image_callback(
    generate_fn: Callable[[nn.Module], torch.Tensor],
    every: int = 10,
) -> Callable[[nn.Module, int], None]:

    if every < 1:
        raise ValueError("every must be >= 1")

    snapshots = []
    output = None

    @torch.no_grad()
    def callback(model: nn.Module, epoch: int) -> None:
        nonlocal output

        if epoch != 1 and epoch % every != 0:
            return

        was_training = model.training
        model.eval()

        try:
            images = generate_fn(model).detach().float().cpu()
        finally:
            model.train(was_training)

        snapshots.append((epoch, images))

        fig, axes = plt.subplots(
            len(snapshots),
            len(images),
            figsize=(1.3 * len(images), 1.1 * len(snapshots)),
            squeeze=False,
        )

        for row, (saved_epoch, saved_images) in zip(axes, snapshots):
            for ax, image in zip(row, saved_images):
                if image.shape[0] == 1:
                    ax.imshow(image.squeeze(0), cmap="gray_r", vmin=0, vmax=1)
                else:
                    ax.imshow(image.permute(1, 2, 0).clamp(0, 1))

                ax.set_xticks([])
                ax.set_yticks([])

            row[0].set_ylabel(
                f"Epoch {saved_epoch}", rotation=0, labelpad=40, va="center"
            )

        for index, ax in enumerate(axes[0]):
            ax.set_title(f"Sample {index + 1}")

        fig.tight_layout(pad=0.3)

        if output is None:
            output = display(fig, display_id=True)
        else:
            output.update(fig)

        plt.close(fig)

    return callback
