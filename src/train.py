import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections.abc import Callable
import copy
import pandas as pd
import time
from dataset import GPUDataLoader

Batch = tuple[torch.Tensor, torch.Tensor]

Optimizers = torch.optim.Optimizer | dict[str, torch.optim.Optimizer]

TrainStepFn = Callable[
    [nn.Module, Batch, Optimizers],
    dict[str, torch.Tensor],
]

ValidationStepFn = Callable[[nn.Module, Batch], dict[str, torch.Tensor]]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader | GPUDataLoader,
    optimizer: Optimizers,
    step_fn: TrainStepFn,
    device: torch.device,
) -> dict[str, float]:

    model.train()

    metrics_accumulator = {}

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        step_result = step_fn(model, (images, labels), optimizer)

        for key, tensor_val in step_result.items():
            metrics_accumulator[key] = metrics_accumulator.get(key, 0) + tensor_val

    epoch_metrics = {
        key: (total_tensor / len(loader)).item()
        for key, total_tensor in metrics_accumulator.items()
    }

    return epoch_metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader | GPUDataLoader,
    step_fn: ValidationStepFn,
    device: torch.device,
) -> dict[str, float]:

    model.eval()

    metrics_accumulator = {}

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        step_result = step_fn(model, (images, labels))

        for key, tensor_val in step_result.items():
            metrics_accumulator[key] = metrics_accumulator.get(key, 0) + tensor_val

    epoch_metrics = {
        key: (total_tensor / len(loader)).item()
        for key, total_tensor in metrics_accumulator.items()
    }

    return epoch_metrics


def fit(
    model: nn.Module,
    train_loader: DataLoader | GPUDataLoader,
    valid_loader: DataLoader | GPUDataLoader,
    optimizer: Optimizers,
    train_step_fn: TrainStepFn,
    validation_step_fn: ValidationStepFn,
    device: torch.device,
    epochs: int,
    patience: int | None = 10,
    verbose: int = 2,
    monitor: str | None = "loss",
    epoch_callback: Callable[[nn.Module, int], None] | None = None,
) -> tuple[pd.DataFrame, int, float | None]:

    if epochs < 1:
        raise ValueError("epochs must be >= 1")

    if patience is not None and patience < 1:
        raise ValueError("patience must be >= 1 or None")

    history = []

    best_epoch = 0
    best_loss = float("inf")
    epoch_without_improvement = 0
    best_state = None

    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        train_metrics = train_one_epoch(
            model=model,
            device=device,
            optimizer=optimizer,
            step_fn=train_step_fn,
            loader=train_loader,
        )

        valid_metrics = validate(
            model=model,
            device=device,
            step_fn=validation_step_fn,
            loader=valid_loader,
        )

        epoch_seconds = time.perf_counter() - epoch_start

        epoch_record = {
            "epoch": epoch + 1,
            "epoch_seconds": epoch_seconds,
        }

        epoch_record.update(
            {f"train_{key}": value for key, value in train_metrics.items()}
        )

        epoch_record.update(
            {f"valid_{key}": value for key, value in valid_metrics.items()}
        )

        history.append(epoch_record)

        if monitor is not None:
            loss = valid_metrics[monitor]

            if loss < best_loss:
                best_loss = loss
                best_epoch = epoch + 1
                epoch_without_improvement = 0

                best_state = copy.deepcopy(model.state_dict())

            else:
                epoch_without_improvement += 1

        if verbose == 2:
            metrics_text = " | ".join(
                f"train {key}: {value:.4f} | valid {key}: {valid_metrics[key]:.4f}"
                for key, value in train_metrics.items()
                if key in valid_metrics
            )

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"{metrics_text} | "
                f"time: {epoch_seconds:.2f} s"
            )

        if epoch_callback is not None:
            epoch_callback(model, epoch + 1)

        if monitor is not None:
            if patience is not None and epoch_without_improvement >= patience:
                if verbose == 2:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

    if monitor is None:
        return pd.DataFrame(history), epoch + 1, None

    if best_state is None:
        raise RuntimeError("No valid checkpoint was saved")

    model.load_state_dict(best_state)

    return pd.DataFrame(history), best_epoch, best_loss
