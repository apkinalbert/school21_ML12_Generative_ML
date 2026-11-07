from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs import TrainConfig
from train import Optimizers, TrainStepFn, ValidationStepFn, fit, validate
from dataset import GPUDataLoader


def make_optimizer(
    model: nn.Module,
    config: TrainConfig,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        fused=next(model.parameters()).is_cuda,
    )


@dataclass
class Experiment:
    name: str
    model_class: type[nn.Module]
    device: torch.device

    train_loader: DataLoader | GPUDataLoader
    valid_loader: DataLoader | GPUDataLoader

    train_step_fn: TrainStepFn
    validation_step_fn: ValidationStepFn

    model_params: dict[str, Any]
    test_loader: DataLoader | GPUDataLoader | None = None

    optimizer_factory: Callable[[nn.Module, TrainConfig], Optimizers] = make_optimizer

    monitor: str | None = "loss"
    epoch_callback: Callable[[nn.Module, int], None] | None = None


def run(
    experiment: Experiment,
    config: TrainConfig,
    verbose: int = 2,
) -> dict[str, Any]:

    if verbose >= 1:
        print(f"\nExperiment: {experiment.name}")

    training_start = time.perf_counter()

    model = experiment.model_class(**experiment.model_params).to(experiment.device)

    optimizer = experiment.optimizer_factory(model, config)

    history, best_epoch, best_loss = fit(
        model=model,
        train_loader=experiment.train_loader,
        valid_loader=experiment.valid_loader,
        optimizer=optimizer,
        train_step_fn=experiment.train_step_fn,
        validation_step_fn=experiment.validation_step_fn,
        device=experiment.device,
        epochs=config.epochs,
        patience=config.patience,
        verbose=verbose,
        monitor=experiment.monitor,
        epoch_callback=experiment.epoch_callback,
    )

    training_seconds = time.perf_counter() - training_start

    if verbose >= 1:
        print(
            f"Completed epochs: {len(history)} | "
            f"Total training time: {training_seconds:.2f} s"
        )
        if best_loss is not None:
            print(
                f"Best epoch: {best_epoch} | "
                f"Best validation {experiment.monitor}: {best_loss:.4f}"
            )

    result = {
        "name": experiment.name,
        "model": model,
        "model_params": experiment.model_params.copy(),
        "train_config": asdict(config),
        "history": history,
        "best_epoch": best_epoch if experiment.monitor is not None else None,
        "best_valid_loss": best_loss,
        "training_seconds": training_seconds,
    }

    if experiment.test_loader is not None:
        metrics = validate(
            model=model,
            loader=experiment.test_loader,
            step_fn=experiment.validation_step_fn,
            device=experiment.device,
        )
        result.update({f"test_{key}": value for key, value in metrics.items()})

        if verbose >= 1:
            print("\nTest metrics:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.4f}")

    return result
