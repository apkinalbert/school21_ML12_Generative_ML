from dataclasses import dataclass


@dataclass
class TrainConfig:
    epochs: int = 200
    patience: int | None = 10
    lr: float = 1e-3
    weight_decay: float = 1e-6
