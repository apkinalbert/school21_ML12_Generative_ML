from torch.utils.data import Dataset, DataLoader
import torch


class TransferDatasetToDevice(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        device: torch.device,
    ) -> None:
        super().__init__()

        full_loader = DataLoader(
            dataset=dataset, batch_size=len(dataset), shuffle=False
        )
        images, labels = next(iter(full_loader))
        self.images = images.to(device=device, dtype=torch.float32)
        self.labels = labels.to(device=device, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[index]
        label = self.labels[index]

        return image, label


class GPUDataLoader:
    def __init__(self, dataset, batch_size=250, shuffle=False):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        if dataset.images.device.type != "cuda":
            raise ValueError("dataset must be on GPU")

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        n = len(self.dataset)
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        images = self.dataset.images
        labels = self.dataset.labels
        n = len(self.dataset)

        if self.shuffle:
            indices = torch.randperm(n, device=images.device)

            for start in range(0, n, self.batch_size):
                batch_indices = indices[start : start + self.batch_size]

                yield (
                    images.index_select(0, batch_indices),
                    labels.index_select(0, batch_indices),
                )
        else:
            for start in range(0, n, self.batch_size):
                yield (
                    images[start : start + self.batch_size],
                    labels[start : start + self.batch_size],
                )
