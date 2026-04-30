import os

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from config import BATCH_SIZE, DATA_ROOT, NUM_WORKERS, RANDOM_SEED, VALID_RATIO


TRANSFORM = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)


def get_FashionMNIST_dataloaders():
    full_train_dataset = datasets.FashionMNIST(
        root=os.path.join(DATA_ROOT, "FashionMNIST"),
        train=True,
        download=True,
        transform=TRANSFORM,
    )
    test_dataset = datasets.FashionMNIST(
        root=os.path.join(DATA_ROOT, "FashionMNIST"),
        train=False,
        download=True,
        transform=TRANSFORM,
    )

    val_size = int(len(full_train_dataset) * VALID_RATIO)
    val_size = min(max(val_size, 1), len(full_train_dataset) - 1)
    train_size = len(full_train_dataset) - val_size
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    return train_loader, val_loader, test_loader