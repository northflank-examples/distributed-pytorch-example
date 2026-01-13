#!/usr/bin/env python3

import os
import time
import argparse
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - [Rank %(rank)s] %(message)s",
)


class RankLogFilter(logging.Filter):
    def filter(self, record):
        record.rank = os.environ.get("RANK", "?")
        return True


logger = logging.getLogger(__name__)
logger.addFilter(RankLogFilter())


class SimpleNet(nn.Module):
    def __init__(
        self, input_size: int = 784, hidden_size: int = 256, num_classes: int = 10
    ):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)


class SyntheticDataset(Dataset):
    def __init__(
        self, num_samples: int = 10000, input_size: int = 784, num_classes: int = 10
    ):
        self.num_samples = num_samples
        self.input_size = input_size
        self.num_classes = num_classes
        self.data = torch.randn(num_samples, input_size)
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def setup_distributed():
    dist.init_process_group(backend="gloo")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    logger.info(
        f"Initialized process group: rank={rank}, world_size={world_size}, "
        f"local_rank={local_rank}"
    )

    return rank, world_size, local_rank


def cleanup_distributed():
    dist.destroy_process_group()


def get_device(local_rank: int) -> torch.device:
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        logger.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU for training")

    return device


def create_data_loader(
    dataset: Dataset, batch_size: int, rank: int, world_size: int
) -> DataLoader:
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    return loader, sampler


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    rank: int,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 10 == 0 and rank == 0:
            logger.info(
                f"Epoch {epoch}, Batch {batch_idx}/{len(loader)}, "
                f"Loss: {loss.item():.4f}"
            )

    avg_loss = total_loss / num_batches
    return avg_loss


def validate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> tuple:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def save_checkpoint(
    model: nn.Module, optimizer: optim.Optimizer, epoch: int, loss: float, path: str
):
    model_state = (
        model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    )

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }

    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved to {path}")


def load_checkpoint(
    model: nn.Module, optimizer: optim.Optimizer, path: str, device: torch.device
) -> int:
    checkpoint = torch.load(path, map_location=device)

    if hasattr(model, "module"):
        model.module.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    logger.info(f"Checkpoint loaded from {path}, epoch {checkpoint['epoch']}")
    return checkpoint["epoch"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()
    device = get_device(local_rank)

    logger.info(f"Starting distributed training with {world_size} processes")
    logger.info(
        f"Configuration: epochs={args.epochs}, batch_size={args.batch_size}, "
        f"lr={args.lr}"
    )

    model = SimpleNet().to(device)
    model = DDP(model)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_dataset = SyntheticDataset(num_samples=args.num_samples)
    val_dataset = SyntheticDataset(num_samples=args.num_samples // 10)

    train_loader, train_sampler = create_data_loader(
        train_dataset, args.batch_size, rank, world_size
    )
    val_loader, _ = create_data_loader(val_dataset, args.batch_size, rank, world_size)

    logger.info(
        f"Dataset size: {len(train_dataset)}, batches per epoch: {len(train_loader)}"
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    if rank == 0:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.resume and os.path.exists(args.resume):
        start_epoch = load_checkpoint(model, optimizer, args.resume, device)

    dist.barrier()

    best_accuracy = 0.0
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        train_sampler.set_epoch(epoch)

        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, rank
        )

        val_loss, val_accuracy = validate(model, val_loader, criterion, device)

        metrics = torch.tensor([train_loss, val_loss, val_accuracy], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        metrics /= world_size

        avg_train_loss = metrics[0].item()
        avg_val_loss = metrics[1].item()
        avg_val_accuracy = metrics[2].item()

        epoch_time = time.time() - epoch_start

        if rank == 0:
            logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
            logger.info(f"  Train Loss: {avg_train_loss:.4f}")
            logger.info(
                f"  Val Loss: {avg_val_loss:.4f}, Val Accuracy: {avg_val_accuracy:.2f}%"
            )

            if avg_val_accuracy > best_accuracy:
                best_accuracy = avg_val_accuracy
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    avg_train_loss,
                    os.path.join(args.checkpoint_dir, "best_model.pt"),
                )

            save_checkpoint(
                model,
                optimizer,
                epoch,
                avg_train_loss,
                os.path.join(args.checkpoint_dir, "latest_model.pt"),
            )

        dist.barrier()

    total_time = time.time() - start_time

    if rank == 0:
        logger.info(f"Training completed in {total_time:.2f}s")
        logger.info(f"Best validation accuracy: {best_accuracy:.2f}%")

    cleanup_distributed()


if __name__ == "__main__":
    main()
