#!/usr/bin/env python3
"""
Train a simple binary classifier to distinguish CPPN-generated images from ImageNet photos.

Example:
python train_cppn_imagenet_classifier.py \
  --cppn-dir initial_populations \
  --imagenet-dir /path/to/imagenet_root \
  --imagenet-split train \
  --epochs 3 --batch-size 32
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable, Sequence

import torch
from PIL import Image
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a binary classifier separating CPPN images from ImageNet photos."
    )
    parser.add_argument(
        "--cppn-dir",
        type=Path,
        default=Path("initial_populations"),
        help="Directory containing CPPN-generated images (recursively searched).",
    )
    parser.add_argument(
        "--imagenet-dir",
        type=Path,
        required=True,
        help="Root directory of ImageNet (expects subfolders like train/val or point directly to one split).",
    )
    parser.add_argument(
        "--imagenet-split",
        type=str,
        default="train",
        help="Subfolder of ImageNet to use (e.g., train or val). If not found, the root is used as-is.",
    )
    parser.add_argument(
        "--image-size", type=int, default=224, help="Square image size after preprocessing."
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW optimizer."
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction of each class reserved for validation.",
    )
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=5000,
        help="Cap samples per class to keep runs lightweight. Use -1 for all available.",
    )
    parser.add_argument(
        "--balance-classes",
        action="store_true",
        help="Downsample both classes to the same number of samples.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Dataloader worker processes per device."
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--save-path",
        type=Path,
        default=Path("cppn_vs_imagenet_classifier.pt"),
        help="Where to store the trained model checkpoint.",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use torchvision's pretrained ResNet weights (may trigger a download).",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default=None,
        help="Device override. Defaults to CUDA if available, then MPS, else CPU.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=25,
        help="How often to log batch loss during training (set 0 to disable).",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_image_paths(root: Path, allowed_exts: set[str]) -> list[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")
    paths = [
        path for path in root.rglob("*") if path.suffix.lower() in allowed_exts and path.is_file()
    ]
    if not paths:
        raise ValueError(f"No images with extensions {sorted(allowed_exts)} found under {root}")
    return sorted(paths)


def choose_indices(num_items: int, limit: int | None, seed: int) -> list[int]:
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(num_items, generator=generator).tolist()
    if limit is None:
        return order
    return order[: min(limit, num_items)]


def split_indices(indices: Sequence[int], val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    if not indices:
        raise ValueError("Cannot split an empty index list.")
    if len(indices) < 2:
        raise ValueError("Need at least 2 samples to make a train/validation split.")
    val_size = max(1, int(len(indices) * val_ratio))
    val_size = min(val_size, len(indices) - 1)
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(indices), generator=generator).tolist()
    shuffled = [indices[i] for i in order]
    split_point = len(indices) - val_size
    return shuffled[:split_point], shuffled[split_point:]


class ImagePathDataset(Dataset):
    """Dataset that loads images from explicit paths and assigns a fixed label."""

    def __init__(self, paths: Iterable[Path], transform: transforms.Compose, label: int) -> None:
        self.paths = list(paths)
        self.transform = transform
        self.label = label

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        return self.transform(img), self.label


class LabelOverrideDataset(Dataset):
    """Wraps any dataset but overrides the label with a fixed value."""

    def __init__(self, base_dataset: Dataset, label: int) -> None:
        self.base_dataset = base_dataset
        self.label = label

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        image, _ = self.base_dataset[idx]
        return image, self.label


def compute_limits(
    cppn_count: int, imagenet_count: int, max_samples: int, balance: bool
) -> tuple[int, int]:
    limit = None if max_samples < 0 else max_samples
    capped_cppn = cppn_count if limit is None else min(cppn_count, limit)
    capped_imagenet = imagenet_count if limit is None else min(imagenet_count, limit)
    if balance:
        shared = min(capped_cppn, capped_imagenet)
        capped_cppn = shared
        capped_imagenet = shared
    if capped_cppn < 2 or capped_imagenet < 2:
        raise ValueError(
            f"Not enough samples after capping/balancing "
            f"(cppn={capped_cppn}, imagenet={capped_imagenet})."
        )
    return capped_cppn, capped_imagenet


def resolve_imagenet_root(imagenet_dir: Path, split: str) -> Path:
    candidate = imagenet_dir / split
    return candidate if candidate.exists() else imagenet_dir


def make_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, eval_transform


def build_datasets(args: argparse.Namespace) -> tuple[ConcatDataset, ConcatDataset]:
    train_transform, eval_transform = make_transforms(args.image_size)

    cppn_paths = collect_image_paths(args.cppn_dir, IMAGE_EXTENSIONS)
    imagenet_root = resolve_imagenet_root(args.imagenet_dir, args.imagenet_split)
    imagenet_train_base = datasets.ImageFolder(imagenet_root, transform=train_transform)
    imagenet_eval_base = datasets.ImageFolder(imagenet_root, transform=eval_transform)

    cppn_limit, imagenet_limit = compute_limits(
        len(cppn_paths), len(imagenet_train_base), args.max_samples_per_class, args.balance_classes
    )

    cppn_indices = choose_indices(len(cppn_paths), cppn_limit, args.seed)
    imagenet_indices = choose_indices(len(imagenet_train_base), imagenet_limit, args.seed + 1)

    cppn_train_idx, cppn_val_idx = split_indices(cppn_indices, args.val_split, args.seed)
    imagenet_train_idx, imagenet_val_idx = split_indices(
        imagenet_indices, args.val_split, args.seed + 1
    )

    cppn_train = Subset(ImagePathDataset(cppn_paths, train_transform, label=0), cppn_train_idx)
    cppn_val = Subset(ImagePathDataset(cppn_paths, eval_transform, label=0), cppn_val_idx)

    imagenet_train = Subset(
        LabelOverrideDataset(imagenet_train_base, label=1), imagenet_train_idx
    )
    imagenet_val = Subset(LabelOverrideDataset(imagenet_eval_base, label=1), imagenet_val_idx)

    print(
        f"Using {len(cppn_train)} CPPN images for training and {len(cppn_val)} for validation "
        f"from {args.cppn_dir}"
    )
    print(
        f"Using {len(imagenet_train)} ImageNet images for training and {len(imagenet_val)} for "
        f"validation from {imagenet_root}"
    )

    train_dataset = ConcatDataset([cppn_train, imagenet_train])
    val_dataset = ConcatDataset([cppn_val, imagenet_val])
    return train_dataset, val_dataset


def make_model(device: torch.device, use_pretrained: bool) -> nn.Module:
    weights = None
    if use_pretrained:
        try:
            weights = models.ResNet18_Weights.DEFAULT
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Falling back to random initialization (pretrained weights unavailable): {exc}")
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model.to(device)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    log_interval: int,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for step, (images, labels) in enumerate(dataloader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

        if log_interval and step % log_interval == 0:
            print(f"  [train] step {step}/{len(dataloader)} - loss {loss.item():.4f}")

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"Using device: {device}")

    train_dataset, val_dataset = build_datasets(args)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = make_model(device, args.pretrained)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = 0.0
    args.save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, args.log_interval
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(
            f"  Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": ["cppn", "imagenet"],
                    "image_size": args.image_size,
                    "config": vars(args),
                },
                args.save_path,
            )
            print(f"  Saved new best model to {args.save_path} (val acc={val_acc:.4f})")

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.4f}")
    print(f"Checkpoint path: {args.save_path.resolve()}")


if __name__ == "__main__":
    main()
