from pathlib import Path
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from dataloaders import get_dataloaders

# --- Configuration ---
DATA_DIR = "data/raw/CUB_200_2011"
CHECKPOINT_PATH = "models/birdbrain_v1-1.pt"
MODEL_OUTPUT_PATH = "models/birdbrain_v1-2.pt"

NUM_CLASSES = 200
BATCH_SIZE = 32
NUM_EPOCHS = 15
NUM_UNFROZEN_BLOCKS = 3

BACKBONE_LR = 1e-4
HEAD_LR = 5e-4
WEIGHT_DECAY = 1e-4
LR_SCHEDULER_PATIENCE = 2


def get_device():
    """Pick the best available accelerator (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(num_classes: int, num_unfrozen_blocks: int = NUM_UNFROZEN_BLOCKS):
    """
    Build EfficientNet-B0 with a 200-class head.

    Phase 3: freeze most of the backbone, then unfreeze the last few blocks
    so mid/high-level features can adapt to bird species.
    """
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)

    for param in model.features.parameters():
        param.requires_grad = False

    for block in model.features[-num_unfrozen_blocks:]:
        for param in block.parameters():
            param.requires_grad = True

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


def get_optimizer_param_groups(model, backbone_lr, head_lr):
    """Split trainable params so the backbone uses a lower learning rate than the head."""
    backbone_params = []
    for block in model.features[-NUM_UNFROZEN_BLOCKS:]:
        backbone_params.extend(p for p in block.parameters() if p.requires_grad)

    head_params = [p for p in model.classifier.parameters() if p.requires_grad]

    return [
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params, "lr": head_lr},
    ]


def load_checkpoint(model, checkpoint_path, device):
    """Load weights from a prior training run, if the checkpoint file exists."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"No checkpoint at {checkpoint_path}; training from ImageNet init.")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    prior_val_acc = checkpoint.get("best_val_acc")
    if prior_val_acc is not None:
        print(f"Loaded {checkpoint_path} (prior best val acc: {prior_val_acc:.4f})")
    else:
        print(f"Loaded {checkpoint_path}")


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Run one pass over the training set; return average loss and accuracy."""
    model.train()

    running_loss = 0.0
    running_corrects = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)

        running_loss += loss.item() * images.size(0)
        running_corrects += torch.sum(preds == labels).item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total

    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    """Run inference on the validation set without updating weights."""
    model.eval()

    running_loss = 0.0
    running_corrects = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels).item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total

    return epoch_loss, epoch_acc


def main():
    Path("models").mkdir(exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader = get_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=4,
    )

    model = build_model(NUM_CLASSES)
    model = model.to(device)

    # Continue from phase-2 weights (last block + classifier).
    load_checkpoint(model, CHECKPOINT_PATH, device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        get_optimizer_param_groups(model, BACKBONE_LR, HEAD_LR),
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=LR_SCHEDULER_PATIENCE,
    )

    best_model_weights = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0

    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 30)

        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_acc = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step(val_acc)

        backbone_lr = optimizer.param_groups[0]["lr"]
        head_lr = optimizer.param_groups[1]["lr"]
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
        print(f"LRs: backbone={backbone_lr:.2e}, head={head_lr:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_weights = copy.deepcopy(model.state_dict())

            torch.save(
                {
                    "model_name": "efficientnet_b0",
                    "num_classes": NUM_CLASSES,
                    "state_dict": best_model_weights,
                    "best_val_acc": best_val_acc,
                    "image_size": 224,
                    "normalization": {
                        "mean": [0.485, 0.456, 0.406],
                        "std": [0.229, 0.224, 0.225],
                    },
                },
                MODEL_OUTPUT_PATH,
            )

            print(f"Saved new best model: {MODEL_OUTPUT_PATH}")

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed / 60:.1f} minutes")
    print(f"Best Val Acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
