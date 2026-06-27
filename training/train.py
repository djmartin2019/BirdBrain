from pathlib import Path
import time
import copy

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from dataloaders import get_dataloaders

# --- Configuration ---
DATA_DIR = "data/raw/CUB_200_2011"
CHECKPOINT_PATH = "models/birdbrain_v1-3.pt"
MODEL_OUTPUT_PATH = "models/birdbrain_v1-4.pt"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH}"
MLFLOW_EXPERIMENT = "birdbrain-cub200"
RUN_NAME = "phase-5-bbox-crop"
PHASE = "5"

NUM_CLASSES = 200
BATCH_SIZE = 32
NUM_EPOCHS = 20
NUM_UNFROZEN_BLOCKS = 5
USE_BBOX_CROP = True

BACKBONE_LR = 5e-5
HEAD_LR = 1e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
LR_SCHEDULER_PATIENCE = 2
EARLY_STOPPING_PATIENCE = 5


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

    Freeze most of the backbone, then unfreeze the last few blocks so
    mid/high-level features can adapt to bird species.
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


def get_optimizer_param_groups(model, backbone_lr, head_lr, num_unfrozen_blocks):
    """Split trainable params so the backbone uses a lower learning rate than the head."""
    backbone_params = []
    for block in model.features[-num_unfrozen_blocks:]:
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
        return None

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    prior_val_acc = checkpoint.get("best_val_acc")
    if prior_val_acc is not None:
        print(f"Loaded {checkpoint_path} (prior best val acc: {prior_val_acc:.4f})")
    else:
        print(f"Loaded {checkpoint_path}")

    return prior_val_acc


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


def save_checkpoint(state_dict, val_acc, output_path):
    torch.save(
        {
            "model_name": "efficientnet_b0",
            "num_classes": NUM_CLASSES,
            "state_dict": state_dict,
            "best_val_acc": val_acc,
            "image_size": 224,
            "use_bbox_crop": USE_BBOX_CROP,
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
        output_path,
    )


def main():
    Path("models").mkdir(exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=RUN_NAME):
        mlflow.set_tags({"phase": PHASE, "model": "efficientnet_b0"})
        mlflow.log_params({
            "data_dir": DATA_DIR,
            "checkpoint_in": CHECKPOINT_PATH,
            "model_output": MODEL_OUTPUT_PATH,
            "num_classes": NUM_CLASSES,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "num_unfrozen_blocks": NUM_UNFROZEN_BLOCKS,
            "use_bbox_crop": USE_BBOX_CROP,
            "backbone_lr": BACKBONE_LR,
            "head_lr": HEAD_LR,
            "weight_decay": WEIGHT_DECAY,
            "label_smoothing": LABEL_SMOOTHING,
            "lr_scheduler_patience": LR_SCHEDULER_PATIENCE,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "device": str(device),
        })

        train_loader, val_loader = get_dataloaders(
            data_dir=DATA_DIR,
            batch_size=BATCH_SIZE,
            num_workers=4,
            use_bbox_crop=USE_BBOX_CROP,
        )

        model = build_model(NUM_CLASSES)
        model = model.to(device)

        prior_val_acc = load_checkpoint(model, CHECKPOINT_PATH, device)
        if prior_val_acc is not None:
            mlflow.log_metric("prior_val_acc", prior_val_acc)

        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

        optimizer = optim.Adam(
            get_optimizer_param_groups(
                model, BACKBONE_LR, HEAD_LR, NUM_UNFROZEN_BLOCKS
            ),
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
        epochs_without_improvement = 0

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
            train_val_gap = train_acc - val_acc

            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
            print(f"Gap: {train_val_gap:.4f} | LRs: backbone={backbone_lr:.2e}, head={head_lr:.2e}")

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "train_val_gap": train_val_gap,
                    "backbone_lr": backbone_lr,
                    "head_lr": head_lr,
                },
                step=epoch,
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_weights = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0

                save_checkpoint(best_model_weights, best_val_acc, MODEL_OUTPUT_PATH)

                mlflow.log_metric("best_val_acc", best_val_acc, step=epoch)
                mlflow.log_artifact(MODEL_OUTPUT_PATH, artifact_path="checkpoints")

                print(f"Saved new best model: {MODEL_OUTPUT_PATH}")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                    print(
                        f"Early stopping: no val improvement for "
                        f"{EARLY_STOPPING_PATIENCE} epochs"
                    )
                    break

        model.load_state_dict(best_model_weights)

        elapsed = time.time() - start_time
        mlflow.log_metric("training_minutes", elapsed / 60)
        mlflow.log_metric("final_best_val_acc", best_val_acc)

        print(f"\nTraining complete in {elapsed / 60:.1f} minutes")
        print(f"Best Val Acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
