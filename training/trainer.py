import copy
import time
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim

from config import IMAGENET_NORMALIZE, TrainConfig
from dataloaders import get_dataloaders
from models import build_model, get_param_groups


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(model, checkpoint_path: Path | None, device):
    if checkpoint_path is None:
        return None

    if not checkpoint_path.exists():
        print(f"No checkpoint at {checkpoint_path}; training from pretrained init.")
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

    return running_loss / total, running_corrects / total


def evaluate(model, dataloader, criterion, device):
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

    return running_loss / total, running_corrects / total


def build_optimizer(cfg: TrainConfig, model):
    param_groups = get_param_groups(model, cfg)
    opt_name = cfg.training.optimizer

    if opt_name == "adam":
        return optim.Adam(param_groups, weight_decay=cfg.training.weight_decay)
    if opt_name == "adamw":
        return optim.AdamW(param_groups, weight_decay=cfg.training.weight_decay)
    if opt_name == "sgd":
        return optim.SGD(
            param_groups,
            momentum=0.9,
            weight_decay=cfg.training.weight_decay,
        )

    raise ValueError(f"Unsupported optimizer: {opt_name}")


def save_checkpoint(state_dict, cfg: TrainConfig, val_acc, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": cfg.model.name,
            "num_classes": cfg.model.num_classes,
            "state_dict": state_dict,
            "best_val_acc": val_acc,
            "image_size": cfg.data.image_size,
            "use_bbox_crop": cfg.data.use_bbox_crop,
            "config_path": str(cfg.config_path),
            "normalization": IMAGENET_NORMALIZE,
        },
        output_path,
    )


def _log_params(cfg: TrainConfig, device):
    mlflow.log_params({
        "experiment_name": cfg.experiment_name,
        "config_path": str(cfg.config_path),
        "model_name": cfg.model.name,
        "checkpoint_in": str(cfg.model.checkpoint_path or ""),
        "model_output": str(cfg.model.output_path),
        "num_classes": cfg.model.num_classes,
        "batch_size": cfg.training.batch_size,
        "num_epochs": cfg.training.epochs,
        "optimizer": cfg.training.optimizer,
        "learning_rate": cfg.training.learning_rate,
        "backbone_learning_rate": cfg.training.backbone_learning_rate or "",
        "weight_decay": cfg.training.weight_decay,
        "label_smoothing": cfg.training.label_smoothing,
        "unfreeze_blocks": cfg.training.unfreeze_blocks,
        "use_bbox_crop": cfg.data.use_bbox_crop,
        "augmentation": cfg.data.augmentation,
        "lr_scheduler_patience": cfg.training.scheduler.patience,
        "early_stopping_patience": cfg.training.early_stopping_patience or "",
        "device": str(device),
    })


def train(cfg: TrainConfig):
    cfg.model.output_path.parent.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment)

    with mlflow.start_run(run_name=cfg.run_name):
        tags = {"model": cfg.model.name}
        if cfg.phase:
            tags["phase"] = cfg.phase
        mlflow.set_tags(tags)
        _log_params(cfg, device)

        train_loader, val_loader = get_dataloaders(cfg)

        model = build_model(cfg).to(device)
        prior_val_acc = load_checkpoint(model, cfg.model.checkpoint_path, device)
        if prior_val_acc is not None:
            mlflow.log_metric("prior_val_acc", prior_val_acc)

        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.training.label_smoothing)
        optimizer = build_optimizer(cfg, model)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=cfg.training.scheduler.factor,
            patience=cfg.training.scheduler.patience,
        )

        best_model_weights = copy.deepcopy(model.state_dict())
        best_val_acc = 0.0
        epochs_without_improvement = 0
        start_time = time.time()

        for epoch in range(cfg.training.epochs):
            print(f"\nEpoch {epoch + 1}/{cfg.training.epochs}")
            print("-" * 30)

            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            scheduler.step(val_acc)

            lr_values = [group["lr"] for group in optimizer.param_groups]
            train_val_gap = train_acc - val_acc

            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
            print(f"Gap: {train_val_gap:.4f} | LRs: {[f'{lr:.2e}' for lr in lr_values]}")

            metrics = {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "train_val_gap": train_val_gap,
            }
            for i, lr in enumerate(lr_values):
                metrics[f"lr_group_{i}"] = lr
            mlflow.log_metrics(metrics, step=epoch)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_weights = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0

                save_checkpoint(
                    best_model_weights,
                    cfg,
                    best_val_acc,
                    cfg.model.output_path,
                )
                mlflow.log_metric("best_val_acc", best_val_acc, step=epoch)
                mlflow.log_artifact(
                    str(cfg.model.output_path),
                    artifact_path="checkpoints",
                )
                print(f"Saved new best model: {cfg.model.output_path}")
            else:
                epochs_without_improvement += 1
                patience = cfg.training.early_stopping_patience
                if patience is not None and epochs_without_improvement >= patience:
                    print(f"Early stopping: no val improvement for {patience} epochs")
                    break

        model.load_state_dict(best_model_weights)

        elapsed = time.time() - start_time
        mlflow.log_metric("training_minutes", elapsed / 60)
        mlflow.log_metric("final_best_val_acc", best_val_acc)

        print(f"\nTraining complete in {elapsed / 60:.1f} minutes")
        print(f"Best Val Acc: {best_val_acc:.4f}")

        return best_val_acc
