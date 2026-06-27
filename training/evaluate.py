"""
Evaluate a trained checkpoint on the held-out val or official test split.

Usage (from training/):
    python evaluate.py --config ../configs/efficientnet_stage5_bbox.yaml \\
        --checkpoint ../models/birdbrain_v1-4.pt

    python evaluate.py --config ../configs/efficientnet_stage5_bbox.yaml \\
        --checkpoint ../models/birdbrain_v1-4.pt --split test --mlflow
"""

import argparse
import json
from pathlib import Path

import mlflow
import torch
import torch.nn as nn

from config import load_config, resolve_cli_path
from dataloaders import get_eval_dataloader
from models import build_model_for_inference
from trainer import get_device


def _resolve_checkpoint_path(path: Path) -> Path:
    return resolve_cli_path(path)


def run_evaluation(model, dataloader, device, top_k: int = 5):
    """Compute loss, top-1 accuracy, and top-k accuracy."""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    top1_correct = 0
    topk_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            total += labels.size(0)

            preds = outputs.argmax(dim=1)
            top1_correct += (preds == labels).sum().item()

            k = min(top_k, outputs.size(1))
            topk_preds = outputs.topk(k, dim=1).indices
            topk_correct += topk_preds.eq(labels.unsqueeze(1)).any(dim=1).sum().item()

    return {
        "loss": running_loss / total,
        "top1_acc": top1_correct / total,
        "top5_acc": topk_correct / total,
        "num_samples": total,
    }


def save_report(report: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved report: {output_path}")


def evaluate_checkpoint(
    cfg,
    checkpoint_path: Path,
    split: str = "test",
    output_path: Path | None = None,
    log_mlflow: bool = False,
):
    checkpoint_path = _resolve_checkpoint_path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = get_device()
    print(f"Using device: {device}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint.get("model_name", cfg.model.name)
    num_classes = checkpoint.get("num_classes", cfg.model.num_classes)
    use_bbox_crop = checkpoint.get("use_bbox_crop", cfg.data.use_bbox_crop)
    checkpoint_val_acc = checkpoint.get("best_val_acc")

    model = build_model_for_inference(model_name, num_classes).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    print(f"Loaded {checkpoint_path}")

    loader = get_eval_dataloader(cfg, split=split, use_bbox_crop=use_bbox_crop)
    print(f"Evaluating on split={split!r}, samples={len(loader.dataset)}, bbox_crop={use_bbox_crop}")

    metrics = run_evaluation(model, loader, device)

    report = {
        "checkpoint": str(checkpoint_path),
        "config_path": str(cfg.config_path),
        "split": split,
        "model_name": model_name,
        "num_classes": num_classes,
        "use_bbox_crop": use_bbox_crop,
        "checkpoint_best_val_acc": checkpoint_val_acc,
        **metrics,
    }

    print(f"\nResults ({split}):")
    print(f"  Loss:     {metrics['loss']:.4f}")
    print(f"  Top-1:    {metrics['top1_acc']:.4f}")
    print(f"  Top-5:    {metrics['top5_acc']:.4f}")
    if checkpoint_val_acc is not None:
        print(f"  (checkpoint val acc at train time: {checkpoint_val_acc:.4f})")

    if output_path is None:
        output_path = checkpoint_path.with_suffix(".eval.json")
    save_report(report, _resolve_checkpoint_path(output_path))

    if log_mlflow:
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        mlflow.set_experiment(cfg.mlflow.experiment)
        run_name = f"eval-{checkpoint_path.stem}-{split}"
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags({"model": model_name, "eval_split": split, "type": "evaluation"})
            mlflow.log_params({
                "checkpoint": str(checkpoint_path),
                "config_path": str(cfg.config_path),
                "split": split,
                "use_bbox_crop": use_bbox_crop,
            })
            mlflow.log_metrics({
                "eval_loss": metrics["loss"],
                "eval_top1_acc": metrics["top1_acc"],
                "eval_top5_acc": metrics["top5_acc"],
            })
            mlflow.log_artifact(str(output_path))

    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained bird classifier checkpoint")
    parser.add_argument("--config", required=True, type=Path, help="Training config YAML")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Model checkpoint .pt")
    parser.add_argument(
        "--split",
        default="test",
        choices=["test", "val"],
        help="test = official CUB test (default); val = holdout val split",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path for JSON report (default: <checkpoint>.eval.json)",
    )
    parser.add_argument("--mlflow", action="store_true", help="Log results to MLflow")
    args = parser.parse_args()

    cfg = load_config(args.config)
    evaluate_checkpoint(
        cfg,
        checkpoint_path=args.checkpoint,
        split=args.split,
        output_path=args.output,
        log_mlflow=args.mlflow,
    )


if __name__ == "__main__":
    main()
