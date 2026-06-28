"""
Standalone confusion matrix analysis for a trained checkpoint.

Usage (from training/):
    python confusion_matrix.py --config ../configs/resnet50_stage5_bbox.yaml \\
        --checkpoint ../models/birdbrain_resnet50_v1-4.pt --split test --top-k 20
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from config import load_config, resolve_cli_path
from dataloaders import get_eval_dataloader
from models import build_model_for_inference
from trainer import get_device

DARK_THEME = {
    "template": "plotly_dark",
    "paper_bgcolor": "#111827",
    "plot_bgcolor": "#1f2937",
    "font": {"color": "#e5e7eb", "size": 11},
    "title": {"font": {"color": "#f9fafb", "size": 16}},
}


def _apply_dark_theme(fig, height=None, width=None, left_margin=200):
    layout = {**DARK_THEME, "margin": {"l": left_margin, "r": 50, "t": 70, "b": 70}}
    if height is not None:
        layout["height"] = height
    if width is not None:
        layout["width"] = width
    fig.update_layout(**layout)
    fig.update_xaxes(gridcolor="#374151", zerolinecolor="#374151", linecolor="#4b5563")
    fig.update_yaxes(gridcolor="#374151", zerolinecolor="#374151", linecolor="#4b5563")


def _set_all_category_ticks(fig, labels: list[str], x_angle: int = -45, font_size: int = 9):
    """Force every category label on heatmap axes via numeric tick positions."""
    n = len(labels)
    tickvals = list(range(n))
    tick_cfg = dict(
        tickmode="array",
        tickvals=tickvals,
        ticktext=labels,
        tickfont=dict(size=font_size),
    )
    fig.update_xaxes(**tick_cfg, tickangle=x_angle, side="top")
    fig.update_yaxes(**tick_cfg, autorange="reversed")


def select_worst_classes(recall_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Pick worst-K classes by recall among those present in the split."""
    present = recall_df[recall_df["support"] > 0].copy()
    if present.empty:
        return present

    return (
        present.sort_values(["recall", "support"], ascending=[True, False])
        .head(top_k)
        .sort_values("recall", ascending=True)
        .reset_index(drop=True)
    )

def load_class_names(labels_path: Path | None, data_dir: Path, num_classes: int) -> dict[int, str]:
    """Load index -> class name map from labels.json or CUB classes.txt."""
    if labels_path is not None:
        labels_path = resolve_cli_path(labels_path)
        if labels_path.exists() and labels_path.stat().st_size > 0:
            with open(labels_path) as f:
                raw = json.load(f)
            if raw:
                return {int(k): v for k, v in raw.items()}

    classes_file = data_dir / "classes.txt"
    if not classes_file.exists():
        return {i: str(i) for i in range(num_classes)}

    classes = pd.read_csv(classes_file, sep=" ", names=["class_id", "class_name"])
    return {
        int(row.class_id) - 1: row.class_name.split(".", 1)[1].replace("_", " ")
        for row in classes.itertuples()
    }


def collect_predictions(model, dataloader, device) -> tuple[np.ndarray, np.ndarray]:
    """Run inference and return (y_true, y_pred) as numpy arrays."""
    model.eval()
    y_true_list = []
    y_pred_list = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            y_true_list.append(labels.cpu().numpy())
            y_pred_list.append(preds.cpu().numpy())

    return np.concatenate(y_true_list), np.concatenate(y_pred_list)


def compute_per_class_recall(cm: np.ndarray, class_names: dict[int, str]) -> pd.DataFrame:
    """Per-class recall = diagonal / row sum."""
    row_sums = cm.sum(axis=1)
    recalls = np.divide(
        np.diag(cm),
        row_sums,
        out=np.zeros(cm.shape[0], dtype=float),
        where=row_sums > 0,
    )
    rows = []
    for idx in range(cm.shape[0]):
        rows.append({
            "class_index": idx,
            "class_name": class_names.get(idx, str(idx)),
            "support": int(row_sums[idx]),
            "recall": float(recalls[idx]),
        })
    return pd.DataFrame(rows)


def compute_confused_pairs(
    cm: np.ndarray,
    class_names: dict[int, str],
    top_n: int,
) -> pd.DataFrame:
    """Off-diagonal (true, pred) pairs sorted by count descending."""
    pairs = []
    for true_idx in range(cm.shape[0]):
        for pred_idx in range(cm.shape[1]):
            if true_idx == pred_idx:
                continue
            count = int(cm[true_idx, pred_idx])
            if count <= 0:
                continue
            pairs.append({
                "true_label": true_idx,
                "pred_label": pred_idx,
                "true_name": class_names.get(true_idx, str(true_idx)),
                "pred_name": class_names.get(pred_idx, str(pred_idx)),
                "count": count,
            })

    df = pd.DataFrame(pairs)
    if df.empty:
        return df
    return df.sort_values("count", ascending=False).head(top_n).reset_index(drop=True)


def plot_worst_recall_bar(recall_df: pd.DataFrame, top_k: int, output_path: Path):
    """Horizontal bar chart of worst-K classes by recall."""
    worst = select_worst_classes(recall_df, top_k)
    if worst.empty:
        raise ValueError("No classes with support > 0 found for bar chart")

    class_names = worst["class_name"].tolist()
    left_margin = max(200, max((len(n) for n in class_names), default=10) * 7)
    text_labels = [f"{r:.1%}" for r in worst["recall"]]

    fig = px.bar(
        worst,
        x="recall",
        y="class_name",
        orientation="h",
        title=f"Worst {len(worst)} Classes by Recall (support > 0)",
        labels={"recall": "Recall", "class_name": "Class"},
        text=text_labels,
        color_discrete_sequence=["#60a5fa"],
    )
    fig.update_traces(
        hovertemplate="%{y}<br>Recall: %{x:.1%}<br>Support: %{customdata[0]}<extra></extra>",
        customdata=worst[["support"]].values,
        textposition="outside",
        cliponaxis=False,
    )
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=class_names,
        tickfont=dict(size=10),
        automargin=True,
    )
    x_max = max(0.35, float(worst["recall"].max()) * 1.2)
    fig.update_xaxes(range=[0, x_max])
    _apply_dark_theme(fig, height=max(400, len(worst) * 32), left_margin=left_margin)
    fig.write_html(str(output_path))


def plot_topk_submatrix(
    cm: np.ndarray,
    class_indices: list[int],
    class_names: dict[int, str],
    normalize_rows: bool,
    output_path: Path,
):
    """K×K heatmap for selected class indices."""
    sub = cm[np.ix_(class_indices, class_indices)].astype(float)
    if normalize_rows:
        row_sums = sub.sum(axis=1, keepdims=True)
        sub = np.divide(sub, row_sums, out=np.zeros_like(sub), where=row_sums > 0)
        color_label = "Row-normalized count"
        title_suffix = " (row-normalized)"
    else:
        color_label = "Count"
        title_suffix = ""

    labels = [class_names.get(i, str(i)) for i in class_indices]
    k = len(labels)
    px_size = max(32, min(40, 800 // k))
    chart_size = max(600, k * px_size)
    tickvals = list(range(k))

    fig = px.imshow(
        sub,
        x=tickvals,
        y=tickvals,
        labels={"x": "Predicted", "y": "True", "color": color_label},
        title=f"Top-{k} Worst-Recall Submatrix{title_suffix}",
        aspect="auto",
        color_continuous_scale="Blues",
    )
    _set_all_category_ticks(fig, labels, x_angle=-45, font_size=max(8, min(10, 200 // k)))
    _apply_dark_theme(fig, height=chart_size, width=chart_size, left_margin=chart_size // 3)
    fig.write_html(str(output_path))


def run_confusion_analysis(
    cfg,
    checkpoint_path: Path,
    split: str = "test",
    output_dir: Path | None = None,
    top_k: int = 20,
    pairs_top_n: int = 50,
    labels_path: Path | None = None,
    normalize_rows: bool = False,
):
    checkpoint_path = resolve_cli_path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if output_dir is None:
        output_dir = checkpoint_path.parent / f"{checkpoint_path.stem}_confusion"
    output_dir = resolve_cli_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint.get("model_name", cfg.model.name)
    num_classes = checkpoint.get("num_classes", cfg.model.num_classes)
    use_bbox_crop = checkpoint.get("use_bbox_crop", cfg.data.use_bbox_crop)

    model = build_model_for_inference(model_name, num_classes).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    print(f"Loaded {checkpoint_path}")

    loader = get_eval_dataloader(cfg, split=split, use_bbox_crop=use_bbox_crop)
    print(f"Analyzing split={split!r}, samples={len(loader.dataset)}, bbox_crop={use_bbox_crop}")

    class_names = load_class_names(labels_path, cfg.data.data_dir, num_classes)
    y_true, y_pred = collect_predictions(model, loader, device)

    labels = list(range(num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    top1_acc = float((y_true == y_pred).mean())
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    recall_df = compute_per_class_recall(cm, class_names)
    confused_pairs_df = compute_confused_pairs(cm, class_names, pairs_top_n)

    worst_df = select_worst_classes(recall_df, top_k)
    worst_indices = worst_df["class_index"].tolist()

    cm_path = output_dir / "confusion_matrix.csv"
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.index.name = "true"
    cm_df.columns.name = "pred"
    cm_df.to_csv(cm_path)

    recall_path = output_dir / "per_class_recall.csv"
    recall_df.to_csv(recall_path, index=False)

    pairs_path = output_dir / "confused_pairs.csv"
    confused_pairs_df.to_csv(pairs_path, index=False)

    bar_path = output_dir / "worst_recall_bar.html"
    plot_worst_recall_bar(recall_df, top_k, bar_path)

    submatrix_path = output_dir / "topk_submatrix.html"
    plot_topk_submatrix(cm, worst_indices, class_names, normalize_rows, submatrix_path)

    clf_report = classification_report(
        y_true, y_pred, labels=labels, zero_division=0, output_dict=True
    )
    report_path = output_dir / "classification_report.json"
    with open(report_path, "w") as f:
        json.dump(clf_report, f, indent=2)

    summary = {
        "checkpoint": str(checkpoint_path),
        "config_path": str(cfg.config_path),
        "split": split,
        "model_name": model_name,
        "num_classes": num_classes,
        "use_bbox_crop": use_bbox_crop,
        "num_samples": int(len(y_true)),
        "top1_acc": top1_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "top_k": top_k,
        "pairs_top_n": pairs_top_n,
        "artifacts": {
            "confusion_matrix": str(cm_path),
            "per_class_recall": str(recall_path),
            "confused_pairs": str(pairs_path),
            "worst_recall_bar": str(bar_path),
            "topk_submatrix": str(submatrix_path),
            "classification_report": str(report_path),
        },
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults ({split}):")
    print(f"  Top-1:       {top1_acc:.4f}")
    print(f"  Macro F1:    {macro_f1:.4f}")
    print(f"  Weighted F1: {weighted_f1:.4f}")
    print(f"\nOutputs written to: {output_dir}")
    for name, path in summary["artifacts"].items():
        print(f"  {name}: {path}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Confusion matrix analysis for a trained bird classifier checkpoint"
    )
    parser.add_argument("--config", required=True, type=Path, help="Training config YAML")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Model checkpoint .pt")
    parser.add_argument(
        "--split",
        default="test",
        choices=["test", "val"],
        help="test = official CUB test (default); val = holdout val split",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <checkpoint_stem>_confusion/)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Worst-recall classes for submatrix and bar chart (default: 20)",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("models/labels.json"),
        help="Class name map JSON (default: models/labels.json)",
    )
    parser.add_argument(
        "--pairs-top-n",
        type=int,
        default=50,
        help="Max rows in confused_pairs.csv (default: 50)",
    )
    parser.add_argument(
        "--normalize-rows",
        action="store_true",
        help="Row-normalize the K×K submatrix heatmap",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_confusion_analysis(
        cfg,
        checkpoint_path=args.checkpoint,
        split=args.split,
        output_dir=args.output_dir,
        top_k=args.top_k,
        pairs_top_n=args.pairs_top_n,
        labels_path=args.labels,
        normalize_rows=args.normalize_rows,
    )


if __name__ == "__main__":
    main()
