# Evaluation & analysis

Tools for measuring model quality **after** training. None of these run during the training epoch loop.

## `evaluate.py` — scalar metrics

Scores a checkpoint on val or official test.

```bash
cd training
python evaluate.py \
  --config ../configs/resnet50_stage5_bbox.yaml \
  --checkpoint ../models/birdbrain_resnet50_v1-4.pt \
  --split test
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | required | YAML for data paths and batch size |
| `--checkpoint` | required | `.pt` file |
| `--split` | `test` | `test` (official CUB) or `val` (600-image holdout) |
| `--output` | `<checkpoint>.eval.json` | JSON report path |
| `--mlflow` | off | Log eval run to MLflow |

### Metrics reported

- **Loss** — cross-entropy
- **Top-1 accuracy** — argmax prediction
- **Top-5 accuracy** — correct class in top 5 logits
- **Sample count**

### Output: `.eval.json`

```json
{
  "checkpoint": "...",
  "split": "test",
  "top1_acc": 0.7755,
  "top5_acc": 0.9417,
  "loss": 0.9698,
  "use_bbox_crop": true,
  "checkpoint_best_val_acc": 0.7755
}
```

**When to use each split:**

| Split | Purpose |
|-------|---------|
| `val` | Quick re-check during development; ~3 images/class (noisy per-class stats) |
| `test` | Final reportable metric; ~29 images/class; run **once** per model version |

---

## `confusion_matrix.py` — per-class analysis

Standalone script for understanding *which* species fail and *how* they are confused.

```bash
python confusion_matrix.py \
  --config ../configs/resnet50_stage5_bbox.yaml \
  --checkpoint ../models/birdbrain_resnet50_v1-4.pt \
  --split test \
  --top-k 20
```

| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir` | `<checkpoint_stem>_confusion/` | All artifacts |
| `--top-k` | 20 | Worst-K classes for charts |
| `--pairs-top-n` | 50 | Rows in confused-pairs CSV |
| `--labels` | `models/labels.json` | Class name map |
| `--normalize-rows` | off | Row-normalize submatrix heatmap |

### Output directory

Written to `models/<checkpoint_stem>_confusion/` by default:

| File | Format | Purpose |
|------|--------|---------|
| `summary.json` | JSON | Top-1, macro/weighted F1, artifact paths |
| `confusion_matrix.csv` | CSV | Full 200×200 count matrix |
| `per_class_recall.csv` | CSV | Every class: name, support, recall |
| `confused_pairs.csv` | CSV | Top off-diagonal (true→pred) mistakes |
| `classification_report.json` | JSON | sklearn per-class precision/recall/F1 |
| `worst_recall_bar.html` | Plotly | Dark-theme horizontal bar chart |
| `topk_submatrix.html` | Plotly | K×K heatmap for worst-recall classes |

### How to read the charts

**Worst recall bar chart**

- Shows the K classes with **lowest recall** among those with `support > 0`
- Recall = diagonal / row sum in confusion matrix = "when true class is X, how often did we predict X?"
- Lower bar = more misclassification for that species
- Zero-recall bars show a `0.0%` label even when the bar has no width

**Top-K submatrix heatmap**

- Same K worst-recall classes on both axes
- **Rows** = true species, **columns** = predicted species
- **Diagonal** = correct predictions within this subset
- **Off-diagonal** = confusions between hard species
- Open in a browser; dark theme; hover for values

**Confused pairs CSV**

- Sorted by count descending
- Columns: `true_label`, `pred_label`, `true_name`, `pred_name`, `count`
- Useful for spotting systematic mix-ups (e.g. similar-looking warblers)

### Val vs test for confusion analysis

On **val** (600 images), many classes have only ~3 samples — recalls cluster at 0%, 33%, 67%, 100%. Charts look flat but data is valid.

On **test** (5,794 images), per-class stats are more stable and charts show more spread. Prefer `--split test` for reports.

---

## MLflow — training history display

Training metrics display in the MLflow UI, not in static files.

```bash
mlflow ui --backend-store-uri "sqlite:///$(pwd)/mlflow.db"
```

Each training stage = one run. Compare `val_acc` curves across stages and models.

Eval runs (`evaluate.py --mlflow`) create separate runs tagged `type=evaluation` (when implemented consistently).

---

## Planned production display

[`db/schema.sql`](../db/schema.sql) sketches Postgres logging for live predictions:

```sql
predictions (
  id, filename, predicted_species, confidence, top_5 JSONB, created_at
)
```

The web UI at `birdbrain.djm-apps.com` (planned) would show upload results in the JSON format documented in the README. This is not implemented yet.

---

## Recommended evaluation workflow

After completing stage 5:

```bash
# 1. Scalar test metrics (once)
python evaluate.py --config ../configs/resnet50_stage5_bbox.yaml \
  --checkpoint ../models/birdbrain_resnet50_v1-4.pt --split test --mlflow

# 2. Per-class analysis
python confusion_matrix.py --config ../configs/resnet50_stage5_bbox.yaml \
  --checkpoint ../models/birdbrain_resnet50_v1-4.pt --split test --top-k 20

# 3. Open HTML reports in browser
open ../models/birdbrain_resnet50_v1-4_confusion/worst_recall_bar.html
open ../models/birdbrain_resnet50_v1-4_confusion/topk_submatrix.html
```
