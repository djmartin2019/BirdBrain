# Data & splits

## CUB-200-2011 source

Download from the [official dataset page](https://www.vision.caltech.edu/datasets/cub_200_2011/) and extract to:

```
data/raw/CUB_200_2011/
├── images/                    # 200 species subfolders, ~11,788 JPEGs
├── images.txt                 # image_id → filepath
├── classes.txt                # class_id → folder name
├── image_class_labels.txt     # image_id → class_id (1–200)
├── train_test_split.txt       # image_id → is_train (1=train, 0=test)
└── bounding_boxes.txt         # image_id → x, y, width, height
```

**License:** Non-commercial research and educational use only.

## Three-way split strategy

CUB provides an official **train** (~5,994) and **test** (~5,794) split. Birdbrain adds a validation holdout carved from official train:

| Split | Filter | Count (approx) | Used for |
|-------|--------|----------------|----------|
| **train** | `is_train=1` AND `is_val=0` | 5,394 | Gradient updates |
| **val** | `is_train=1` AND `is_val=1` | 600 | Early stopping, best checkpoint |
| **test** | `is_train=0` | 5,794 | Final evaluation only |

The val assignment is stored in [`splits/val_split.txt`](../splits/val_split.txt) — one row per official-train image:

```
image_id is_val
```

Official test images are **not** in this file; `dataset.py` treats missing rows as `is_val=0`.

### Regenerating the val split

```bash
python scripts/create_val_split.py
python scripts/create_val_split.py --ratio 0.1 --seed 42
```

The script (`scripts/create_val_split.py`):

- Takes only official train images
- Stratifies by `class_id` so every species appears in val
- Writes `splits/val_split.txt`
- Never touches official test images

Configure the path in YAML: `data.val_split_file: splits/val_split.txt`

## Dataset class: `CUBDataset`

Implemented in [`training/dataset.py`](../training/dataset.py).

**Pipeline per sample:**

1. Merge `images.txt`, `image_class_labels.txt`, `train_test_split.txt`, and `val_split.txt`
2. Filter by requested `split` (`train`, `val`, or `test`)
3. Optionally merge `bounding_boxes.txt` and crop to bbox
4. Apply torchvision transform
5. Return `(tensor, label)` where label is **0–199** (CUB class IDs are 1–200)

### Bounding box crop

When `use_bbox_crop=True` (stage 5):

- Reads bbox from `bounding_boxes.txt`
- Crops with clamping to image bounds
- Falls back to full image if bbox is invalid

**Production note:** User-uploaded photos will not have CUB bboxes at inference time unless you add a detector or train a separate full-image stage.

## Augmentation presets

Defined in [`training/dataloaders.py`](../training/dataloaders.py). All presets use ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).

| Preset | Train transforms | Used in |
|--------|------------------|---------|
| **minimal** | Resize 256 → RandomResizedCrop 224 → HFlip | Stages 1–2 |
| **standard** | minimal + ColorJitter(0.2) | Stages 3–4 |
| **strong** | standard + RandomRotation(15) + RandomAffine | Stage 5 |

**Val / test / eval:** Resize 256 → CenterCrop 224 → normalize. No random augmentation.

## Label display names

Training uses integer labels 0–199. For human-readable names:

```bash
cd training
python make_labels.py
```

Writes `models/labels.json`: `{ "0": "Black footed Albatross", ... }`

The confusion matrix script falls back to parsing `classes.txt` if `labels.json` is missing or empty.

## DataLoaders

| Function | Splits | Shuffle | Augmentation |
|----------|--------|---------|--------------|
| `get_dataloaders(cfg)` | train + val | train yes, val no | From config |
| `get_eval_dataloader(cfg, split)` | val or test | no | Val transform only |

Both pass `val_split_file` from config into `CUBDataset`.
