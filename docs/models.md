# Models

Model definitions live in [`training/models.py`](../training/models.py).

## Architectures

### EfficientNet-B0 (`efficientnet_b0`)

- Source: `torchvision.models.efficientnet_b0`
- Pretrained: ImageNet (`EfficientNet_B0_Weights.DEFAULT`)
- Classification head: replaces `model.classifier[1]` with `Linear(in_features, 200)`
- Backbone blocks: `model.features` (sequential); last N blocks controlled by `unfreeze_blocks`

### ResNet50 (`resnet50`)

- Source: `torchvision.models.resnet50`
- Pretrained: ImageNet (`ResNet50_Weights.DEFAULT`)
- Classification head: replaces `model.fc` with `Linear(in_features, 200)`
- Backbone layers: `layer1`, `layer2`, `layer3`, `layer4`; last N layers controlled by `unfreeze_blocks`

## Freeze policy

`apply_freeze_policy(model, cfg)`:

1. Freeze **all** parameters
2. Unfreeze the last `unfreeze_blocks` backbone modules + head
3. Head is **always** trainable when any training occurs

| `unfreeze_blocks` | EfficientNet | ResNet50 |
|-------------------|--------------|----------|
| 0 | Head only | fc only |
| 1 | Last 1 feature block | layer4 |
| 2 | Last 2 feature blocks | layer3 + layer4 |
| 3 | Last 3 feature blocks | layer2 + layer3 + layer4 |
| 4 | — | layer1 + layer2 + layer3 + layer4 |
| 5 | Last 5 feature blocks | — |

## Split learning rates

When `backbone_learning_rate` is set and `unfreeze_blocks > 0`:

```python
[
  {"params": backbone_params, "lr": backbone_learning_rate},
  {"params": head_params,     "lr": learning_rate},
]
```

Otherwise a single param group uses `learning_rate`.

## Checkpoints

Saved by `trainer.save_checkpoint()` to `model.output_path` (e.g. `models/birdbrain_resnet50_v1-4.pt`).

### Checkpoint contents

```python
{
    "model_name": "resnet50",           # or "efficientnet_b0"
    "num_classes": 200,
    "state_dict": ...,                  # model weights
    "best_val_acc": 0.725,              # val acc when saved
    "image_size": 224,
    "use_bbox_crop": True,              # critical for inference parity
    "config_path": "...",               # YAML used for this stage
    "normalization": {                  # ImageNet mean/std
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
}
```

### Loading for continued training

`trainer.load_checkpoint()` loads `state_dict` with `strict=False` (allows head shape changes between stages) and returns prior `best_val_acc`.

### Loading for inference / eval

`build_model_for_inference(model_name, num_classes)` builds the architecture **without** pretrained weights, then loads `state_dict` from the checkpoint.

Eval scripts read `use_bbox_crop` and `model_name` from the checkpoint first, falling back to config.

### Production deployment

Training writes all stages to `models/`. Promoted inference checkpoints live in [`prod-models/`](../prod-models/README.md) and are mounted into the API container at runtime (see [`api/models.yaml`](../api/models.yaml)).

## Label map

[`make_labels.py`](../training/make_labels.py) exports class index → display name:

```json
{
  "0": "Black footed Albatross",
  "15": "Painted Bunting",
  "16": "Cardinal"
}
```

CUB folder names like `015.Red_winged_Blackbird` become `Red winged Blackbird`.

## Inference API

Upload inference is implemented in [`training/inference.py`](../training/inference.py) and served by [`api/`](../api/README.md).

### Preprocessing

User uploads use eval transforms from [`dataloaders.py`](../training/dataloaders.py): resize → center crop → ImageNet normalize, using `image_size` and `normalization` from the checkpoint.

**Bounding boxes:** Stage-5 checkpoints store `use_bbox_crop: true`, but arbitrary uploads have no CUB bbox metadata. The API **skips bbox crop** and runs on the full image. CUB test/val eval still uses bbox when the checkpoint flag is set.

### Response shape

`POST /api/predict` returns top-5 softmax percentages (0–100):

```json
{
  "model_id": "efficientnet_b0",
  "model_name": "EfficientNet-B0",
  "prediction": "Cardinal",
  "confidence": 94.2,
  "top_5": [
    { "species": "Cardinal", "percent": 94.2 },
    { "species": "Scarlet Tanager", "percent": 3.1 }
  ]
}
```

Users choose between the two production models via `GET /api/models` (see [`api/models.yaml`](../api/models.yaml)).
