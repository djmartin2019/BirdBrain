# Production model checkpoints

Final inference checkpoints served by the API in Docker/production.

| File | Model |
|------|--------|
| `birdbrain_inat_v1-3.pt` | BirdBrain Voyager (EfficientNet-B0, iNat birds, stage 4) |
| `birdbrain_v1-4.pt` | EfficientNet-B0 (CUB-200, stage 5) |
| `birdbrain_resnet50_v1-4.pt` | ResNet-50 (CUB-200, stage 5) |

Weights are gitignored (`.gitignore`); ship to the VPS with `rsync` or run
[`scripts/sync-prod-models.sh`](../scripts/sync-prod-models.sh) locally after
retraining, then copy this directory to the server.

Training still writes all stages to `models/`; copy promoted checkpoints here
when ready for production.
