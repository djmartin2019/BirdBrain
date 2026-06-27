import torch.nn as nn
from torchvision import models

from config import TrainConfig


def _build_efficientnet_b0(num_classes: int, pretrained: bool):
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def _build_resnet50(num_classes: int, pretrained: bool):
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


MODEL_BUILDERS = {
    "efficientnet_b0": _build_efficientnet_b0,
    "resnet50": _build_resnet50,
}


def _efficientnet_backbone_modules(model, num_blocks: int):
    if num_blocks <= 0:
        return []
    return list(model.features[-num_blocks:])


def _efficientnet_head_modules(model):
    return [model.classifier]


def _resnet_backbone_modules(model, num_blocks: int):
    layers = [model.layer1, model.layer2, model.layer3, model.layer4]
    if num_blocks <= 0:
        return []
    return layers[-num_blocks:]


def _resnet_head_modules(model):
    return [model.fc]


def apply_freeze_policy(model, cfg: TrainConfig):
    """Freeze backbone layers according to config; head always trainable."""
    for param in model.parameters():
        param.requires_grad = False

    if cfg.model.name == "efficientnet_b0":
        backbone_modules = _efficientnet_backbone_modules(
            model, cfg.training.unfreeze_blocks
        )
        head_modules = _efficientnet_head_modules(model)
    elif cfg.model.name == "resnet50":
        backbone_modules = _resnet_backbone_modules(
            model, cfg.training.unfreeze_blocks
        )
        head_modules = _resnet_head_modules(model)
    else:
        raise ValueError(f"Unsupported model: {cfg.model.name}")

    for module in backbone_modules + head_modules:
        for param in module.parameters():
            param.requires_grad = True


def build_model(cfg: TrainConfig):
    builder = MODEL_BUILDERS[cfg.model.name]
    model = builder(cfg.model.num_classes, cfg.model.pretrained)
    apply_freeze_policy(model, cfg)
    return model


def get_param_groups(model, cfg: TrainConfig):
    """Build optimizer param groups with optional split learning rates."""
    trainable = [p for p in model.parameters() if p.requires_grad]

    use_split_lr = (
        cfg.training.unfreeze_blocks > 0
        and cfg.training.backbone_learning_rate is not None
    )

    if not use_split_lr:
        return [{"params": trainable, "lr": cfg.training.learning_rate}]

    if cfg.model.name == "efficientnet_b0":
        backbone_modules = _efficientnet_backbone_modules(
            model, cfg.training.unfreeze_blocks
        )
        head_modules = _efficientnet_head_modules(model)
    else:
        backbone_modules = _resnet_backbone_modules(
            model, cfg.training.unfreeze_blocks
        )
        head_modules = _resnet_head_modules(model)

    backbone_params = []
    for module in backbone_modules:
        backbone_params.extend(p for p in module.parameters() if p.requires_grad)

    head_params = []
    for module in head_modules:
        head_params.extend(p for p in module.parameters() if p.requires_grad)

    return [
        {"params": backbone_params, "lr": cfg.training.backbone_learning_rate},
        {"params": head_params, "lr": cfg.training.learning_rate},
    ]
