import argparse
from pathlib import Path

from config import load_config
from trainer import train


def main():
    parser = argparse.ArgumentParser(description="Train a bird classifier from YAML config")
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to training config YAML (e.g. ../configs/efficientnet_stage1_head.yaml)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
