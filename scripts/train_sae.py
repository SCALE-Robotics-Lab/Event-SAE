"""CLI: offline SAE training on cached activation shards.

Reads an `SAETrainConfig` from a YAML file and calls `event_sae.train.train_sae`.
"""

import argparse
from typing import Any

import yaml

from event_sae.train import SAETrainConfig, train_sae


def _coerce(raw: str) -> Any:
    if raw.lower() in {"true", "false"}:
        return raw.lower() == "true"
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to SAE training config YAML")
    parser.add_argument("--save-dir", required=True, help="Where to write SAE checkpoints")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override SAETrainConfig field, e.g. data_dir=logs/... or steps=200",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    for pair in args.override:
        if "=" not in pair:
            parser.error("--override requires field=value")
        key, raw = pair.split("=", 1)
        data[key] = _coerce(raw)
    cfg = SAETrainConfig(**data)
    train_sae(cfg, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
