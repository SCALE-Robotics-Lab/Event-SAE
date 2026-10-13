"""Read SAE intervention settings and select one ranked feature."""

import json
from pathlib import Path

import yaml


def load_steering_config(path):
    with Path(path).open() as stream:
        config = yaml.safe_load(stream)
    if config["mode"] not in ("native", "reconstruction", "feature"):
        raise ValueError("Steering mode must be native, reconstruction, or feature")
    if config["mode"] == "feature" and "candidates" in config:
        with Path(config["candidates"]).open() as stream:
            candidate = next(
                row for row in map(json.loads, stream)
                if row["ranking"] == config["ranking"] and row["rank"] == config["rank"]
            )
        config["feature_id"] = candidate["feature_id"]
        config["candidate"] = candidate
    return config
