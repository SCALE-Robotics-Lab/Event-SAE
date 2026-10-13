"""SAE classes and checkpoint loading for Event-SAE."""

import json
from pathlib import Path

from dictionary_learning.trainers.batch_top_k import BatchTopKSAE

__all__ = ["BatchTopKSAE", "load_batch_topk_sae"]


def load_batch_topk_sae(checkpoint_path: Path, device: str):
    """Load a BatchTopK SAE and its training config from a checkpoint directory."""
    checkpoint_path = Path(checkpoint_path)
    trainer_dir = checkpoint_path.parent if checkpoint_path.is_file() else checkpoint_path
    config_path = trainer_dir / "config.json"
    if not config_path.is_file():
        config_path = trainer_dir.parent / "config.json"
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    trainer_cfg = config["trainer"]
    if trainer_cfg.get("dict_class") != "BatchTopKSAE":
        raise ValueError(
            f"Expected BatchTopKSAE checkpoint, got dict_class={trainer_cfg.get('dict_class')!r}"
        )
    ae_path = checkpoint_path if checkpoint_path.is_file() else (trainer_dir / "ae.pt")
    sae = BatchTopKSAE.from_pretrained(str(ae_path), k=int(trainer_cfg["k"]), device=device)
    sae.eval()
    return sae, config
