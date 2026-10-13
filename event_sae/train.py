"""Train a BatchTopK SAE on offline activation shards."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from dictionary_learning.training import trainSAE
from dictionary_learning.trainers.batch_top_k import BatchTopKSAE, BatchTopKTrainer
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


def _auto_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


@dataclass
class SAETrainConfig:
    """Single-run config for training one BatchTopK SAE."""
    data_dir: str
    layer_idx: int             # Should match the layer number in shard filenames.
    activation_dim: int        # Activation width d_model (e.g., OpenVLA post-residual = 4096).
    dict_size: int             # Number of SAE features (dictionary width).
    k: int
    lr: float
    steps: int
    batch_size: int
    run_tag: str = "offline"
    submodule_name: str = "resid_post"
    device: str = ""           # Empty = auto (cuda if available else cpu).
    wandb_project: str = ""    # Empty = wandb disabled.
    wandb_group: str = ""
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    seed: int = 0
    warmup_steps: int = 1000
    decay_start_fraction: float = 0.8
    threshold_start_step: int = 1000
    auxk_alpha: float = 1 / 32
    save_interval: int = 1500
    log_steps: int = 500
    normalize_activations: bool = True

    def resolved_device(self) -> str:
        return self.device or _auto_device()


class _ActivationShardDataset(IterableDataset):
    def __init__(self, cfg: SAETrainConfig, shard_paths: list[Path]):
        super().__init__()
        self.cfg = cfg
        self.shard_paths = shard_paths

    def __iter__(self):
        worker = get_worker_info()
        if worker is None:
            local_shards = list(self.shard_paths)
        else:
            local_shards = list(self.shard_paths[worker.id :: worker.num_workers])
        if not local_shards:
            return
        generator = torch.Generator().manual_seed(
            self.cfg.seed + (worker.id if worker is not None else 0)
        )
        while True:
            if len(local_shards) > 1:
                order = torch.randperm(len(local_shards), generator=generator).tolist()
                shard_iter = [local_shards[i] for i in order]
            else:
                shard_iter = local_shards
            for shard_path in shard_iter:
                activations = torch.load(shard_path, map_location="cpu", weights_only=True)
                if activations.ndim != 2:
                    raise ValueError(
                        f"Expected 2D activations in {shard_path}, got shape {tuple(activations.shape)}"
                    )
                order = torch.randperm(activations.shape[0], generator=generator)
                activations = activations[order].to(torch.float32)
                for start in range(0, activations.shape[0], self.cfg.batch_size):
                    batch = activations[start : start + self.cfg.batch_size]
                    if batch.shape[0] < self.cfg.batch_size:
                        continue
                    yield batch


class ActivationShardDataLoader:
    """Iterable loader for layer-specific activation shards."""

    def __init__(self, cfg: SAETrainConfig):
        self.cfg = cfg
        self.device = cfg.resolved_device()
        shard_paths = sorted(Path(cfg.data_dir).glob(f"layer_{cfg.layer_idx:02d}_shard_*.pt"))
        if not shard_paths:
            raise FileNotFoundError(
                f"No activation shards found in {cfg.data_dir} for layer {cfg.layer_idx}."
            )
        viable_shards = []
        for path in shard_paths:
            tensor = torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
            if tensor.ndim != 2 or tensor.shape[1] != cfg.activation_dim:
                raise ValueError(f"Expected [rows, {cfg.activation_dim}] in {path}, got {tuple(tensor.shape)}")
            if tensor.shape[0] >= cfg.batch_size:
                viable_shards.append(path)
        if not viable_shards:
            raise ValueError(f"No shard has at least {cfg.batch_size} rows; reduce batch_size.")
        self.shard_paths = viable_shards
        self._dataset = _ActivationShardDataset(cfg, viable_shards)
        num_workers = min(cfg.num_workers, len(viable_shards))
        dataloader_kwargs: dict[str, Any] = {
            "dataset": self._dataset,
            "batch_size": None,
            "num_workers": num_workers,
            "pin_memory": cfg.pin_memory and self.device.startswith("cuda"),
            "persistent_workers": num_workers > 0,
        }
        if num_workers > 0:
            dataloader_kwargs["prefetch_factor"] = cfg.prefetch_factor
        self._dataloader = DataLoader(**dataloader_kwargs)

    def __iter__(self):
        for batch in self._dataloader:
            batch = batch.to(self.device, non_blocking=True)
            yield batch


def build_batch_topk_trainer_config(cfg: SAETrainConfig) -> dict[str, Any]:
    """Build a single trainer config dict for `dictionary_learning.trainSAE`."""
    return {
        "trainer": BatchTopKTrainer,
        "dict_class": BatchTopKSAE,
        "activation_dim": cfg.activation_dim,
        "dict_size": cfg.dict_size,
        "k": cfg.k,
        "lr": cfg.lr,
        "steps": cfg.steps,
        "warmup_steps": cfg.warmup_steps,
        "decay_start": int(cfg.steps * cfg.decay_start_fraction),
        "seed": cfg.seed,
        "threshold_start_step": cfg.threshold_start_step,
        "auxk_alpha": cfg.auxk_alpha,
        "device": cfg.resolved_device(),
        "layer": cfg.layer_idx,
        "lm_name": cfg.run_tag,
        "submodule_name": cfg.submodule_name,
        "wandb_name": f"{cfg.run_tag}-l{cfg.layer_idx:02d}",
    }


def train_sae(cfg: SAETrainConfig, save_dir: str) -> None:
    """Train one BatchTopK SAE run on offline activation shards."""
    dataloader = ActivationShardDataLoader(cfg)
    trainer_cfg = build_batch_topk_trainer_config(cfg)
    save_steps = list(range(cfg.save_interval, cfg.steps, cfg.save_interval))
    trainSAE(
        data=dataloader,
        trainer_configs=[trainer_cfg],
        steps=cfg.steps,
        save_dir=save_dir,
        save_steps=save_steps,
        log_steps=cfg.log_steps,
        normalize_activations=cfg.normalize_activations,
        use_wandb=bool(cfg.wandb_project),
        wandb_project=cfg.wandb_project,
        wandb_group=cfg.wandb_group,
        device=cfg.resolved_device(),
        verbose=True,
        autocast_dtype=torch.float32,
    )
