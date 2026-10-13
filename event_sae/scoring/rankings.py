"""Rank SAE features by event alignment, window mean, task mean or random selection.

Score-based rankings read the scorer's matrices. Suite-level rankings filter
cluster rows by episode coverage; each strategy defines its aggregation weights.
Random controls sample positive-activation features excluding informed selections.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def _load_manifest(topk_run_dir: Path) -> dict:
    manifest_path = topk_run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest.json: {manifest_path}")
    import json
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("format") != "token_topk_sparse_v1":
        raise ValueError(f"Unsupported manifest format: {manifest.get('format')!r}")
    return manifest


def _iter_shards(topk_run_dir: Path, manifest: dict, *, desc: str | None = None):
    shards = manifest["shards"]
    it = shards
    if tqdm is not None and desc:
        it = tqdm(shards, desc=desc, unit="shard")
    for shard_meta in it:
        payload = torch.load(topk_run_dir / shard_meta["path"], map_location="cpu")
        yield payload


def _top_pairs(vec: torch.Tensor, top_n: int) -> list[dict]:
    k = min(top_n, int(vec.numel()))
    values, indices = torch.topk(vec, k=k, largest=True)
    return [{"feature_id": int(i), "score": float(v)} for i, v in zip(indices.tolist(), values.tolist())]


def _load_matrix(scores_pt_path: Path, key: str) -> tuple[torch.Tensor, list[dict]]:
    """Load a named matrix and row keys from the scorer output."""
    payload = torch.load(Path(scores_pt_path).resolve(), map_location="cpu")
    return payload[key].to(dtype=torch.float32), list(payload["row_keys"])


def _load_event_aligned_matrix(scores_pt_path: Path) -> tuple[torch.Tensor, list[dict]]:
    return _load_matrix(scores_pt_path, "matrix_raw")


def event_aligned_top_features_per_row(scores_pt_path: Path, top_n: int) -> list[dict]:
    """Return the top-N event-aligned features for each cluster row."""
    matrix, row_keys = _load_event_aligned_matrix(scores_pt_path)
    out: list[dict] = []
    for row_idx, meta in enumerate(row_keys):
        out.append(
            {
                "ranking": "event_aligned",
                "task_id": meta["task_id"],
                "task_suite_name": meta["task_suite_name"],
                "task_description": str(meta["task_description"]),
                "cluster_id": str(meta["cluster_id"]),
                "phrase": str(meta.get("phrase", "")),
                "phase": str(meta.get("phase", "")),
                "top_features": _top_pairs(matrix[row_idx], top_n),
            }
        )
    return out


def event_aligned_suite_top_k(
    scores_pt_path: Path, top_k: int, *, min_coverage: float = 0.5
) -> list[dict]:
    """Average rows with ``episode_coverage >= min_coverage`` equally, then take top-K."""
    matrix, row_keys = _load_event_aligned_matrix(scores_pt_path)
    keep_idx = [
        i
        for i, row in enumerate(row_keys)
        if float(row.get("episode_coverage", 0.0)) >= min_coverage
    ]
    if not keep_idx:
        raise RuntimeError(
            f"No canonical rows with episode_coverage >= {min_coverage}; "
            f"total rows={len(row_keys)}."
        )
    suite_vec = matrix[keep_idx].mean(dim=0)
    return _top_pairs(suite_vec, top_k)


def window_mean_top_features_per_row(
    *,
    scores_pt_path: Path,
    top_n: int,
) -> list[dict]:
    """Return top-N window-mean features per cluster, omitting zero rows."""
    matrix, row_keys = _load_matrix(scores_pt_path, "matrix_window_mean")
    out: list[dict] = []
    for row_idx, meta in enumerate(row_keys):
        if float(matrix[row_idx].abs().sum().item()) == 0.0:
            continue
        out.append(
            {
                "ranking": "window_mean",
                "task_id": meta["task_id"],
                "task_suite_name": meta["task_suite_name"],
                "task_description": str(meta["task_description"]),
                "cluster_id": str(meta["cluster_id"]),
                "phrase": str(meta.get("phrase", "")),
                "phase": str(meta.get("phase", "")),
                "top_features": _top_pairs(matrix[row_idx], top_n),
            }
        )
    return out


def window_mean_suite_top_k(
    *,
    scores_pt_path: Path,
    top_k: int,
    min_coverage: float = 0.5,
) -> list[dict]:
    """Weight coverage-filtered cluster rows by ``num_events``, then take top-K."""
    matrix, row_keys = _load_matrix(scores_pt_path, "matrix_window_mean")
    keep_idx = [
        i
        for i, row in enumerate(row_keys)
        if float(row.get("episode_coverage", 0.0)) >= min_coverage
    ]
    if not keep_idx:
        raise RuntimeError(
            f"No canonical rows with episode_coverage >= {min_coverage}; "
            f"total rows={len(row_keys)}."
        )
    matrix = matrix[keep_idx]
    num_events_per_row = [int(row.get("num_events", 0)) for row in row_keys]
    weights = torch.tensor(
        [float(num_events_per_row[i]) for i in keep_idx], dtype=torch.float32
    )
    if float(weights.sum().item()) <= 0:
        raise RuntimeError("Total event-weight is zero for window-mean aggregation.")
    suite_vec = (matrix * weights[:, None]).sum(dim=0) / weights.sum()
    return _top_pairs(suite_vec, top_k)


def task_mean_top_features_per_task(
    *,
    scores_pt_path: Path,
    top_n: int,
) -> list[dict]:
    """Return top-N task-mean features, deduplicating cluster rows by suite and task ID."""
    matrix, row_keys = _load_matrix(scores_pt_path, "matrix_task_mean")
    seen: dict[tuple, int] = {}
    for i, meta in enumerate(row_keys):
        task = (meta["task_suite_name"], meta["task_id"])
        if task not in seen:
            seen[task] = i
    out: list[dict] = []
    for idx in sorted(seen.values(), key=lambda i: row_keys[i]["task_description"]):
        meta = row_keys[idx]
        out.append(
            {
                "ranking": "task_mean",
                "task_description": str(meta["task_description"]),
                "task_id": meta["task_id"],
                "task_suite_name": meta["task_suite_name"],
                "top_features": _top_pairs(matrix[idx], top_n),
            }
        )
    return out


def task_mean_suite_top_k(
    *,
    scores_pt_path: Path,
    top_k: int,
    min_coverage: float = 0.5,
) -> list[dict]:
    """Deduplicate coverage-filtered rows by suite and task ID, then rank task means.

    Weight each task by its cached timestep count, using 1 for missing or zero counts.
    """
    payload = torch.load(Path(scores_pt_path).resolve(), map_location="cpu")
    matrix = payload["matrix_task_mean"].to(dtype=torch.float32)
    row_keys = list(payload["row_keys"])
    keep_idx = [
        i
        for i, row in enumerate(row_keys)
        if float(row.get("episode_coverage", 0.0)) >= min_coverage
    ]
    if not keep_idx:
        raise RuntimeError(
            f"No canonical rows with episode_coverage >= {min_coverage}; "
            f"total rows={len(row_keys)}."
        )
    task_timestep_counts = (
        payload.get("selection_counts", {}).get("task_timestep_counts", {}) or {}
    )

    seen_tasks: set[tuple] = set()
    vectors: list[torch.Tensor] = []
    weights: list[float] = []
    for i in keep_idx:
        meta = row_keys[i]
        task = (meta["task_suite_name"], meta["task_id"])
        if task in seen_tasks:
            continue
        seen_tasks.add(task)
        vectors.append(matrix[i])
        task_id = meta.get("task_id")
        weight = task_timestep_counts.get(task_id)
        if weight is None and task_id is not None:
            weight = task_timestep_counts.get(str(task_id))
        weights.append(float(weight) if weight else 1.0)
    if not vectors:
        raise RuntimeError("No tasks remained for task_mean suite aggregation.")
    stacked = torch.stack(vectors, dim=0)
    weight_t = torch.tensor(weights, dtype=torch.float32)
    if float(weight_t.sum().item()) <= 0:
        raise RuntimeError("Total task-timestep weight is zero for task_mean.")
    suite_vec = (stacked * weight_t[:, None]).sum(dim=0) / weight_t.sum()
    return _top_pairs(suite_vec, top_k)


def alive_feature_ids(topk_run_dir: Path) -> set[int]:
    """Feature IDs with a positive activation anywhere in the cache."""
    topk_run_dir = Path(topk_run_dir).resolve()
    manifest = _load_manifest(topk_run_dir)
    alive: set[int] = set()
    for payload in _iter_shards(topk_run_dir, manifest, desc="alive scan"):
        ids = payload["top_feature_ids"].to(dtype=torch.int64)
        values = payload["top_feature_vals"]
        alive.update(torch.unique(ids[values > 0]).tolist())
    return alive


def random_alive_features(
    *,
    topk_run_dir: Path,
    num_features: int,
    exclude_feature_ids: set[int],
    seed: int = 0,
) -> list[int]:
    """Uniformly sample positive-activation features excluding informed selections."""
    alive = alive_feature_ids(topk_run_dir)
    candidates = sorted(alive - set(int(x) for x in exclude_feature_ids))
    if len(candidates) < num_features:
        raise RuntimeError(
            f"Not enough alive features after exclusion: have {len(candidates)}, need {num_features}"
        )
    rng = random.Random(seed)
    return sorted(rng.sample(candidates, num_features))
