from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from fastwam.datasets.lerobot.utils.normalizer import save_dataset_stats_to_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="configs/data/libero_2cam_fastwam_v30.yaml")
    parser.add_argument(
        "--stats-output",
        default="data/fastwam_release/dataset_stats.json",
    )
    parser.add_argument(
        "--metric-output",
        default="data/fastwam_release/fdwam_action_metric.pt",
    )
    parser.add_argument("--shrinkage", type=float, default=0.2)
    parser.add_argument("--std-epsilon", type=float, default=1.0e-6)
    return parser.parse_args()


def load_train_config(path: str) -> Any:
    loaded = OmegaConf.load(path)
    root = OmegaConf.create({"data": {"train": loaded.train}})
    OmegaConf.resolve(root)
    return root.data.train


def tensor_column(dataset: Any, key: str) -> torch.Tensor:
    values = dataset.hf_dataset[key]
    if isinstance(values, torch.Tensor):
        result = values
    else:
        result = torch.stack([torch.as_tensor(value) for value in values])
    if result.ndim == 1:
        result = result.unsqueeze(-1)
    return result.to(dtype=torch.float32)


def collect_raw_actions(base_dataset: Any) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    if int(base_dataset.global_sample_stride) != 1:
        raise RuntimeError("The FDWAM metric scan requires global_sample_stride=1.")

    action_meta = base_dataset.action_meta
    state_meta = base_dataset.state_meta
    action_rows = []
    manifest = []
    seen_dirs = set()

    for dataset in base_dataset.multi_dataset._datasets:
        root = str(Path(dataset.root).resolve())
        if root in seen_dirs:
            raise RuntimeError(f"Dataset directory is listed more than once: {root}")
        seen_dirs.add(root)

        action_parts = []
        state_parts = []
        for meta in action_meta:
            action_parts.append(tensor_column(dataset, meta["lerobot_key"]))
        for meta in state_meta:
            state_parts.append(tensor_column(dataset, meta["lerobot_key"]))

        actions = torch.cat(action_parts, dim=-1)
        states = torch.cat(state_parts, dim=-1)
        if actions.shape[0] != states.shape[0]:
            raise RuntimeError(f"Action/state row count mismatch in {root}.")
        if not torch.isfinite(actions).all() or not torch.isfinite(states).all():
            raise RuntimeError(f"Non-finite action or state value found in {root}.")

        action_rows.append((actions, states))
        manifest.append(
            {
                "root": root,
                "frames": int(actions.shape[0]),
                "episodes": int(dataset.num_episodes),
            }
        )

    if not action_rows:
        raise RuntimeError("No training datasets were found.")
    return action_rows, manifest


def normalize_actions(
    action_rows: list[tuple[torch.Tensor, torch.Tensor]],
    processor: Any,
) -> torch.Tensor:
    normalized_rows = []
    action_key = processor.shape_meta["action"][0]["key"]
    state_key = processor.shape_meta["state"][0]["key"]

    for actions, states in action_rows:
        batch = {
            "action": {action_key: actions},
            "state": {state_key: states},
        }
        batch = processor.action_state_transform(batch)
        batch = processor.normalizer.forward(batch)
        batch = processor.action_state_merger.forward(batch)
        if bool(batch["action_dim_is_pad"].any().item()):
            raise RuntimeError("The action processor introduced padded action dimensions.")
        normalized_rows.append(batch["action"].to(dtype=torch.float64))

    actions = torch.cat(normalized_rows, dim=0)
    if actions.ndim != 2:
        raise RuntimeError(f"Expected normalized actions with shape [K,D], got {tuple(actions.shape)}")
    if not torch.isfinite(actions).all():
        raise RuntimeError("The normalized action matrix contains NaN or Inf.")
    return actions


def build_metric(actions: torch.Tensor, std_epsilon: float, shrinkage: float) -> dict[str, torch.Tensor | float | int]:
    count, action_dim = actions.shape
    if count < 2:
        raise RuntimeError(f"At least two action rows are required, got {count}.")
    if not 0.0 < shrinkage <= 1.0:
        raise RuntimeError(f"shrinkage must be in (0, 1], got {shrinkage}.")

    actions = actions.to(dtype=torch.float64)
    mean = actions.mean(dim=0)
    centered = actions - mean
    covariance = centered.transpose(0, 1) @ centered / float(count - 1)
    std = covariance.diagonal().clamp_min(0.0).sqrt()
    active = std > float(std_epsilon)

    correlation = torch.eye(action_dim, dtype=torch.float64)
    active_indices = torch.nonzero(active, as_tuple=False).flatten()
    if active_indices.numel() > 0:
        active_covariance = covariance.index_select(0, active_indices).index_select(1, active_indices)
        active_std = std.index_select(0, active_indices)
        active_correlation = active_covariance / (active_std[:, None] * active_std[None, :])
        correlation[active_indices[:, None], active_indices[None, :]] = active_correlation
    correlation = 0.5 * (correlation + correlation.transpose(0, 1))

    identity = torch.eye(action_dim, dtype=torch.float64)
    regularized_correlation = (1.0 - shrinkage) * correlation + shrinkage * identity
    chol_correlation = torch.linalg.cholesky(regularized_correlation)
    precision = torch.cholesky_solve(identity, chol_correlation)
    metric = float(action_dim) * precision / precision.trace()
    metric = 0.5 * (metric + metric.transpose(0, 1))
    metric_cholesky = torch.linalg.cholesky(metric)

    return {
        "count": int(count),
        "action_dim": int(action_dim),
        "mean": mean,
        "covariance": covariance,
        "std": std,
        "active": active,
        "correlation": correlation,
        "regularized_correlation": regularized_correlation,
        "precision": precision,
        "metric": metric,
        "metric_cholesky": metric_cholesky,
        "shrinkage": float(shrinkage),
        "std_epsilon": float(std_epsilon),
    }


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    train_cfg = load_train_config(args.data_config)
    dataset_cfg = OmegaConf.to_container(train_cfg, resolve=True)
    processor_cfg = dataset_cfg.pop("processor")
    dataset_cfg.pop("pretrained_norm_stats", None)
    dataset_cfg["processor"] = None

    dataset = instantiate(dataset_cfg)
    processor = instantiate(processor_cfg)
    processor.train()
    stats = dataset.lerobot_dataset.get_dataset_stats(processor)

    stats_path = Path(args.stats_output)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    save_dataset_stats_to_json(stats, str(stats_path))
    processor.set_normalizer_from_stats(stats)
    dataset.lerobot_dataset.set_processor(processor)

    raw_actions, manifest = collect_raw_actions(dataset.lerobot_dataset)
    normalized_actions = normalize_actions(raw_actions, processor)
    metric = build_metric(
        normalized_actions,
        std_epsilon=args.std_epsilon,
        shrinkage=args.shrinkage,
    )

    channel_order = []
    for meta in processor.shape_meta["action"]:
        for index in range(int(meta["shape"])):
            channel_order.append(f"{meta['key']}[{index}]")
    metric["channel_order"] = channel_order
    metric["normalizer_sha256"] = hashlib.sha256(stats_path.read_bytes()).hexdigest()
    metric["dataset_manifest"] = {
        "dataset_config": str(Path(args.data_config).resolve()),
        "datasets": manifest,
        "unique_index_rule": "one raw training frame index, current action only",
        "row_count": int(normalized_actions.shape[0]),
    }
    metric["git_commit"] = git_commit()

    metric_path = Path(args.metric_output)
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(metric, str(metric_path))

    print(
        json.dumps(
            {
                "stats": str(stats_path),
                "metric": str(metric_path),
                "count": metric["count"],
                "action_dim": metric["action_dim"],
                "active": metric["active"].tolist(),
                "metric_trace": float(metric["metric"].trace().item()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
