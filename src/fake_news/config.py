from __future__ import annotations



from dataclasses import dataclass
from pathlib import Path


@dataclass
class PathsConfig:
    project_root: Path
    raw_data_dir: Path
    processed_data_dir: Path
    model_dir: Path
    report_dir: Path


@dataclass
class TrainingConfig:
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_len: int = 256
    hidden_dim: int = 128
    embed_dim: int = 100  # Changed from 200 to match GloVe 6B size
    num_layers: int = 1
    dropout: float = 0.2
    seed: int = 42


def default_paths(project_root: Path) -> PathsConfig:
    return PathsConfig(
        project_root=project_root,
        raw_data_dir=project_root / "data" / "raw",
        processed_data_dir=project_root / "data" / "processed",
        model_dir=project_root / "models",
        report_dir=project_root / "reports",
    )

