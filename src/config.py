from dataclasses import dataclass
from pathlib import Path
from typing import Literal


_REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class CFG:
    ml100k_path: str
    sep: str
    snapshot_gran: str
    window_sids: int

    node_dim: int
    embed_dim: int
    compressed_dim: int

    n_layers: int
    dropout: float

    lr: float
    weight_decay: float
    epochs: int

    train_ratio: float
    val_ratio: float
    test_ratio: float

    k: int

    seed: int
    device: str

    debug_sids: int
    full_train_epochs: int

    distillation_weight: float
    grad_clip: float

    use_scheduler: bool
    use_grad_checkpointing: bool

    project: str
    run_name: str
    checkpoint_dir: str
    use_wandb: bool = True
    checkpoint_filename: str = "best-{epoch}-{val_ndcg_small:.4f}"

    training_objective: Literal[
        "internal_distillation",
        "student_ce",
        "external_distillation",
    ] = "internal_distillation"
    teacher_checkpoint_path: str = ""
    teacher_config_path: str = ""
    kd_weight: float = 0.0
    kd_temperature: float = 1.0
    kd_teacher_branch: Literal["big", "small"] = "big"
    kd_student_branch: Literal["big", "small"] = "small"


def _parse_scalar(value: str):
    value = value.strip()
    if value == "":
        return ""
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]

    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        return value


def _load_flat_yaml(path: Path) -> dict:
    data = {}
    with open(path, "r") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if ":" not in stripped:
                raise ValueError(
                    f"Invalid config line {line_no} in {path}: {line.rstrip()}"
                )

            key, value = stripped.split(":", 1)
            data[key.strip()] = _parse_scalar(value)
    return data


def _candidate_roots(config_dir: Path | None = None) -> list[Path]:
    roots = [Path.cwd(), _REPO_ROOT]
    if config_dir is not None:
        roots.append(config_dir)
        roots.append(config_dir.parent)
        roots.append(config_dir.parent.parent)

    unique_roots = []
    for root in roots:
        root = root.resolve()
        if root not in unique_roots:
            unique_roots.append(root)
    return unique_roots


def _resolve_existing_path(path: str, config_dir: Path | None = None) -> Path:
    path_obj = Path(path).expanduser()
    candidates = [path_obj] if path_obj.is_absolute() else []
    if not path_obj.is_absolute():
        candidates.extend(root / path_obj for root in _candidate_roots(config_dir))

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"File not found: {path}. Searched: {searched}")


def _resolve_if_exists(path: str, config_dir: Path | None = None) -> str:
    if not path:
        return path

    try:
        return str(_resolve_existing_path(path, config_dir))
    except FileNotFoundError:
        return path


def load_config(path: str) -> CFG:
    config_path = _resolve_existing_path(path)
    data = _load_flat_yaml(config_path)

    config_dir = config_path.parent
    data["ml100k_path"] = _resolve_if_exists(data["ml100k_path"], config_dir)
    data["teacher_config_path"] = _resolve_if_exists(
        data.get("teacher_config_path", ""),
        config_dir,
    )
    data["teacher_checkpoint_path"] = _resolve_if_exists(
        data.get("teacher_checkpoint_path", ""),
        config_dir,
    )

    return CFG(**data)
