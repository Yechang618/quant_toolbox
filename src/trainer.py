"""
Model Training Pipeline.

Skeleton for training an order-book prediction model.
Replace / extend the placeholder sections with your actual model, data
loading, and evaluation logic.

WARNING: Never hard-code API keys or credentials here.
         Use environment variables via util/config.py.
"""

import logging
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from util.config import settings
from util.helpers import Timer, ensure_dir, format_duration, read_parquet
from util.logger import get_logger

logger: logging.Logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int = settings.RANDOM_SEED) -> None:
    """Seed Python, NumPy (and optionally PyTorch/TensorFlow) for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass  # PyTorch is optional
    logger.debug("Random seed set to %d", seed)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_dataset(
    data_path: Path,
    target_col: str = "mid_price",
    test_frac: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and split a processed Parquet dataset into train / test sets.

    Args:
        data_path: Path to the processed ``.parquet`` file.
        target_col: Name of the column used as the prediction target.
        test_frac: Fraction of data to reserve for testing (chronological split).

    Returns:
        Tuple ``(train_df, test_df)``.

    Raises:
        FileNotFoundError: If *data_path* does not exist.
        KeyError: If *target_col* is not present in the dataset.
    """
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = read_parquet(data_path)

    if target_col not in df.columns:
        raise KeyError(f"Target column {target_col!r} not found in {list(df.columns)}")

    split_idx = int(len(df) * (1 - test_frac))
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    logger.info("Dataset split: train=%d, test=%d rows", len(train_df), len(test_df))
    return train_df, test_df


# ---------------------------------------------------------------------------
# Model placeholder
# ---------------------------------------------------------------------------


def build_model(input_dim: int, **kwargs: Any) -> Any:
    """Build (or instantiate) the prediction model.

    Replace this stub with your actual model construction code, e.g. a
    PyTorch ``nn.Module``, a scikit-learn pipeline, or a Keras model.

    Args:
        input_dim: Number of input features.
        **kwargs: Additional model hyper-parameters.

    Returns:
        An untrained model object.
    """
    logger.info("Building model with input_dim=%d", input_dim)
    # TODO: Replace with real model implementation
    return {"type": "placeholder", "input_dim": input_dim, "params": kwargs}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(
    model: Any,
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    target_col: str = "mid_price",
    num_epochs: int = settings.NUM_EPOCHS,
    batch_size: int = settings.BATCH_SIZE,
    learning_rate: float = settings.LEARNING_RATE,
) -> Dict[str, Any]:
    """Run the main training loop.

    Args:
        model: Model object returned by :func:`build_model`.
        train_df: Training split DataFrame.
        val_df: Optional validation split DataFrame.
        target_col: Name of the target column.
        num_epochs: Number of training epochs.
        batch_size: Mini-batch size.
        learning_rate: Initial learning rate.

    Returns:
        Dictionary of training metrics (``train_loss``, ``val_loss``, etc.).
    """
    logger.info(
        "Starting training: epochs=%d, batch_size=%d, lr=%.5f",
        num_epochs,
        batch_size,
        learning_rate,
    )

    history: Dict[str, Any] = {"train_loss": [], "val_loss": []}

    with Timer() as timer:
        for epoch in range(1, num_epochs + 1):
            # ------------------------------------------------------------------
            # TODO: Replace with real forward pass, loss computation,
            #       back-propagation, and optimizer step.
            # ------------------------------------------------------------------
            train_loss = float("nan")  # placeholder
            val_loss = float("nan") if val_df is not None else None

            history["train_loss"].append(train_loss)
            if val_loss is not None:
                history["val_loss"].append(val_loss)

            if epoch % max(1, num_epochs // 10) == 0 or epoch == num_epochs:
                msg = f"Epoch {epoch}/{num_epochs}  train_loss={train_loss:.4f}"
                if val_loss is not None:
                    msg += f"  val_loss={val_loss:.4f}"
                logger.info(msg)

    logger.info("Training finished in %s", format_duration(timer.elapsed))
    return history


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def save_model(model: Any, path: Path) -> None:
    """Persist model weights / parameters to disk.

    Args:
        model: Trained model object.
        path: Destination file path (e.g. ``model/best.pth``).
    """
    path = Path(path)
    ensure_dir(path.parent)
    # TODO: Replace with framework-specific save logic (torch.save, joblib, …)
    logger.info("Model saved to %s (placeholder)", path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(
    data_path: Optional[Path] = None,
    model_out: Optional[Path] = None,
) -> None:
    """Main training entry point.

    Args:
        data_path: Path to the processed Parquet dataset.
        model_out: Path to save the trained model weights.
    """
    set_seed()

    data_path = data_path or settings.DATA_PROCESSED_DIR / "dataset.parquet"
    model_out = model_out or settings.MODEL_DIR / "model.pth"

    train_df, test_df = load_dataset(data_path)

    feature_cols = [c for c in train_df.columns if c not in ("timestamp", "mid_price", "bids", "asks")]
    input_dim = len(feature_cols)

    model = build_model(input_dim=input_dim)
    history = train(model, train_df, val_df=test_df)
    save_model(model, model_out)

    logger.info("Final training history keys: %s", list(history.keys()))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train order-book prediction model")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--model-out", type=Path, default=None)
    args = parser.parse_args()

    main(data_path=args.data_path, model_out=args.model_out)
