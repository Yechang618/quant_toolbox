"""
Model Inference Pipeline.

Skeleton for loading a trained model and running batch or online inference.
Replace / extend placeholder sections with your actual model-loading and
prediction logic.

WARNING: Never hard-code API keys or credentials here.
         Use environment variables via util/config.py.
"""

import logging
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd

from util.config import settings
from util.helpers import Timer, read_parquet
from util.logger import get_logger

logger: logging.Logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(path: Path) -> Any:
    """Load a trained model from disk.

    Args:
        path: Path to the saved model file (e.g. ``.pth``, ``.h5``, ``.onnx``).

    Returns:
        Loaded model object.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    # TODO: Replace with framework-specific load logic:
    #   - PyTorch:    torch.load(path)
    #   - scikit-learn / joblib:  joblib.load(path)
    #   - ONNX Runtime:  ort.InferenceSession(str(path))
    logger.info("Model loaded from %s (placeholder)", path)
    return {"path": str(path), "type": "placeholder"}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_features(df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> np.ndarray:
    """Extract a NumPy feature matrix from a processed DataFrame.

    Args:
        df: DataFrame with feature columns (output of the preprocessing
            pipeline).
        feature_cols: Columns to use as features.  If *None*, all numeric
            columns except ``timestamp``, ``bids``, ``asks``, and
            ``mid_price`` are used.

    Returns:
        2-D NumPy array of shape ``(n_samples, n_features)``.
    """
    if feature_cols is None:
        exclude = {"timestamp", "bids", "asks", "mid_price"}
        feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    logger.debug("Using feature columns: %s", feature_cols)
    return df[feature_cols].to_numpy(dtype=np.float32)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def predict_batch(
    model: Any,
    features: np.ndarray,
    batch_size: int = settings.BATCH_SIZE,
) -> np.ndarray:
    """Run batch inference and return raw predictions.

    Args:
        model: Loaded model object.
        features: Feature matrix ``(n_samples, n_features)``.
        batch_size: Number of samples to process per forward pass.

    Returns:
        1-D NumPy array of predictions ``(n_samples,)``.
    """
    n_samples = features.shape[0]
    predictions: List[np.ndarray] = []

    with Timer() as timer:
        for start in range(0, n_samples, batch_size):
            batch = features[start : start + batch_size]
            # TODO: Replace with real model forward pass, e.g.:
            #   preds = model(torch.tensor(batch)).detach().numpy()
            preds = np.full(len(batch), fill_value=float("nan"))  # placeholder
            predictions.append(preds)

    logger.info("Predicted %d samples in %.3fs", n_samples, timer.elapsed)
    return np.concatenate(predictions)


def predict_single(
    model: Any,
    sample: Union[np.ndarray, List[float]],
) -> float:
    """Run inference on a single feature vector.

    Args:
        model: Loaded model object.
        sample: 1-D feature vector of length ``n_features``.

    Returns:
        Scalar prediction value.
    """
    arr = np.asarray(sample, dtype=np.float32).reshape(1, -1)
    result = predict_batch(model, arr, batch_size=1)
    return float(result[0])


# ---------------------------------------------------------------------------
# End-to-end inference pipeline
# ---------------------------------------------------------------------------


def run_inference(
    model_path: Path,
    data_path: Path,
    output_path: Optional[Path] = None,
    feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Full inference pipeline: load model → load data → predict → save.

    Args:
        model_path: Path to the saved model file.
        data_path: Path to the processed Parquet dataset.
        output_path: Optional path to write predictions CSV.  If *None*,
            predictions are returned but not persisted.
        feature_cols: Feature columns to pass to :func:`extract_features`.

    Returns:
        DataFrame with original data plus a ``prediction`` column.
    """
    model = load_model(model_path)
    df = read_parquet(data_path)

    features = extract_features(df, feature_cols=feature_cols)
    df["prediction"] = predict_batch(model, features)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Predictions saved to %s", output_path)

    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run order-book model inference")
    parser.add_argument("--model-path", type=Path, default=settings.MODEL_DIR / "model.pth")
    parser.add_argument("--data-path", type=Path, default=settings.DATA_PROCESSED_DIR / "dataset.parquet")
    parser.add_argument("--output-path", type=Path, default=None)
    args = parser.parse_args()

    result_df = run_inference(
        model_path=args.model_path,
        data_path=args.data_path,
        output_path=args.output_path,
    )
    print(result_df[["prediction"]].describe())
