from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "train"))
from model import INPUT_COLUMNS, TARGET_COLUMNS, get_custom_objects  # noqa: E402


def predict(model_dir: Path, input_csv: Path, output_csv: Path) -> None:
    import tensorflow as tf

    model_path = model_dir / "model.keras"
    stats_path = model_dir / "normalization_stats.npz"
    stats = np.load(stats_path, allow_pickle=True)

    table = pd.read_csv(input_csv)
    missing = [name for name in INPUT_COLUMNS if name not in table.columns]
    if missing:
        raise ValueError(f"{input_csv.name} is missing columns: {missing}")

    x = table[INPUT_COLUMNS].to_numpy(dtype=np.float32)
    x_norm = (x - stats["input_mean"]) / stats["input_std"]
    x_norm = x_norm[None, :, :]

    net = tf.keras.models.load_model(
        model_path,
        custom_objects=get_custom_objects(),
    )
    y_norm = net.predict(x_norm, verbose=0)[0]
    y_pred = y_norm * stats["target_std"] + stats["target_mean"]

    result = table.copy()
    for i, name in enumerate(TARGET_COLUMNS):
        result[f"{name}_pred"] = y_pred[:, i]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    print(f"Saved predictions to {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    args = parser.parse_args()
    predict(args.model_dir, args.input_csv, args.output_csv)


if __name__ == "__main__":
    main()
