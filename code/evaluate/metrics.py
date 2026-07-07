from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_COLUMNS = ["Fx", "Fy", "Fz", "Mx", "My", "Mz", "COPx", "COPy"]


def summarize_metrics(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in TARGET_COLUMNS:
        pred_name = f"{name}_pred"
        if name not in table.columns or pred_name not in table.columns:
            continue

        y_true = table[name].to_numpy(dtype=float)
        y_pred = table[pred_name].to_numpy(dtype=float)
        err = y_pred - y_true
        rmse = float(np.sqrt(np.mean(err**2)))
        mae = float(np.mean(np.abs(err)))
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])
        rows.append({"variable": name, "rmse": rmse, "mae": mae, "r": corr})

    if not rows:
        raise ValueError("No target/prediction column pairs were found.")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    args = parser.parse_args()

    table = pd.read_csv(args.csv)
    metrics = summarize_metrics(table)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_csv, index=False)
    print(metrics.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
