"""Pseudo-label adaptation for target-domain insole sequences."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "train"))
from model import INPUT_COLUMNS, get_custom_objects, pad_sequences  # noqa: E402


def _read_unlabeled_sequences(data_dir: Path) -> tuple[list[np.ndarray], list[str]]:
    sequences: list[np.ndarray] = []
    names: list[str] = []
    for csv_path in sorted(data_dir.glob("*.csv")):
        table = pd.read_csv(csv_path)
        if all(name in table.columns for name in INPUT_COLUMNS):
            sequences.append(table[INPUT_COLUMNS].to_numpy(dtype=np.float32))
            names.append(csv_path.name)
    if not sequences:
        raise FileNotFoundError(f"No valid target CSV files found in {data_dir}")
    return sequences, names


def generate_pseudo_labels(
    model_dir: Path,
    target_data_dir: Path,
    output_npz: Path,
    mc_samples: int = 20,
    uncertainty_threshold: float = 0.35,
) -> None:
    """Generate high-confidence pseudo labels with MC-dropout inference."""
    import tensorflow as tf

    model_path = model_dir / "model.keras"
    stats_path = model_dir / "normalization_stats.npz"
    stats = np.load(stats_path, allow_pickle=True)
    net = tf.keras.models.load_model(
        model_path,
        custom_objects=get_custom_objects(),
    )

    sequences, names = _read_unlabeled_sequences(target_data_dir)
    accepted_x = []
    accepted_y = []
    accepted_names = []

    for sequence, name in zip(sequences, names):
        x_norm = (sequence - stats["input_mean"]) / stats["input_std"]
        x_in = x_norm[None, :, :]

        predictions = []
        for _ in range(mc_samples):
            predictions.append(net(x_in, training=True).numpy()[0])
        stacked = np.stack(predictions, axis=0)
        mean_pred = stacked.mean(axis=0)
        uncertainty = float(stacked.std(axis=0).mean())

        if uncertainty <= uncertainty_threshold:
            accepted_x.append(x_norm.astype(np.float32))
            accepted_y.append(mean_pred.astype(np.float32))
            accepted_names.append(name)
            print(f"Accepted pseudo label: {name}, uncertainty={uncertainty:.4f}")
        else:
            print(f"Skipped target file: {name}, uncertainty={uncertainty:.4f}")

    if not accepted_x:
        raise RuntimeError("No target sequence passed the uncertainty threshold.")

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_npz,
        x=np.array(accepted_x, dtype=object),
        y=np.array(accepted_y, dtype=object),
        files=np.array(accepted_names, dtype=object),
    )
    print(f"Saved pseudo labels to {output_npz}")


def fine_tune_with_pseudo_labels(
    model_dir: Path,
    pseudo_npz: Path,
    output_dir: Path,
    epochs: int = 50,
    batch_size: int = 4,
) -> None:
    """Fine-tune the trained model using accepted pseudo-labeled sequences."""
    import tensorflow as tf

    pseudo = np.load(pseudo_npz, allow_pickle=True)
    x_list = [np.asarray(x, dtype=np.float32) for x in pseudo["x"]]
    y_list = [np.asarray(y, dtype=np.float32) for y in pseudo["y"]]

    x_padded = pad_sequences(x_list)
    y_padded = pad_sequences(y_list)

    net = tf.keras.models.load_model(
        model_dir / "model.keras",
        custom_objects=get_custom_objects(),
    )
    net.fit(x_padded, y_padded, epochs=epochs, batch_size=batch_size, verbose=1)

    output_dir.mkdir(parents=True, exist_ok=True)
    net.save(output_dir / "model.keras")
    (output_dir / "normalization_stats.npz").write_bytes(
        (model_dir / "normalization_stats.npz").read_bytes()
    )
    print(f"Saved adapted model to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    make = subparsers.add_parser("generate")
    make.add_argument("--model_dir", type=Path, required=True)
    make.add_argument("--target_data_dir", type=Path, required=True)
    make.add_argument("--output_npz", type=Path, required=True)
    make.add_argument("--mc_samples", type=int, default=20)
    make.add_argument("--uncertainty_threshold", type=float, default=0.35)

    tune = subparsers.add_parser("finetune")
    tune.add_argument("--model_dir", type=Path, required=True)
    tune.add_argument("--pseudo_npz", type=Path, required=True)
    tune.add_argument("--output_dir", type=Path, required=True)
    tune.add_argument("--epochs", type=int, default=50)
    tune.add_argument("--batch_size", type=int, default=4)

    args = parser.parse_args()
    if args.command == "generate":
        generate_pseudo_labels(
            model_dir=args.model_dir,
            target_data_dir=args.target_data_dir,
            output_npz=args.output_npz,
            mc_samples=args.mc_samples,
            uncertainty_threshold=args.uncertainty_threshold,
        )
    else:
        fine_tune_with_pseudo_labels(
            model_dir=args.model_dir,
            pseudo_npz=args.pseudo_npz,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
