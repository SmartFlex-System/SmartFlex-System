"""Topology-enhanced LSTM model and training workflow.

This public model file keeps the same overall process as the manuscript training
code: read CSV sequences, standardize inputs and outputs, pad variable-length
trials, train a pressure-graph encoder plus BiLSTM, and save the model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

IMU_COLUMNS = ["aX", "aY", "aZ", "Gx", "Gy", "Gz"]
PRESSURE_COLUMNS = [f"F{i}" for i in range(1, 17)]
INPUT_COLUMNS = IMU_COLUMNS + PRESSURE_COLUMNS
TARGET_COLUMNS = ["Fx", "Fy", "Fz", "Mx", "My", "Mz", "COPx", "COPy"]


def plantar_adjacency() -> np.ndarray:
    """Manual 16-sensor adjacency matrix used for pressure topology encoding."""
    adjacency = np.zeros((16, 16), dtype=np.float32)
    connections = [
        (12, 8), (12, 0), (8, 0), (4, 8), (1, 4), (1, 13), (7, 13),
        (7, 1), (7, 4), (7, 11), (11, 4), (11, 8), (11, 0),
        (5, 7), (5, 9), (9, 11), (3, 5), (3, 15), (15, 2),
        (15, 14), (15, 10), (14, 6), (6, 10), (6, 2),
        (2, 14), (14, 10), (2, 3), (3, 9),
    ]
    for i, j in connections:
        adjacency[i, j] = 1.0
        adjacency[j, i] = 1.0
    np.fill_diagonal(adjacency, 1.0)
    return adjacency


def _require_tensorflow():
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models, optimizers
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TensorFlow is required for model training. Install it with: "
            "pip install tensorflow"
        ) from exc
    return tf, layers, models, optimizers


def node_pool_mean(t):
    tf, _, _, _ = _require_tensorflow()
    return tf.reduce_mean(t, axis=2)


def take_imu(t):
    return t[:, :, :6]


def take_pressure(t):
    return t[:, :, 6:]


def expand_pressure_feature_dim(t):
    tf, _, _, _ = _require_tensorflow()
    return tf.expand_dims(t, -1)


def get_custom_objects() -> dict:
    return {
        "GraphConvLayer": GraphConvLayer,
        "node_pool_mean": node_pool_mean,
        "take_imu": take_imu,
        "take_pressure": take_pressure,
        "expand_pressure_feature_dim": expand_pressure_feature_dim,
    }


class GraphConvLayer(_require_tensorflow()[1].Layer):
    """Graph convolution over the 16 plantar-pressure sensor nodes."""

    def __init__(self, units: int, activation: str = "relu", **kwargs):
        super().__init__(**kwargs)
        _, layers, _, _ = _require_tensorflow()
        self.units = units
        self.act = layers.Activation(activation)

    def build(self, input_shape):
        feature_dim = int(input_shape[-1])
        self.weight = self.add_weight(
            shape=(feature_dim, self.units),
            initializer="glorot_uniform",
            name="weight",
        )
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            name="bias",
        )

    def call(self, x, adjacency):
        tf, _, _, _ = _require_tensorflow()
        batch = tf.shape(x)[0]
        steps = tf.shape(x)[1]
        nodes = tf.shape(x)[2]
        features = tf.shape(x)[3]
        x_bt = tf.reshape(x, (batch * steps, nodes, features))
        ax = tf.matmul(adjacency, x_bt)
        axw = tf.tensordot(ax, self.weight, axes=[[2], [0]]) + self.bias
        y = tf.reshape(axw, (batch, steps, nodes, self.units))
        return self.act(y)


def build_model(
    lstm_units: int = 64,
    graph_units: int = 32,
    dropout: float = 0.2,
    learning_rate: float = 1e-4,
):
    """Build the pressure-topology + BiLSTM reconstruction model."""
    tf, layers, models, optimizers = _require_tensorflow()
    adjacency = tf.constant(plantar_adjacency(), dtype=tf.float32)

    inputs = layers.Input(shape=(None, len(INPUT_COLUMNS)), name="all_features")
    imu = layers.Lambda(take_imu, name="take_imu")(inputs)
    pressure = layers.Lambda(take_pressure, name="take_pressure")(inputs)
    pressure = layers.Lambda(
        expand_pressure_feature_dim,
        name="expand_pressure_feature_dim",
    )(pressure)

    pressure = GraphConvLayer(graph_units, activation="relu", name="graph_conv_1")(
        pressure,
        adjacency,
    )
    pressure = layers.Dropout(dropout, name="graph_dropout")(pressure)
    pressure = GraphConvLayer(graph_units, activation="relu", name="graph_conv_2")(
        pressure,
        adjacency,
    )
    pressure = layers.Lambda(node_pool_mean, name="node_pool_mean")(pressure)

    fused = layers.Concatenate(name="imu_pressure_fusion")([pressure, imu])
    sequence = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True),
        name="bilstm",
    )(fused)
    sequence = layers.Dropout(dropout, name="sequence_dropout")(sequence)
    outputs = layers.TimeDistributed(
        layers.Dense(len(TARGET_COLUMNS)),
        name="kinetics_output",
    )(sequence)

    model = models.Model(inputs=inputs, outputs=outputs, name="TopologyEnhancedLSTM")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def read_sequences(data_dir: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    for csv_path in csv_files:
        table = pd.read_csv(csv_path)
        required = INPUT_COLUMNS + TARGET_COLUMNS
        missing = [name for name in required if name not in table.columns]
        if missing:
            raise ValueError(f"{csv_path.name} is missing columns: {missing}")
        x_list.append(table[INPUT_COLUMNS].to_numpy(dtype=np.float32))
        y_list.append(table[TARGET_COLUMNS].to_numpy(dtype=np.float32))
    return x_list, y_list


def standardize_sequences(
    sequences: list[np.ndarray],
) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
    stacked = np.vstack(sequences)
    mean = stacked.mean(axis=0, keepdims=True)
    std = stacked.std(axis=0, keepdims=True) + 1e-8
    return [(seq - mean) / std for seq in sequences], {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
    }


def pad_sequences(sequences: list[np.ndarray]) -> np.ndarray:
    max_len = max(len(seq) for seq in sequences)
    feature_dim = sequences[0].shape[1]
    padded = np.zeros((len(sequences), max_len, feature_dim), dtype=np.float32)
    for i, seq in enumerate(sequences):
        padded[i, : len(seq), :] = seq
    return padded


def train(
    data_dir: Path,
    output_dir: Path,
    epochs: int = 150,
    batch_size: int = 4,
    validation_split: float = 0.2,
) -> None:
    x_raw, y_raw = read_sequences(data_dir)
    x_norm, input_stats = standardize_sequences(x_raw)
    y_norm, target_stats = standardize_sequences(y_raw)

    x_padded = pad_sequences(x_norm)
    y_padded = pad_sequences(y_norm)

    model = build_model()
    model.summary()
    model.fit(
        x_padded,
        y_padded,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "model.keras")
    np.savez(
        output_dir / "normalization_stats.npz",
        input_mean=input_stats["mean"],
        input_std=input_stats["std"],
        target_mean=target_stats["mean"],
        target_std=target_stats["std"],
        input_columns=np.array(INPUT_COLUMNS),
        target_columns=np.array(TARGET_COLUMNS),
    )
    print(f"Saved model and normalization statistics to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("models/topology_lstm"))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--validation_split", type=float, default=0.2)
    args = parser.parse_args()
    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
    )


if __name__ == "__main__":
    main()
