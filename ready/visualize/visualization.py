"""Spatial plantar-pressure heatmap reconstructed from 16 sensor readings.

The heatmap is computed from the same principle described in the manuscript:
each discrete pressure sensor contributes to nearby insole positions through a
Gaussian kernel,

    P(x, y) = sum_i p_i * exp(-((x - x_i)^2 + (y - y_i)^2) / (2 * sigma^2))

where ``p_i`` is the pressure value of sensor ``i`` and ``(x_i, y_i)`` is its
location in the insole coordinate system.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SENSOR_RENAME_MAP = {
    "R2-C2": "1",
    "R1-C2": "2",
    "R3-C2": "3",
    "R3-C3": "4",
    "R0-C2": "5",
    "R0-C3": "6",
    "R1-C1": "7",
    "R2-C1": "8",
    "R1-C3": "9",
    "R2-C3": "10",
    "R3-C1": "11",
    "R0-C1": "12",
    "R1-C0": "13",
    "R2-C0": "14",
    "R0-C0": "15",
    "R3-C0": "16",
}

SENSOR_ORDER = [str(i) for i in range(1, 17)]
DEFAULT_PRESSURE_COLUMNS = [f"F{i}" for i in range(1, 17)]

# Approximate normalized insole coordinates. y=0 is heel, y=1 is forefoot/toe.
SENSOR_COORDINATES = {
    "1": (-0.18, 0.86),
    "2": (-0.06, 0.92),
    "3": (0.08, 0.88),
    "4": (0.20, 0.82),
    "5": (-0.22, 0.66),
    "6": (-0.08, 0.70),
    "7": (0.06, 0.70),
    "8": (0.20, 0.64),
    "9": (-0.20, 0.46),
    "10": (-0.06, 0.48),
    "11": (0.08, 0.48),
    "12": (0.20, 0.44),
    "13": (-0.16, 0.22),
    "14": (-0.04, 0.18),
    "15": (0.08, 0.18),
    "16": (0.18, 0.24),
}


def _renamed_header(header: str) -> str:
    if header in SENSOR_RENAME_MAP:
        return SENSOR_RENAME_MAP[header]
    for raw_name, sensor_id in SENSOR_RENAME_MAP.items():
        if raw_name in header:
            return sensor_id
    return header


def select_pressure_matrix(table: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Rename hardware-style columns and reorder pressure sensors from 1 to 16."""
    renamed = {_renamed_header(col): col for col in table.columns}

    if all(col in table.columns for col in DEFAULT_PRESSURE_COLUMNS):
        columns = DEFAULT_PRESSURE_COLUMNS
        labels = SENSOR_ORDER
    else:
        missing = [sensor_id for sensor_id in SENSOR_ORDER if sensor_id not in renamed]
        if missing:
            raise ValueError(
                "Could not find a complete 16-channel pressure set. "
                f"Missing sensors: {missing}"
            )
        columns = [renamed[sensor_id] for sensor_id in SENSOR_ORDER]
        labels = SENSOR_ORDER
    return table[columns].to_numpy(dtype=float), labels


def summarize_pressure_frame(
    pressure_matrix: np.ndarray,
    frame: int | None,
    aggregate: str,
) -> np.ndarray:
    """Choose one frame or summarize a full pressure sequence."""
    if frame is not None:
        if frame < 0 or frame >= pressure_matrix.shape[0]:
            raise IndexError(f"frame must be in [0, {pressure_matrix.shape[0] - 1}]")
        return pressure_matrix[frame]

    if aggregate == "mean":
        return pressure_matrix.mean(axis=0)
    if aggregate == "max":
        return pressure_matrix.max(axis=0)
    if aggregate == "stance":
        total_pressure = pressure_matrix.sum(axis=1)
        return pressure_matrix[int(np.argmax(total_pressure))]
    raise ValueError("aggregate must be one of: mean, max, stance")


def create_pressure_colormap():
    from matplotlib.colors import LinearSegmentedColormap

    colors = [
        (0.0, "#5470C2"),
        (0.5, "#E0E0E0"),
        (1.0, "#D97B5E"),
    ]
    return LinearSegmentedColormap.from_list("pressure_map", colors)


def reconstruct_pressure_field(
    sensor_values: np.ndarray,
    sensor_labels: list[str],
    grid_size: int = 180,
    sigma: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct a continuous insole pressure field by Gaussian summation."""
    x = np.linspace(-0.34, 0.34, grid_size)
    y = np.linspace(0.0, 1.0, grid_size)
    xx, yy = np.meshgrid(x, y)
    field = np.zeros_like(xx, dtype=float)

    for value, label in zip(sensor_values, sensor_labels):
        sensor_x, sensor_y = SENSOR_COORDINATES[label]
        distance2 = (xx - sensor_x) ** 2 + (yy - sensor_y) ** 2
        field += value * np.exp(-distance2 / (2.0 * sigma**2))

    return xx, yy, field


def insole_mask(xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    """Approximate foot-shaped mask in normalized insole coordinates."""
    center = 0.03 * np.sin(2.0 * np.pi * yy)
    half_width = 0.12 + 0.12 * np.sin(np.pi * yy) + 0.07 * np.exp(-((yy - 0.88) / 0.20) ** 2)
    return np.abs(xx - center) <= half_width


def plot_pressure_heatmap(
    input_csv: Path,
    output_png: Path | None = None,
    frame: int | None = None,
    aggregate: str = "stance",
    grid_size: int = 180,
    sigma: float = 0.10,
    vmax: float | None = None,
) -> None:
    import matplotlib.pyplot as plt

    table = pd.read_csv(input_csv)
    pressure_matrix, sensor_labels = select_pressure_matrix(table)
    sensor_values = summarize_pressure_frame(pressure_matrix, frame, aggregate)
    xx, yy, pressure_field = reconstruct_pressure_field(
        sensor_values=sensor_values,
        sensor_labels=sensor_labels,
        grid_size=grid_size,
        sigma=sigma,
    )
    pressure_field = np.where(insole_mask(xx, yy), pressure_field, np.nan)

    fig, ax = plt.subplots(figsize=(4.8, 8.2))
    ax.imshow(
        pressure_field,
        extent=[xx.min(), xx.max(), yy.min(), yy.max()],
        origin="lower",
        cmap=create_pressure_colormap(),
        vmin=0,
        vmax=vmax,
        interpolation="bilinear",
    )
    ax.scatter(
        [SENSOR_COORDINATES[label][0] for label in sensor_labels],
        [SENSOR_COORDINATES[label][1] for label in sensor_labels],
        s=10,
        c="black",
        alpha=0.25,
        linewidths=0,
    )

    ax.set_xlim(-0.34, 0.34)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0)

    if output_png is None:
        plt.show()
    else:
        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png, dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        print(f"Saved pressure heatmap to {output_png}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_png", type=Path)
    parser.add_argument("--frame", type=int)
    parser.add_argument("--aggregate", choices=["mean", "max", "stance"], default="stance")
    parser.add_argument("--grid_size", type=int, default=180)
    parser.add_argument("--sigma", type=float, default=0.10)
    parser.add_argument("--vmax", type=float)
    args = parser.parse_args()
    plot_pressure_heatmap(
        input_csv=args.input_csv,
        output_png=args.output_png,
        frame=args.frame,
        aggregate=args.aggregate,
        grid_size=args.grid_size,
        sigma=args.sigma,
        vmax=args.vmax,
    )


if __name__ == "__main__":
    main()
