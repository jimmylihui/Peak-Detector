from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def read_split_info(path: str | Path) -> tuple[list[str], list[str]]:
    path = Path(path)
    if not path.exists():
        return [], []
    with path.open() as f:
        meta = json.load(f)
    return list(meta.get("train_records", [])), list(meta.get("test_records", []))


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)
    std = float(signal.std())
    if std < 1e-8:
        return signal - float(signal.mean())
    return (signal - float(signal.mean())) / std


def window_pair(signal: np.ndarray, labels: np.ndarray, window: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(signal).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    usable = min(signal.size, labels.size)
    usable = (usable // window) * window
    if usable == 0:
        return np.empty((0, window), dtype=np.float32), np.empty((0, window), dtype=np.float32)
    x = signal[:usable].reshape(-1, window).astype(np.float32)
    y = labels[:usable].reshape(-1, window).astype(np.float32)
    return x, y


def labels_from_peaks(length: int, peaks: np.ndarray, sigma: float = 0.0) -> np.ndarray:
    labels = np.zeros(length, dtype=np.float32)
    peaks = np.asarray(peaks, dtype=int).reshape(-1)
    peaks = peaks[(peaks >= 0) & (peaks < length)]
    if sigma <= 0:
        labels[peaks] = 1.0
        return labels

    radius = int(3 * sigma)
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-(offsets**2) / (2 * sigma * sigma)).astype(np.float32)
    for peak in peaks:
        lo = max(0, peak - radius)
        hi = min(length, peak + radius + 1)
        k_lo = lo - (peak - radius)
        k_hi = k_lo + (hi - lo)
        labels[lo:hi] = np.maximum(labels[lo:hi], kernel[k_lo:k_hi])
    return labels


def split_train_val(X: np.ndarray, y: np.ndarray, val_fraction: float = 0.2) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(X) == 0:
        return X, y, X.copy(), y.copy()
    split = int(round(len(X) * (1.0 - val_fraction)))
    split = min(max(split, 1), len(X))
    return X[:split], y[:split], X[split:], y[split:]


def concat_or_empty(items: list[np.ndarray], window: int = 1000) -> np.ndarray:
    if not items:
        return np.empty((0, window), dtype=np.float32)
    return np.concatenate(items, axis=0).astype(np.float32)


def save_splits(out_dir: str | Path, **arrays: np.ndarray) -> None:
    out_dir = ensure_dir(out_dir)
    for name, array in arrays.items():
        np.save(out_dir / f"{name}.npy", np.asarray(array))
        print(f"{name}: {np.asarray(array).shape}")


def copy_split_arrays(source_dir: str | Path, out_dir: str | Path) -> None:
    source_dir = Path(source_dir)
    out_dir = ensure_dir(out_dir)
    names = ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]
    copied = []
    for name in names:
        src = source_dir / f"{name}.npy"
        if src.exists():
            dst = out_dir / src.name
            shutil.copy2(src, dst)
            copied.append(name)
            print(f"{name}: {np.load(dst, mmap_mode='r').shape}")
    if not copied:
        raise FileNotFoundError(f"No split arrays found in {source_dir}")
