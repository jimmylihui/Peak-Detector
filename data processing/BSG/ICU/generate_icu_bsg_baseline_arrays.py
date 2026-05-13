#!/usr/bin/env python3
"""Generate baseline-ready ICU BSG arrays from split ICU .npy files."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

sys.path.append(str(Path(__file__).resolve().parents[2]))
from baseline_array_utils import labels_from_peaks, normalize_signal, save_splits


DATA_DIR = Path("/path/to/workspace/project-BCG-LLM/ICU_3d_hr/benchmark/dataset/split_icu_dataset")
OUT_DIR = Path("/path/to/workspace/project-BCG-LLM/ICU_3d_hr/benchmark/dataset/baseline_arrays")
WINDOW = 1000
FS = 100


def load_split(name: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(DATA_DIR / f"{name}_data.npy", mmap_mode="r")
    X, y = [], []
    for sample in data:
        bsg = normalize_signal(sample[:WINDOW])
        ecg = normalize_signal(sample[3000:4000])
        if bsg.size != WINDOW or ecg.size != WINDOW:
            continue
        peaks, _ = find_peaks(ecg, distance=int(0.35 * FS), prominence=0.5)
        if len(peaks) < 2:
            continue
        X.append(bsg)
        y.append(labels_from_peaks(WINDOW, peaks, sigma=5.0))
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def main() -> None:
    X_train, y_train = load_split("train")
    X_test, y_test = load_split("test")
    save_splits(OUT_DIR, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)


if __name__ == "__main__":
    main()
