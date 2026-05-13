#!/usr/bin/env python3
"""Generate baseline-ready CapnoBase PPG arrays from raw .npy signal/peak pairs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))
from baseline_array_utils import concat_or_empty, labels_from_peaks, normalize_signal, save_splits, window_pair


NPY_DIR = Path("/path/to/workspace/project-BCG-LLM/PPG_peaks/dataset/capnobase/data/npy")
OUT_DIR = Path("/path/to/workspace/project-BCG-LLM/PPG_peaks/splitted_dataset/capnobase")
SPLIT_METADATA = Path(__file__).with_name("split_metadata.json")
WINDOW = 1000
FALLBACK_TEST_SUBJECTS = {"0023", "0031", "0104", "0125", "0147", "0148", "0309", "0311", "0332"}


def subject_id(path: Path) -> str:
    return path.name.split("_")[0]


def load_test_subjects() -> set[str]:
    if not SPLIT_METADATA.exists():
        return FALLBACK_TEST_SUBJECTS
    with SPLIT_METADATA.open() as f:
        meta = json.load(f)
    test = meta.get("subject_splits", {}).get("test", [])
    return {str(subject).split("_")[0] for subject in test}


def main() -> None:
    train_x, train_y, test_x, test_y = [], [], [], []
    train_y_gauss, test_y_gauss = [], []
    test_subjects = load_test_subjects()

    for ppg_file in sorted(NPY_DIR.glob("*_ppg.npy")):
        sid = subject_id(ppg_file)
        peaks_file = NPY_DIR / f"{sid}_peaks.npy"
        if not peaks_file.exists():
            continue

        ppg = normalize_signal(np.load(ppg_file).reshape(-1))
        peaks_or_mask = np.load(peaks_file).reshape(-1)
        if peaks_or_mask.size == ppg.size:
            labels = (peaks_or_mask > 0).astype(np.float32)
            peak_positions = np.flatnonzero(labels)
        else:
            peak_positions = peaks_or_mask.astype(int)
            labels = labels_from_peaks(ppg.size, peak_positions, sigma=0.0)
        labels_gauss = labels_from_peaks(ppg.size, peak_positions, sigma=5.0)

        x, y = window_pair(ppg, labels, WINDOW)
        _, y_gauss = window_pair(ppg, labels_gauss, WINDOW)
        if sid in test_subjects:
            test_x.append(x)
            test_y.append(y)
            test_y_gauss.append(y_gauss)
        else:
            train_x.append(x)
            train_y.append(y)
            train_y_gauss.append(y_gauss)

    save_splits(
        OUT_DIR,
        X_train=concat_or_empty(train_x, WINDOW),
        y_train=concat_or_empty(train_y, WINDOW),
        X_test=concat_or_empty(test_x, WINDOW),
        y_test=concat_or_empty(test_y, WINDOW),
        y_train_gauss=concat_or_empty(train_y_gauss, WINDOW),
        y_test_gauss=concat_or_empty(test_y_gauss, WINDOW),
    )


if __name__ == "__main__":
    main()
