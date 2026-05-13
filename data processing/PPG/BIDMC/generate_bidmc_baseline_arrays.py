#!/usr/bin/env python3
"""Generate baseline-ready BIDMC PPG arrays from processed subject folders."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

sys.path.append(str(Path(__file__).resolve().parents[2]))
from baseline_array_utils import concat_or_empty, labels_from_peaks, normalize_signal, save_splits, split_train_val


PROCESSED_DIR = Path("/path/to/workspace/project-BCG-LLM/PPG_peaks/processed_dataset/BIDMC")
OUT_DIR = Path("/path/to/workspace/project-BCG-LLM/PPG_peaks/Benchmark/unet++/bidmc_baseline_arrays")
SPLIT_INFO = Path(__file__).with_name("split_info.json")
FALLBACK_TEST_SUBJECTS = {"04", "06", "13", "14", "18", "20", "33", "42", "44", "48", "51"}
FS = 125


def labels_from_ecg_segment(ecg: np.ndarray) -> np.ndarray:
    ecg = normalize_signal(ecg)
    peaks, _ = find_peaks(ecg, distance=int(0.35 * FS), prominence=0.5)
    return labels_from_peaks(len(ecg), peaks, sigma=3.0)


def load_subject(subject_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    ppg = np.load(subject_dir / "ppg_segments.npy")
    ecg = np.load(subject_dir / "ecg_segments.npy")
    resp_file = subject_dir / "resp_segments.npy"
    resp = np.load(resp_file) if resp_file.exists() else None
    X = np.asarray([normalize_signal(segment) for segment in ppg], dtype=np.float32)
    y = np.asarray([labels_from_ecg_segment(segment) for segment in ecg], dtype=np.float32)
    return X, y, resp


def load_test_subjects() -> set[str]:
    if not SPLIT_INFO.exists():
        return FALLBACK_TEST_SUBJECTS
    with SPLIT_INFO.open() as f:
        split = json.load(f)
    return {str(subject).zfill(2) for subject in split.get("test_subjects", [])}


def main() -> None:
    train_x, train_y, test_x, test_y = [], [], [], []
    train_resp, test_resp = [], []
    test_subjects = load_test_subjects()

    for subject_dir in sorted(PROCESSED_DIR.glob("subject_*")):
        subject_id = subject_dir.name.removeprefix("subject_")
        X, y, resp = load_subject(subject_dir)
        if subject_id in test_subjects:
            test_x.append(X)
            test_y.append(y)
            if resp is not None:
                test_resp.append(resp)
        else:
            train_x.append(X)
            train_y.append(y)
            if resp is not None:
                train_resp.append(resp)

    X_train_all = concat_or_empty(train_x)
    y_train_all = concat_or_empty(train_y)
    X_train, y_train, X_val, y_val = split_train_val(X_train_all, y_train_all)

    arrays = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": concat_or_empty(test_x),
        "y_test": concat_or_empty(test_y),
    }
    if train_resp:
        resp_train, _, resp_val, _ = split_train_val(concat_or_empty(train_resp), concat_or_empty(train_resp))
        arrays["X_train_resp"] = resp_train
        arrays["X_val_resp"] = resp_val
    if test_resp:
        arrays["X_test_resp"] = concat_or_empty(test_resp)

    save_splits(OUT_DIR, **arrays)


if __name__ == "__main__":
    main()
