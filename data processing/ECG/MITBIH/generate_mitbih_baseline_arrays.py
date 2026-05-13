#!/usr/bin/env python3
"""Generate baseline-ready MIT-BIH arrays from processed record folders."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))
from baseline_array_utils import concat_or_empty, read_split_info, save_splits, split_train_val, window_pair


PROCESSED_DIR = Path("/path/to/workspace/project-BCG-LLM/ECG_peak/data_process/processed_records")
SPLIT_INFO = Path(__file__).with_name("split_info.json")
OUT_DIR = Path("/path/to/workspace/project-BCG-LLM/ECG_peak/benchmark/MITBIH/baseline_arrays")
WINDOW = 1000


def load_record(record_id: str) -> tuple[np.ndarray, np.ndarray]:
    record_dir = PROCESSED_DIR / f"processed_{record_id}"
    signal_files = sorted(record_dir.glob(f"{record_id}_*_series.npy"))
    label_file = record_dir / f"{record_id}_binary_labels.npy"
    if not signal_files or not label_file.exists():
        raise FileNotFoundError(f"Missing processed signal or labels for record {record_id}")
    signal = np.load(signal_files[0])
    labels = (np.load(label_file) > 0).astype(np.float32)
    return window_pair(signal, labels, WINDOW)


def collect(record_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for record_id in record_ids:
        try:
            x, y = load_record(record_id)
        except FileNotFoundError as exc:
            print(f"Skipping {record_id}: {exc}")
            continue
        xs.append(x)
        ys.append(y)
    return concat_or_empty(xs, WINDOW), concat_or_empty(ys, WINDOW)


def main() -> None:
    train_records, test_records = read_split_info(SPLIT_INFO)
    if not train_records:
        train_records = sorted(p.name.removeprefix("processed_") for p in PROCESSED_DIR.glob("processed_*"))

    X_train_all, y_train_all = collect(train_records)
    X_train, y_train, X_val, y_val = split_train_val(X_train_all, y_train_all)
    X_test, y_test = collect(test_records)

    save_splits(
        OUT_DIR,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )


if __name__ == "__main__":
    main()
