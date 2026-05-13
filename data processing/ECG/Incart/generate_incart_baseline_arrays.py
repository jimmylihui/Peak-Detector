#!/usr/bin/env python3
"""Generate baseline-ready INCART arrays directly from WFDB records."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import wfdb

sys.path.append(str(Path(__file__).resolve().parents[2]))
from baseline_array_utils import (
    concat_or_empty,
    labels_from_peaks,
    normalize_signal,
    read_split_info,
    save_splits,
    split_train_val,
    window_pair,
)


DATA_DIR = Path("/path/to/workspace/project-BCG-LLM/ECG_peak/dataset/incart/files")
SPLIT_INFO = Path("/path/to/workspace/project-BCG-LLM/ECG_peak/data_formatter/formatted_incart_dataset/split_info.json")
OUT_DIR = Path("/path/to/workspace/project-BCG-LLM/ECG_peak/benchmark/incart/baseline_arrays")
WINDOW = 1000
BEAT_SYMBOLS = set("NLRBAaJSEVFrFfejn/Q?x")


def available_records() -> list[str]:
    return sorted(path.stem for path in DATA_DIR.glob("*.hea"))


def load_record(record_id: str) -> tuple[np.ndarray, np.ndarray]:
    base = str(DATA_DIR / record_id)
    record = wfdb.rdrecord(base)
    annotation = wfdb.rdann(base, "atr")
    signal = normalize_signal(record.p_signal[:, 0])
    peaks = [sample for sample, symbol in zip(annotation.sample, annotation.symbol) if symbol in BEAT_SYMBOLS]
    labels = labels_from_peaks(len(signal), np.asarray(peaks), sigma=0.0)
    return window_pair(signal, labels, WINDOW)


def collect(record_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for record_id in record_ids:
        try:
            x, y = load_record(record_id)
        except Exception as exc:
            print(f"Skipping {record_id}: {exc}")
            continue
        xs.append(x)
        ys.append(y)
    return concat_or_empty(xs, WINDOW), concat_or_empty(ys, WINDOW)


def main() -> None:
    train_records, test_records = read_split_info(SPLIT_INFO)
    if not train_records:
        records = available_records()
        split = int(len(records) * 0.8)
        train_records, test_records = records[:split], records[split:]

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
