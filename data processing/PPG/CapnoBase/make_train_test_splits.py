import os
import glob
import json
import numpy as np


def read_test_ids(split_meta_path: str, fallback_ids: set[str]) -> set[str]:
    if os.path.exists(split_meta_path):
        with open(split_meta_path, "r") as f:
            meta = json.load(f)
        test = meta.get("subject_splits", {}).get("test", [])
        return {sid.split("_")[0] for sid in test}
    return set(fallback_ids)


def subject_id_from_filename(filename: str) -> str:
    base = os.path.basename(filename)
    # Examples: 0009_ppg.npy, 0009_peaks.npy
    return base.split("_")[0]


def load_pairs(npy_dir: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    ppg_files = sorted(glob.glob(os.path.join(npy_dir, "*_ppg.npy")))
    for ppg_fp in ppg_files:
        sid = subject_id_from_filename(ppg_fp)
        peaks_fp = os.path.join(npy_dir, f"{sid}_peaks.npy")
        if not os.path.exists(peaks_fp):
            continue
        ppg = np.load(ppg_fp)
        peaks = np.load(peaks_fp)
        # Some peak files might be saved with shape (N,) or (N,1); flatten to 1D
        ppg = np.asarray(ppg).reshape(-1)
        peaks = np.asarray(peaks).reshape(-1)
        length = min(ppg.shape[0], peaks.shape[0])
        if length == 0:
            continue
        pairs[sid] = (ppg[:length], peaks[:length])
    return pairs


def window_signal_and_labels(signal: np.ndarray, labels: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    assert signal.shape[0] == labels.shape[0]
    total = signal.shape[0]
    usable = (total // window) * window
    if usable == 0:
        return np.empty((0, window), dtype=signal.dtype), np.empty((0, window), dtype=labels.dtype)
    x = signal[:usable].reshape(-1, window)
    y = labels[:usable].reshape(-1, window)
    return x, y


def main():
    npy_dir = "/path/to/workspace/project-BCG-LLM/PPG_peaks/dataset/capnobase/data/npy"
    out_dir = "/path/to/workspace/project-BCG-LLM/PPG_peaks/splitted_dataset/capnobase"
    os.makedirs(out_dir, exist_ok=True)

    split_meta = os.path.join(out_dir, "split_metadata.json")
    fallback_test_ids = {"0023", "0031", "0104", "0125", "0147", "0148", "0309", "0311", "0332"}
    window = 1000
    sigma = 5.0

    test_ids = read_test_ids(split_meta, fallback_test_ids)

    sid_to_pair = load_pairs(npy_dir)

    # Precompute Gaussian kernel with radius 3*sigma
    radius = int(3 * sigma)
    kernel_x = np.arange(-radius, radius + 1)
    gaussian_kernel = np.exp(-(kernel_x ** 2) / (2.0 * sigma * sigma))
    # normalize to peak of 1 at center
    gaussian_kernel /= gaussian_kernel.max() if gaussian_kernel.max() != 0 else 1.0

    x_train_list: list[np.ndarray] = []
    y_train_list: list[np.ndarray] = []
    x_test_list: list[np.ndarray] = []
    y_test_list: list[np.ndarray] = []

    # Accumulate binary and gaussian-smoothed labels
    x_train_list_bin: list[np.ndarray] = []
    y_train_list_bin: list[np.ndarray] = []
    x_test_list_bin: list[np.ndarray] = []
    y_test_list_bin: list[np.ndarray] = []

    x_train_list_gauss: list[np.ndarray] = []
    y_train_list_gauss: list[np.ndarray] = []
    x_test_list_gauss: list[np.ndarray] = []
    y_test_list_gauss: list[np.ndarray] = []

    for sid, (ppg, peaks) in sid_to_pair.items():
        # Binary windows
        x_bin, y_bin = window_signal_and_labels(ppg, peaks, window)
        # Gaussian-smoothed labels over full sequence then window
        y_full_gauss = np.convolve(peaks.astype(np.float32), gaussian_kernel.astype(np.float32), mode="same")
        y_full_gauss = np.clip(y_full_gauss, 0.0, 1.0)
        x_gauss, y_gauss = window_signal_and_labels(ppg, y_full_gauss, window)
        if x_bin.shape[0] == 0:
            continue
        if sid in test_ids:
            x_test_list_bin.append(x_bin)
            y_test_list_bin.append(y_bin)
            x_test_list_gauss.append(x_gauss)
            y_test_list_gauss.append(y_gauss)
        else:
            x_train_list_bin.append(x_bin)
            y_train_list_bin.append(y_bin)
            x_train_list_gauss.append(x_gauss)
            y_train_list_gauss.append(y_gauss)

    X_train = np.concatenate(x_train_list_bin, axis=0) if x_train_list_bin else np.empty((0, window), dtype=np.float64)
    Y_train = np.concatenate(y_train_list_bin, axis=0) if y_train_list_bin else np.empty((0, window), dtype=np.uint8)
    X_test = np.concatenate(x_test_list_bin, axis=0) if x_test_list_bin else np.empty((0, window), dtype=np.float64)
    Y_test = np.concatenate(y_test_list_bin, axis=0) if y_test_list_bin else np.empty((0, window), dtype=np.uint8)

    X_train_gauss = np.concatenate(x_train_list_gauss, axis=0) if x_train_list_gauss else np.empty((0, window), dtype=np.float64)
    Y_train_gauss = np.concatenate(y_train_list_gauss, axis=0) if y_train_list_gauss else np.empty((0, window), dtype=np.float32)
    X_test_gauss = np.concatenate(x_test_list_gauss, axis=0) if x_test_list_gauss else np.empty((0, window), dtype=np.float64)
    Y_test_gauss = np.concatenate(y_test_list_gauss, axis=0) if y_test_list_gauss else np.empty((0, window), dtype=np.float32)

    # Save as separate .npy files for easy loading
    np.save(os.path.join(out_dir, "X_train.npy"), X_train)
    np.save(os.path.join(out_dir, "Y_train.npy"), Y_train)
    np.save(os.path.join(out_dir, "X_test.npy"), X_test)
    np.save(os.path.join(out_dir, "Y_test.npy"), Y_test)

    # Save Gaussian versions
    np.save(os.path.join(out_dir, "Y_train_gauss.npy"), Y_train_gauss.astype(np.float32))
    np.save(os.path.join(out_dir, "Y_test_gauss.npy"), Y_test_gauss.astype(np.float32))

    # Also save a compact .npz bundle
    np.savez_compressed(
        os.path.join(out_dir, "splits_len1000.npz"),
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        Y_test=Y_test,
        window=window,
        test_ids=np.array(sorted(list(test_ids))),
    )
    np.savez_compressed(
        os.path.join(out_dir, "splits_len1000_gauss.npz"),
        X_train=X_train_gauss,
        Y_train=Y_train_gauss.astype(np.float32),
        X_test=X_test_gauss,
        Y_test=Y_test_gauss.astype(np.float32),
        window=window,
        sigma=sigma,
        test_ids=np.array(sorted(list(test_ids))),
    )

    print("Saved:")
    print(os.path.join(out_dir, "X_train.npy"), X_train.shape)
    print(os.path.join(out_dir, "Y_train.npy"), Y_train.shape)
    print(os.path.join(out_dir, "X_test.npy"), X_test.shape)
    print(os.path.join(out_dir, "Y_test.npy"), Y_test.shape)
    print(os.path.join(out_dir, "Y_train_gauss.npy"), Y_train_gauss.shape)
    print(os.path.join(out_dir, "Y_test_gauss.npy"), Y_test_gauss.shape)


if __name__ == "__main__":
    main()


