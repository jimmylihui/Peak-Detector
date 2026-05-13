#!/usr/bin/env python3
"""
Test PINO algorithm on held-out test set and report metrics.

- Uses PINO (Pino et al.) algorithm for peak detection
- Loads X_test.npy and y_test.npy from provided paths
- Detects peaks, computes HR MAE, precision, recall, F1
"""

import os
import numpy as np
from scipy import signal
from tqdm import tqdm
import sys
import time

# Ensure PINO algorithms package is importable
sys.path.append('/path/to/workspace/project-BCG-LLM/BSG_LLM/benchmark/bcg-hr-dl')
from algorithms import pino as pino_alg


"""
PINO detector wrapper
"""

def pino_detect(signal_array: np.ndarray, fs: int = 360) -> np.ndarray:
    """Run PINO J-peak detection and return peak indices as numpy array."""
    try:
        peaks = pino_alg.pino(signal_array.astype(float), f=fs)
        return np.asarray(peaks, dtype=int)
    except Exception as e:
        print(f"PINO detection failed: {e}")
        return np.array([], dtype=int)


# -----------------------------
# Utility functions (unchanged)
# -----------------------------

def preprocess_signal(signal_array: np.ndarray) -> np.ndarray:
    """Normalize per sample."""
    signal_array = np.ascontiguousarray(signal_array)
    return (signal_array - np.mean(signal_array)) / (np.std(signal_array) + 1e-8)



def extract_gt_peaks_from_label(gt_label: np.ndarray, fs: int = 360) -> np.ndarray:
    """Derive GT peak indices from soft/binary label sequence using peak picking."""
    refractory = int(fs * 0.2)
    peaks, _ = signal.find_peaks(gt_label, height=0.5, distance=refractory)
    return peaks.astype(int)

def calculate_detection_metrics(pred_peaks: np.ndarray, gt_peaks: np.ndarray, tolerance_samples: int) -> dict:
    if len(pred_peaks) == 0 and len(gt_peaks) == 0:
        return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'tp': 0, 'fp': 0, 'fn': 0}
    if len(pred_peaks) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': 0, 'fn': len(gt_peaks)}
    if len(gt_peaks) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': len(pred_peaks), 'fn': 0}
    tp = 0
    matched_gt = set()
    for pred_idx in pred_peaks:
        # match once within tolerance
        candidates = np.where(np.abs(gt_peaks - pred_idx) <= tolerance_samples)[0]
        for c in candidates:
            if c not in matched_gt:
                matched_gt.add(c)
                tp += 1
                break
    fp = len(pred_peaks) - tp
    fn = len(gt_peaks) - tp
    precision = tp / len(pred_peaks) if len(pred_peaks) > 0 else 0.0
    recall = tp / len(gt_peaks) if len(gt_peaks) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {'precision': precision, 'recall': recall, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn}

def calculate_heart_rate_from_peaks(peaks: np.ndarray, fs: int = 360) -> float:
    if peaks is None or len(peaks) < 2:
        return np.nan
    rr_intervals = np.diff(peaks) / fs
    if len(rr_intervals) == 0:
        return np.nan
    avg_rr = np.mean(rr_intervals)
    if avg_rr <= 0:
        return np.nan
    return 60.0 / avg_rr

def calculate_hrv_from_peaks(peaks: np.ndarray, fs: int = 360, metric: str = 'sdnn') -> float:
    """Calculate HRV (SDNN or RMSSD) from peaks in milliseconds."""
    if peaks is None or len(peaks) < 2:
        return np.nan
    rr_intervals = np.diff(peaks) / fs * 1000.0
    if len(rr_intervals) == 0:
        return np.nan
    if metric.lower() == 'sdnn':
        return float(np.std(rr_intervals, ddof=1))
    elif metric.lower() == 'rmssd':
        return float(np.sqrt(np.mean(np.diff(rr_intervals) ** 2))) if len(rr_intervals) >= 2 else np.nan
    return np.nan

# -----------------------------
# Main testing routine
# -----------------------------

def main():
    # Config
    fs = 100
    window_length = 1000
    tolerance_samples = int(0.03 * fs)
    x_test_path = '/path/to/workspace/project-BCG-LLM/combined_data/combined_splitted_data/BCG_hospital/X_test.npy'
    y_test_path = '/path/to/workspace/project-BCG-LLM/combined_data/combined_splitted_data/BCG_hospital/y_test.npy'

    # Detection method: PINO
    detection_mode = 'pino'

    # Load data
    X_test = np.load(x_test_path)
    y_test = np.load(y_test_path)
    assert X_test.shape[0] == y_test.shape[0], 'Mismatched number of samples'
    assert X_test.shape[1] == window_length, f'Expected window length {window_length}'

    print(f"Loaded {X_test.shape[0]} test samples")
    print(f"Using detection mode: {detection_mode}")

    # Accumulators
    all_tp = 0
    all_fp = 0
    all_fn = 0
    hr_errors = []
    hrv_sdnn_errors = []
    hrv_rmssd_errors = []
    all_predicted_peaks_binary = []  # Store binary predicted peaks for each sample
    
    # Timing for inference only
    total_inference_time = 0.0

    # Iterate samples
    for i in tqdm(range(X_test.shape[0]), desc='Testing with PINO'):
        signal_np = preprocess_signal(X_test[i])
        label_np = y_test[i]

        # GT peaks from label
        gt_peaks = extract_gt_peaks_from_label(label_np, fs=fs)

        # Time only the inference/detection step
        inference_start = time.time()
        pred_peaks = pino_detect(signal_np, fs=fs)
        total_inference_time += time.time() - inference_start

        # Convert to binary format (1000 length)
        binary_peaks = np.zeros(1000, dtype=np.float32)
        valid_peaks = pred_peaks[pred_peaks < 1000]  # Only keep peaks within range
        if len(valid_peaks) > 0:
            binary_peaks[valid_peaks] = 1.0
        all_predicted_peaks_binary.append(binary_peaks)

        # Only compute metrics if there are predicted peaks
        if len(pred_peaks) > 0:
            # Detection metrics
            metrics = calculate_detection_metrics(pred_peaks, gt_peaks, tolerance_samples)
            all_tp += metrics['tp']
            all_fp += metrics['fp']
            all_fn += metrics['fn']

            # Heart rate
            pred_hr = calculate_heart_rate_from_peaks(pred_peaks, fs=fs)
            gt_hr = calculate_heart_rate_from_peaks(gt_peaks, fs=fs)
            if not (np.isnan(pred_hr) or np.isnan(gt_hr)):
                hr_errors.append(abs(pred_hr - gt_hr))
            
            # HRV
            for metric, errors in [('sdnn', hrv_sdnn_errors), ('rmssd', hrv_rmssd_errors)]:
                pred_hrv = calculate_hrv_from_peaks(pred_peaks, fs=fs, metric=metric)
                gt_hrv = calculate_hrv_from_peaks(gt_peaks, fs=fs, metric=metric)
                if not (np.isnan(pred_hrv) or np.isnan(gt_hrv)):
                    errors.append(abs(pred_hrv - gt_hrv))
    
    # Calculate inference throughput (excluding metric calculations)
    # Calculate as 1 / inference_time_per_sample
    avg_inference_time_per_sample = total_inference_time / X_test.shape[0]
    inference_throughput = 1 / avg_inference_time_per_sample

    # Aggregate metrics
    overall_precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    overall_recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
    hr_mae = float(np.mean(hr_errors)) if len(hr_errors) > 0 else float('nan')
    hrv_sdnn_mae = float(np.mean(hrv_sdnn_errors)) if len(hrv_sdnn_errors) > 0 else float('nan')
    hrv_rmssd_mae = float(np.mean(hrv_rmssd_errors)) if len(hrv_rmssd_errors) > 0 else float('nan')

    print('\n=== PINO Test Metrics ===')
    print(f'Detection Mode: {detection_mode}')
    print(f'Total Inference Time: {total_inference_time:.2f} seconds')
    print(f'Avg Inference Time per Sample: {avg_inference_time_per_sample:.6f} seconds')
    print(f'Inference Throughput: {inference_throughput:.2f} samples/sec')
    print(f'Heart Rate MAE (BPM): {hr_mae:.2f}' if not np.isnan(hr_mae) else 'Heart Rate MAE (BPM): NaN')
    print(f'HRV SDNN MAE (ms): {hrv_sdnn_mae:.2f}' if not np.isnan(hrv_sdnn_mae) else 'HRV SDNN MAE (ms): NaN')
    print(f'HRV RMSSD MAE (ms): {hrv_rmssd_mae:.2f}' if not np.isnan(hrv_rmssd_mae) else 'HRV RMSSD MAE (ms): NaN')
    print(f'Precision: {overall_precision:.4f}')
    print(f'Recall:    {overall_recall:.4f}')
    print(f'F1-Score:  {overall_f1:.4f}')
    print(f'Total TP: {all_tp}, FP: {all_fp}, FN: {all_fn}')

    # Save predicted peaks as binary array with shape (n, 1000)
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MITBIH_predicted_peaks.npy')
    predicted_peaks_array = np.array(all_predicted_peaks_binary, dtype=np.float32)
    np.save(output_path, predicted_peaks_array)
    print(f'\nPredicted peaks saved to: {output_path}')
    print(f'Shape: {predicted_peaks_array.shape}')

if __name__ == '__main__':
    main()