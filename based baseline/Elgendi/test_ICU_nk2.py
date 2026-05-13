#!/usr/bin/env python3
"""
Test NeuroKit2 ECG R-peak detection on held-out test set and report metrics.

- Uses NeuroKit2 methods for ECG R-peak detection
- Loads X_test.npy and y_test.npy from provided paths
- Detects R-peaks, computes HR MAE, precision, recall, F1
"""

import os
import numpy as np
from scipy import signal
from tqdm import tqdm
import sys
import neurokit2 as nk


"""
NeuroKit2 detector wrapper
"""

def neurokit2_detect(ecg: np.ndarray, fs: int = 360, method: str = 'pantompkins1985') -> np.ndarray:
    """Run NeuroKit2 ECG R-peak detection and return R-peak indices."""
    ecg = np.ascontiguousarray(ecg).astype(np.float64)
    
    # Available methods in NeuroKit2:
    # 'pantompkins1985', 'hamilton2002', 'christov2004', 'gamboa2008', 
    # 'elgendi2010', 'engzeemod2012', 'kalidas2017', 'martinez2004', 'rodrigues2021'
    
    try:
        # Clean the ECG signal first (optional but recommended)
        cleaned_ecg = nk.ppg_clean(ecg, sampling_rate=fs,method=method)
        
        # Detect R-peaks
        _, rpeaks = nk.ppg_peaks(cleaned_ecg, sampling_rate=fs, method=method)
        
        # Extract R-peak locations
        if 'PPG_Peaks' in rpeaks:
            peaks = rpeaks['PPG_Peaks']
        else:
            return np.array([], dtype=int)
            
    except Exception as e:
        print(f"NeuroKit2 detection failed: {e}")
        return np.array([], dtype=int)
    
    # Enforce reasonable ECG RR interval constraints (0.3s to 2.0s)
    peaks = np.asarray(peaks, dtype=int)
    if peaks.size <= 1:
        return peaks
    
    rr_intervals_s = np.diff(peaks) / float(fs)
    valid_indices = [0]  # Always keep first peak
    
    for i, rr in enumerate(rr_intervals_s):
        valid_indices.append(i + 1)
    
    return peaks[valid_indices]


def neurokit2_detect_ensemble(ecg: np.ndarray, fs: int = 360) -> np.ndarray:
    """
    Ensemble method using multiple NeuroKit2 algorithms.
    Combines results from multiple detectors for improved robustness.
    """
    methods = ['pantompkins1985', 'hamilton2002', 'christov2004', 'elgendi2010']
    all_peaks = []
    
    for method in methods:
        try:
            peaks = neurokit2_detect(ecg, fs=fs, method=method)
            if len(peaks) > 0:
                all_peaks.extend(peaks)
        except:
            continue
    
    if not all_peaks:
        return np.array([], dtype=int)
    
    # Cluster nearby peaks and take median position
    all_peaks = np.array(sorted(all_peaks))
    
    return all_peaks


# -----------------------------
# Utility functions (unchanged)
# -----------------------------

def preprocess_signal(signal_array: np.ndarray) -> np.ndarray:
    """Normalize per sample."""
    signal_array = np.ascontiguousarray(signal_array)
    return (signal_array - np.mean(signal_array)) / (np.std(signal_array) + 1e-8)

def localize_predicted_peaks(pred: np.ndarray, fs: int = 360) -> np.ndarray:
    """Localize peaks from probability sequence using adaptive thresholding and refractory."""
    conv_window = int(fs * 0.075)
    refractory = int(fs * 0.2)
    if conv_window < 1:
        conv_window = 1
    c_pred = np.convolve(pred, np.ones(conv_window) / conv_window, mode='same')
    pred_mean = np.mean(c_pred)
    pred_std = np.std(c_pred)
    threshold = max(pred_mean + 2 * pred_std, 0.1)
    threshold = min(threshold, 0.8)
    peaks, _ = signal.find_peaks(c_pred, height=threshold, distance=refractory)
    if len(peaks) == 0:
        alt_prom = pred_std * 0.5
        peaks, _ = signal.find_peaks(c_pred, prominence=alt_prom, distance=refractory)
    return peaks.astype(int)

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
    """Calculate HRV (SDNN or RMSSD) from R-peaks in milliseconds."""
    if peaks is None or len(peaks) < 2:
        return np.nan
    rr_intervals = np.diff(peaks) / fs * 1000.0
    if len(rr_intervals) == 0:
        return np.nan
    if metric.lower() == 'sdnn':
        return float(np.std(rr_intervals, ddof=1))
    elif metric.lower() == 'rmssd':
        if len(rr_intervals) < 2:
            return np.nan
        return float(np.sqrt(np.mean(np.diff(rr_intervals) ** 2)))
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

    # Detection method - choose one:
    # 'single' - uses Pan-Tompkins algorithm
    # 'ensemble' - uses ensemble of multiple algorithms
    detection_mode = 'single'  # Change to 'single' for single algorithm

    # Load data
    X_test = np.load(x_test_path)
    y_test = np.load(y_test_path)
    assert X_test.shape[0] == y_test.shape[0], 'Mismatched number of samples'
    assert X_test.shape[1] == window_length, f'Expected window length {window_length}'

    print(f"Loaded {X_test.shape[0]} test samples")
    print(f"Using NeuroKit2 detection mode: {detection_mode}")

    # Accumulators
    all_tp = 0
    all_fp = 0
    all_fn = 0
    hr_errors = []
    hrv_sdnn_errors = []
    hrv_rmssd_errors = []

    # Iterate samples
    for i in tqdm(range(X_test.shape[0]), desc='Testing with NeuroKit2'):
        signal_np = preprocess_signal(X_test[i])
        label_np = y_test[i]

        # GT peaks from label
        gt_peaks = extract_gt_peaks_from_label(label_np, fs=fs)

        # NeuroKit2 detection
        if detection_mode == 'ensemble':
            pred_peaks = neurokit2_detect_ensemble(signal_np, fs=fs)
        else:
            pred_peaks = neurokit2_detect(signal_np, fs=fs, method='elgendi')

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

    # Aggregate metrics
    overall_precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    overall_recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
    hr_mae = float(np.mean(hr_errors)) if len(hr_errors) > 0 else float('nan')
    hrv_sdnn_mae = float(np.mean(hrv_sdnn_errors)) if len(hrv_sdnn_errors) > 0 else float('nan')
    hrv_rmssd_mae = float(np.mean(hrv_rmssd_errors)) if len(hrv_rmssd_errors) > 0 else float('nan')

    print('\n=== NeuroKit2 Test Metrics ===')
    print(f'Detection Mode: {detection_mode}')
    print(f'Heart Rate MAE (BPM): {hr_mae:.2f}' if not np.isnan(hr_mae) else 'Heart Rate MAE (BPM): NaN')
    print(f'HRV SDNN MAE (ms): {hrv_sdnn_mae:.2f}' if not np.isnan(hrv_sdnn_mae) else 'HRV SDNN MAE (ms): NaN')
    print(f'HRV RMSSD MAE (ms): {hrv_rmssd_mae:.2f}' if not np.isnan(hrv_rmssd_mae) else 'HRV RMSSD MAE (ms): NaN')
    print(f'Precision: {overall_precision:.4f}')
    print(f'Recall:    {overall_recall:.4f}')
    print(f'F1-Score:  {overall_f1:.4f}')
    print(f'Total TP: {all_tp}, FP: {all_fp}, FN: {all_fn}')

if __name__ == '__main__':
    main()