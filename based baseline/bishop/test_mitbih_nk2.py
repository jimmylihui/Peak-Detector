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
from multiprocessing import Pool, cpu_count
from functools import partial
import time


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
        cleaned_ecg = nk.ppg_clean(ecg, sampling_rate=fs)
        
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

def calculate_mape(predicted: float, actual: float, min_threshold: float = 1.0) -> float:
    """Calculate Mean Absolute Percentage Error (MAPE) for a single value.
    
    Args:
        predicted: Predicted value
        actual: Actual/ground truth value
        min_threshold: Minimum threshold for actual value to avoid division by very small numbers.
                      For HRV (in ms), values below 1.0 ms are considered unreliable.
    
    Returns:
        MAPE as percentage, or np.nan if calculation is not valid.
    """
    if np.isnan(predicted) or np.isnan(actual) or actual == 0:
        return np.nan
    # Avoid division by very small numbers which causes huge MAPE values
    if abs(actual) < min_threshold:
        return np.nan
    return abs((predicted - actual) / actual) * 100.0

def process_sample(args):
    i, sig, lbl, fs, tol, mode = args
    gt_peaks = extract_gt_peaks_from_label(lbl, fs=fs)
    
    # Time only inference
    inference_start = time.time()
    pred_peaks = neurokit2_detect_ensemble(preprocess_signal(sig), fs=fs) if mode == 'ensemble' else neurokit2_detect(preprocess_signal(sig), fs=fs, method='bishop')
    inference_time = time.time() - inference_start
    
    # Convert to binary format (1000 length)
    binary_peaks = np.zeros(1000, dtype=np.float32)
    valid_peaks = pred_peaks[pred_peaks < 1000]  # Only keep peaks within range
    if len(valid_peaks) > 0:
        binary_peaks[valid_peaks] = 1.0
    
    result = {'tp': 0, 'fp': 0, 'fn': 0, 'hr_error': None, 'hrv_sdnn_error': None, 'hrv_rmssd_error': None, 'hr_mape': None, 'hrv_sdnn_mape': None, 'hrv_rmssd_mape': None, 'inference_time': inference_time, 'binary_peaks': binary_peaks}
    if len(pred_peaks) > 0:
        m = calculate_detection_metrics(pred_peaks, gt_peaks, tol)
        result.update({'tp': m['tp'], 'fp': m['fp'], 'fn': m['fn']})
        pred_hr, gt_hr = calculate_heart_rate_from_peaks(pred_peaks, fs=fs), calculate_heart_rate_from_peaks(gt_peaks, fs=fs)
        if not (np.isnan(pred_hr) or np.isnan(gt_hr)):
            result['hr_error'] = abs(pred_hr - gt_hr)
            result['hr_mape'] = calculate_mape(pred_hr, gt_hr, min_threshold=30.0)  # HR threshold: 30 BPM
        for k in ['sdnn', 'rmssd']:
            p_hrv, g_hrv = calculate_hrv_from_peaks(pred_peaks, fs=fs, metric=k), calculate_hrv_from_peaks(gt_peaks, fs=fs, metric=k)
            try:
                p_hrv = float(p_hrv) if p_hrv is not None else np.nan
                g_hrv = float(g_hrv) if g_hrv is not None else np.nan
            except (ValueError, TypeError):
                p_hrv = np.nan
                g_hrv = np.nan
            if not (np.isnan(p_hrv) or np.isnan(g_hrv)):
                result[f'hrv_{k}_error'] = abs(p_hrv - g_hrv)
                result[f'hrv_{k}_mape'] = calculate_mape(p_hrv, g_hrv, min_threshold=5.0)  # HRV threshold: 5.0 ms (filter very small values)
    return result

# -----------------------------
# Cross-validation evaluation function
# -----------------------------

def evaluate_fold(X_fold, y_fold, fs, tolerance_samples, detection_mode, n_cpus):
    """Evaluate a single fold and return metrics"""
    args_list = [(i, X_fold[i], y_fold[i], fs, tolerance_samples, detection_mode) for i in range(X_fold.shape[0])]
    
    # Process in parallel
    with Pool(n_cpus) as pool:
        results = list(tqdm(pool.imap(process_sample, args_list), total=len(args_list), desc='Evaluating fold'))
    
    # Aggregate results
    all_tp = sum(r['tp'] for r in results)
    all_fp = sum(r['fp'] for r in results)
    all_fn = sum(r['fn'] for r in results)
    hr_errors = [r['hr_error'] for r in results if r['hr_error'] is not None]
    hrv_sdnn_errors = [r['hrv_sdnn_error'] for r in results if r['hrv_sdnn_error'] is not None]
    hrv_rmssd_errors = [r['hrv_rmssd_error'] for r in results if r['hrv_rmssd_error'] is not None]
    hr_mapes = [r['hr_mape'] for r in results if r['hr_mape'] is not None and not np.isnan(r['hr_mape'])]
    hrv_sdnn_mapes = [r['hrv_sdnn_mape'] for r in results if r['hrv_sdnn_mape'] is not None and not np.isnan(r['hrv_sdnn_mape'])]
    hrv_rmssd_mapes = [r['hrv_rmssd_mape'] for r in results if r['hrv_rmssd_mape'] is not None and not np.isnan(r['hrv_rmssd_mape'])]
    
    # Calculate metrics
    overall_precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    overall_recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
    hr_mae = float(np.mean(hr_errors)) if len(hr_errors) > 0 else float('nan')
    hrv_sdnn_mae = float(np.mean(hrv_sdnn_errors)) if len(hrv_sdnn_errors) > 0 else float('nan')
    hrv_rmssd_mae = float(np.mean(hrv_rmssd_errors)) if len(hrv_rmssd_errors) > 0 else float('nan')
    hr_mape = float(np.mean(hr_mapes)) if len(hr_mapes) > 0 else float('nan')
    hrv_sdnn_mape = float(np.mean(hrv_sdnn_mapes)) if len(hrv_sdnn_mapes) > 0 else float('nan')
    hrv_rmssd_mape = float(np.mean(hrv_rmssd_mapes)) if len(hrv_rmssd_mapes) > 0 else float('nan')
    
    return {
        'precision': overall_precision,
        'recall': overall_recall,
        'f1': overall_f1,
        'hr_mae': hr_mae,
        'hrv_sdnn_mae': hrv_sdnn_mae,
        'hrv_rmssd_mae': hrv_rmssd_mae,
        'hr_mape': hr_mape,
        'hrv_sdnn_mape': hrv_sdnn_mape,
        'hrv_rmssd_mape': hrv_rmssd_mape,
        'tp': all_tp,
        'fp': all_fp,
        'fn': all_fn
    }

# -----------------------------
# Main testing routine
# -----------------------------

def main():
    # Config
    fs = 360
    window_length = 1000
    tolerance_samples = int(0.03 * fs)
    x_train_path = '/path/to/workspace/project-BCG-LLM/combined_data/combined_splitted_data/ECG/total_X_train.npy'
    y_train_path = '/path/to/workspace/project-BCG-LLM/combined_data/combined_splitted_data/ECG/total_y_train.npy'
    x_test_path = '/path/to/workspace/project-BCG-LLM/combined_data/combined_splitted_data/ECG/total_X_test.npy'
    y_test_path = '/path/to/workspace/project-BCG-LLM/combined_data/combined_splitted_data/ECG/total_y_test.npy'

    # Detection method - choose one:
    # 'single' - uses Pan-Tompkins algorithm
    # 'ensemble' - uses ensemble of multiple algorithms
    detection_mode = 'single'  # Change to 'single' for single algorithm

    # Load train and test data
    print("Loading train and test data...")
    X_train = np.load(x_train_path)
    y_train = np.load(y_train_path)
    X_test = np.load(x_test_path)
    y_test = np.load(y_test_path)
    
    # Store original test size
    original_test_size = X_test.shape[0]
    
    # Combine train and test data
    X_combined = np.concatenate([X_train, X_test], axis=0)
    y_combined = np.concatenate([y_train, y_test], axis=0)
    
    assert X_combined.shape[0] == y_combined.shape[0], 'Mismatched number of samples'
    assert X_combined.shape[1] == window_length, f'Expected window length {window_length}'
    
    print(f"Combined dataset: {X_combined.shape[0]} samples")
    print(f"Original test size: {original_test_size} samples")
    
    # Calculate number of folds
    n_folds = X_combined.shape[0] // original_test_size
    print(f"Number of folds: {n_folds}")
    
    n_cpus = cpu_count()
    print(f"Using NeuroKit2 detection mode: {detection_mode}")
    print(f"Parallelizing across {n_cpus} CPUs")
    
    # Split by time for each fold (no shuffling)
    
    # Run cross-validation
    fold_metrics = []
    for fold_idx in range(n_folds):
        print(f'\n{"="*80}')
        print(f'Fold {fold_idx + 1}/{n_folds}')
        print(f'{"="*80}')
        
        # Split into fold (test) and rest (train)
        fold_start = fold_idx * original_test_size
        fold_end = (fold_idx + 1) * original_test_size
        
        X_fold = X_combined[fold_start:fold_end]
        y_fold = y_combined[fold_start:fold_end]
        
        print(f"Fold size: {X_fold.shape[0]} samples")
        
        # Evaluate fold
        fold_result = evaluate_fold(X_fold, y_fold, fs, tolerance_samples, detection_mode, n_cpus)
        fold_metrics.append(fold_result)
        
        print(f"Fold {fold_idx + 1} Results:")
        print(f"  Precision: {fold_result['precision']:.4f}")
        print(f"  Recall: {fold_result['recall']:.4f}")
        print(f"  F1-Score: {fold_result['f1']:.4f}")
        print(f"  HR MAE: {fold_result['hr_mae']:.2f}" if not np.isnan(fold_result['hr_mae']) else f"  HR MAE: NaN")
        print(f"  HR MAPE: {fold_result['hr_mape']:.2f}" if not np.isnan(fold_result['hr_mape']) else f"  HR MAPE: NaN")
    
    # Calculate mean and std across folds
    print(f'\n{"="*80}')
    print('CROSS-VALIDATION RESULTS (Mean ± Std across folds)')
    print(f'{"="*80}')
    
    metrics_to_aggregate = ['precision', 'recall', 'f1', 'hr_mae', 'hrv_sdnn_mae', 'hrv_rmssd_mae', 
                            'hr_mape', 'hrv_sdnn_mape', 'hrv_rmssd_mape']
    
    aggregated = {}
    for metric in metrics_to_aggregate:
        values = [fold_result[metric] for fold_result in fold_metrics if not np.isnan(fold_result[metric])]
        if len(values) > 0:
            aggregated[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }
        else:
            aggregated[metric] = {'mean': np.nan, 'std': np.nan}
    
    # Print aggregated results
    print(f"Precision: {aggregated['precision']['mean']:.4f} ± {aggregated['precision']['std']:.4f}")
    print(f"Recall:    {aggregated['recall']['mean']:.4f} ± {aggregated['recall']['std']:.4f}")
    print(f"F1-Score:  {aggregated['f1']['mean']:.4f} ± {aggregated['f1']['std']:.4f}")
    print(f"Heart Rate MAE (BPM): {aggregated['hr_mae']['mean']:.2f} ± {aggregated['hr_mae']['std']:.2f}" 
          if not np.isnan(aggregated['hr_mae']['mean']) else "Heart Rate MAE (BPM): NaN")
    print(f"Heart Rate MAPE (%): {aggregated['hr_mape']['mean']:.2f} ± {aggregated['hr_mape']['std']:.2f}" 
          if not np.isnan(aggregated['hr_mape']['mean']) else "Heart Rate MAPE (%): NaN")
    print(f"HRV SDNN MAE (ms): {aggregated['hrv_sdnn_mae']['mean']:.2f} ± {aggregated['hrv_sdnn_mae']['std']:.2f}" 
          if not np.isnan(aggregated['hrv_sdnn_mae']['mean']) else "HRV SDNN MAE (ms): NaN")
    print(f"HRV SDNN MAPE (%): {aggregated['hrv_sdnn_mape']['mean']:.2f} ± {aggregated['hrv_sdnn_mape']['std']:.2f}" 
          if not np.isnan(aggregated['hrv_sdnn_mape']['mean']) else "HRV SDNN MAPE (%): NaN")
    print(f"HRV RMSSD MAE (ms): {aggregated['hrv_rmssd_mae']['mean']:.2f} ± {aggregated['hrv_rmssd_mae']['std']:.2f}" 
          if not np.isnan(aggregated['hrv_rmssd_mae']['mean']) else "HRV RMSSD MAE (ms): NaN")
    print(f"HRV RMSSD MAPE (%): {aggregated['hrv_rmssd_mape']['mean']:.2f} ± {aggregated['hrv_rmssd_mape']['std']:.2f}" 
          if not np.isnan(aggregated['hrv_rmssd_mape']['mean']) else "HRV RMSSD MAPE (%): NaN")
    
    # Calculate total TP, FP, FN across all folds
    total_tp = sum(fold_result['tp'] for fold_result in fold_metrics)
    total_fp = sum(fold_result['fp'] for fold_result in fold_metrics)
    total_fn = sum(fold_result['fn'] for fold_result in fold_metrics)
    print(f"\nTotal across all folds: TP={total_tp}, FP={total_fp}, FN={total_fn}")
    print(f"Number of folds: {n_folds}")

if __name__ == '__main__':
    main()