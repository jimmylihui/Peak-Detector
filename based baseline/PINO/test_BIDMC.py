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
from multiprocessing import Pool, cpu_count
from functools import partial

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

def calculate_mape(predicted: float, actual: float, min_threshold: float = 1.0) -> float:
    """Calculate Mean Absolute Percentage Error (MAPE) for a single value."""
    if np.isnan(predicted) or np.isnan(actual) or actual == 0:
        return np.nan
    if abs(actual) < min_threshold:
        return np.nan
    return abs((predicted - actual) / actual) * 100.0

def process_sample(args):
    i, sig, lbl, fs, tol = args
    gt_peaks = extract_gt_peaks_from_label(lbl, fs=fs)

    inference_start = time.time()
    pred_peaks = pino_detect(preprocess_signal(sig), fs=fs)
    inference_time = time.time() - inference_start

    binary_peaks = np.zeros(1000, dtype=np.float32)
    valid_peaks = pred_peaks[pred_peaks < 1000]
    if len(valid_peaks) > 0:
        binary_peaks[valid_peaks] = 1.0

    result = {'tp': 0, 'fp': 0, 'fn': 0, 'hr_error': None, 'hrv_sdnn_error': None, 'hrv_rmssd_error': None, 'hr_mape': None, 'hrv_sdnn_mape': None, 'hrv_rmssd_mape': None, 'inference_time': inference_time, 'binary_peaks': binary_peaks}
    if len(pred_peaks) > 0:
        m = calculate_detection_metrics(pred_peaks, gt_peaks, tol)
        result.update({'tp': m['tp'], 'fp': m['fp'], 'fn': m['fn']})
        pred_hr, gt_hr = calculate_heart_rate_from_peaks(pred_peaks, fs=fs), calculate_heart_rate_from_peaks(gt_peaks, fs=fs)
        if not (np.isnan(pred_hr) or np.isnan(gt_hr)):
            result['hr_error'] = abs(pred_hr - gt_hr)
            result['hr_mape'] = calculate_mape(pred_hr, gt_hr, min_threshold=30.0)
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
                result[f'hrv_{k}_mape'] = calculate_mape(p_hrv, g_hrv, min_threshold=5.0)
    return result

def evaluate_fold(X_fold, y_fold, fs, tolerance_samples, n_cpus):
    """Evaluate a single fold and return metrics and binary peaks"""
    args_list = [(i, X_fold[i], y_fold[i], fs, tolerance_samples) for i in range(X_fold.shape[0])]

    with Pool(n_cpus) as pool:
        results = list(tqdm(pool.imap(process_sample, args_list), total=len(args_list), desc='Evaluating fold'))

    all_tp = sum(r['tp'] for r in results)
    all_fp = sum(r['fp'] for r in results)
    all_fn = sum(r['fn'] for r in results)
    hr_errors = [r['hr_error'] for r in results if r['hr_error'] is not None]
    hrv_sdnn_errors = [r['hrv_sdnn_error'] for r in results if r['hrv_sdnn_error'] is not None]
    hrv_rmssd_errors = [r['hrv_rmssd_error'] for r in results if r['hrv_rmssd_error'] is not None]
    hr_mapes = [r['hr_mape'] for r in results if r['hr_mape'] is not None and not np.isnan(r['hr_mape'])]
    hrv_sdnn_mapes = [r['hrv_sdnn_mape'] for r in results if r['hrv_sdnn_mape'] is not None and not np.isnan(r['hrv_sdnn_mape'])]
    hrv_rmssd_mapes = [r['hrv_rmssd_mape'] for r in results if r['hrv_rmssd_mape'] is not None and not np.isnan(r['hrv_rmssd_mape'])]
    inference_times = [r['inference_time'] for r in results]

    overall_precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    overall_recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
    hr_mae = float(np.mean(hr_errors)) if len(hr_errors) > 0 else float('nan')
    hrv_sdnn_mae = float(np.mean(hrv_sdnn_errors)) if len(hrv_sdnn_errors) > 0 else float('nan')
    hrv_rmssd_mae = float(np.mean(hrv_rmssd_errors)) if len(hrv_rmssd_errors) > 0 else float('nan')
    hr_mape = float(np.mean(hr_mapes)) if len(hr_mapes) > 0 else float('nan')
    hrv_sdnn_mape = float(np.mean(hrv_sdnn_mapes)) if len(hrv_sdnn_mapes) > 0 else float('nan')
    hrv_rmssd_mape = float(np.mean(hrv_rmssd_mapes)) if len(hrv_rmssd_mapes) > 0 else float('nan')
    total_inference_time = sum(inference_times)
    avg_inference_time_per_sample = total_inference_time / len(results) if len(results) > 0 else 0.0
    inference_throughput = 1 / avg_inference_time_per_sample if avg_inference_time_per_sample > 0 else 0.0

    binary_peaks = [r['binary_peaks'] for r in results]

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
        'fn': all_fn,
        'binary_peaks': binary_peaks,
        'total_inference_time': total_inference_time,
        'avg_inference_time_per_sample': avg_inference_time_per_sample,
        'inference_throughput': inference_throughput
    }

# -----------------------------
# Main testing routine
# -----------------------------

def main():
    # Config
    fs = 125
    window_length = 1000
    tolerance_samples = int(0.03 * fs)
    x_train_path = '/path/to/workspace/project-BCG-LLM/combined_data/combined_splitted_data/PPG/processed/X_train.npy'
    y_train_path = '/path/to/workspace/project-BCG-LLM/combined_data/combined_splitted_data/PPG/processed/Y_train.npy'
    x_test_path = '/path/to/workspace/project-BCG-LLM/combined_data/combined_splitted_data/PPG/processed/X_test.npy'
    y_test_path = '/path/to/workspace/project-BCG-LLM/combined_data/combined_splitted_data/PPG/processed/Y_test.npy'

    # Detection method: PINO
    detection_mode = 'pino'

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
    print(f"Using detection mode: {detection_mode}")
    print(f"Parallelizing across {n_cpus} CPUs")

    # Run cross-validation
    fold_metrics = []
    all_predicted_peaks_binary = []
    for fold_idx in range(n_folds):
        print(f'\n{"="*80}')
        print(f'Fold {fold_idx + 1}/{n_folds}')
        print(f'{"="*80}')

        fold_start = fold_idx * original_test_size
        fold_end = (fold_idx + 1) * original_test_size

        X_fold = X_combined[fold_start:fold_end]
        y_fold = y_combined[fold_start:fold_end]

        print(f"Fold size: {X_fold.shape[0]} samples")

        fold_result = evaluate_fold(X_fold, y_fold, fs, tolerance_samples, n_cpus)
        fold_metrics.append(fold_result)

        all_predicted_peaks_binary.extend(fold_result['binary_peaks'])

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
                            'hr_mape', 'hrv_sdnn_mape', 'hrv_rmssd_mape', 'avg_inference_time_per_sample', 'inference_throughput']

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
    
    total_inference_time = sum(fold_result['total_inference_time'] for fold_result in fold_metrics)
    print(f"\nTotal Inference Time: {total_inference_time:.2f} seconds")
    print(f"Avg Inference Time per Sample: {aggregated['avg_inference_time_per_sample']['mean']:.6f} ± {aggregated['avg_inference_time_per_sample']['std']:.6f} seconds"
          if not np.isnan(aggregated['avg_inference_time_per_sample']['mean']) else "Avg Inference Time per Sample: NaN")
    print(f"Inference Throughput: {aggregated['inference_throughput']['mean']:.2f} ± {aggregated['inference_throughput']['std']:.2f} samples/sec"
          if not np.isnan(aggregated['inference_throughput']['mean']) else "Inference Throughput: NaN")

    total_tp = sum(fold_result['tp'] for fold_result in fold_metrics)
    total_fp = sum(fold_result['fp'] for fold_result in fold_metrics)
    total_fn = sum(fold_result['fn'] for fold_result in fold_metrics)
    print(f"\nTotal across all folds: TP={total_tp}, FP={total_fp}, FN={total_fn}")
    print(f"Number of folds: {n_folds}")

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'BIDMC_predicted_peaks.npy')
    predicted_peaks_array = np.array(all_predicted_peaks_binary, dtype=np.float32)
    np.save(output_path, predicted_peaks_array)
    print(f'\nPredicted peaks saved to: {output_path}')
    print(f'Shape: {predicted_peaks_array.shape}')

if __name__ == '__main__':
    main()