#!/usr/bin/env python3
"""
Test Pan-Tompkins++ R-peak detection on held-out test set and report metrics.

Uses the Pan-Tompkins++ implementation from:
  /path/to/workspace/Pan-Tompkins-Plus-Plus/algos/pan_tompkins_plus_plus.py
"""

import os
import sys
import numpy as np
from scipy import signal
from tqdm import tqdm
import time
from multiprocessing import Pool, cpu_count

sys.path.insert(0, '/path/to/workspace/Pan-Tompkins-Plus-Plus')
from algos.pan_tompkins_plus_plus import Pan_Tompkins_Plus_Plus


def ptpp_detect(ecg: np.ndarray, fs: int = 360) -> np.ndarray:
    """Run Pan-Tompkins++ R-peak detection with post-processing."""
    ecg = np.ascontiguousarray(ecg).astype(np.float64)
    detector = Pan_Tompkins_Plus_Plus()

    try:
        qrs_i_raw = detector.rpeak_detection(ecg, fs)
    except Exception:
        return np.array([], dtype=int)

    if qrs_i_raw is None or len(qrs_i_raw) == 0:
        return np.array([], dtype=int)

    peaks = np.asarray(qrs_i_raw, dtype=int)

    # Post-processing from the reference main.py: remove peaks closer than 200ms
    refractory = 0.200 * fs
    corrected = []
    skip_next = False
    for i in range(len(peaks)):
        if skip_next:
            skip_next = False
            continue
        if i > 0 and (peaks[i] - peaks[i - 1]) < refractory:
            skip_next = True
            continue
        corrected.append(int(peaks[i]))
        skip_next = False

    return np.asarray(corrected, dtype=int) if corrected else np.array([], dtype=int)


# --------------- metric helpers (same as nk2 script) ---------------

def extract_gt_peaks_from_label(gt_label: np.ndarray, fs: int = 360) -> np.ndarray:
    refractory = int(fs * 0.2)
    peaks, _ = signal.find_peaks(gt_label, height=0.5, distance=refractory)
    return peaks.astype(int)

def calculate_detection_metrics(pred_peaks, gt_peaks, tolerance_samples):
    if len(pred_peaks) == 0 and len(gt_peaks) == 0:
        return {'tp': 0, 'fp': 0, 'fn': 0}
    if len(pred_peaks) == 0:
        return {'tp': 0, 'fp': 0, 'fn': len(gt_peaks)}
    if len(gt_peaks) == 0:
        return {'tp': 0, 'fp': len(pred_peaks), 'fn': 0}
    tp = 0
    matched_gt = set()
    for pred_idx in pred_peaks:
        candidates = np.where(np.abs(gt_peaks - pred_idx) <= tolerance_samples)[0]
        for c in candidates:
            if c not in matched_gt:
                matched_gt.add(c)
                tp += 1
                break
    return {'tp': tp, 'fp': len(pred_peaks) - tp, 'fn': len(gt_peaks) - tp}

def calculate_heart_rate_from_peaks(peaks, fs=360):
    if peaks is None or len(peaks) < 2:
        return np.nan
    rr = np.diff(peaks) / fs
    avg_rr = np.mean(rr)
    return 60.0 / avg_rr if avg_rr > 0 else np.nan

def calculate_hrv_from_peaks(peaks, fs=360, metric='sdnn'):
    if peaks is None or len(peaks) < 3:
        return np.nan
    rr_ms = np.diff(peaks) / fs * 1000.0
    if metric == 'sdnn':
        return float(np.std(rr_ms, ddof=1))
    elif metric == 'rmssd':
        return float(np.sqrt(np.mean(np.diff(rr_ms) ** 2)))
    return np.nan

def calculate_mape(predicted, actual, min_threshold=1.0):
    if np.isnan(predicted) or np.isnan(actual) or actual == 0:
        return np.nan
    if abs(actual) < min_threshold:
        return np.nan
    return abs((predicted - actual) / actual) * 100.0


# --------------- per-sample worker ---------------

def process_sample(args):
    i, sig, lbl, fs, tol = args
    gt_peaks = extract_gt_peaks_from_label(lbl, fs=fs)

    inference_start = time.time()
    pred_peaks = ptpp_detect(sig, fs=fs)
    inference_time = time.time() - inference_start

    binary_peaks = np.zeros(len(sig), dtype=np.float32)
    valid = pred_peaks[(pred_peaks >= 0) & (pred_peaks < len(sig))]
    if len(valid) > 0:
        binary_peaks[valid] = 1.0

    result = {
        'tp': 0, 'fp': 0, 'fn': 0,
        'hr_error': None, 'hrv_sdnn_error': None, 'hrv_rmssd_error': None,
        'hr_mape': None, 'hrv_sdnn_mape': None, 'hrv_rmssd_mape': None,
        'inference_time': inference_time, 'binary_peaks': binary_peaks
    }

    if len(pred_peaks) > 0:
        m = calculate_detection_metrics(pred_peaks, gt_peaks, tol)
        result.update({'tp': m['tp'], 'fp': m['fp'], 'fn': m['fn']})
        pred_hr = calculate_heart_rate_from_peaks(pred_peaks, fs=fs)
        gt_hr = calculate_heart_rate_from_peaks(gt_peaks, fs=fs)
        if not (np.isnan(pred_hr) or np.isnan(gt_hr)):
            result['hr_error'] = abs(pred_hr - gt_hr)
            result['hr_mape'] = calculate_mape(pred_hr, gt_hr, min_threshold=30.0)
        for k in ['sdnn', 'rmssd']:
            p_hrv = calculate_hrv_from_peaks(pred_peaks, fs=fs, metric=k)
            g_hrv = calculate_hrv_from_peaks(gt_peaks, fs=fs, metric=k)
            try:
                p_hrv = float(p_hrv) if p_hrv is not None else np.nan
                g_hrv = float(g_hrv) if g_hrv is not None else np.nan
            except (ValueError, TypeError):
                p_hrv, g_hrv = np.nan, np.nan
            if not (np.isnan(p_hrv) or np.isnan(g_hrv)):
                result[f'hrv_{k}_error'] = abs(p_hrv - g_hrv)
                result[f'hrv_{k}_mape'] = calculate_mape(p_hrv, g_hrv, min_threshold=5.0)
    return result


# --------------- main ---------------

def main():
    fs = 360
    tolerance_samples = int(0.05 * fs)  # 50 ms (same as nk2 script)
    x_test_path = '/path/to/workspace/project-BCG-LLM/combined_data/combined_splitted_data/ECG/total_X_test.npy'
    y_test_path = '/path/to/workspace/project-BCG-LLM/combined_data/combined_splitted_data/ECG/total_y_test.npy'

    print("Loading held-out test data...")
    X_test = np.load(x_test_path)
    y_test = np.load(y_test_path)
    assert X_test.shape[0] == y_test.shape[0]
    assert X_test.shape[1] == y_test.shape[1]

    print(f"Test dataset: {X_test.shape[0]} samples, window: {X_test.shape[1]}")

    n_cpus = cpu_count()
    print(f"Using Pan-Tompkins++ detector")
    print(f"Tolerance: {tolerance_samples} samples ({tolerance_samples / fs * 1000:.0f} ms)")
    print(f"Parallelizing across {n_cpus} CPUs")

    args_list = [(i, X_test[i], y_test[i], fs, tolerance_samples) for i in range(X_test.shape[0])]

    print(f'\n{"="*80}')
    print('PAN-TOMPKINS++ HELD-OUT TEST RESULTS')
    print(f'{"="*80}')

    with Pool(n_cpus) as pool:
        results = list(tqdm(pool.imap(process_sample, args_list),
                            total=len(args_list), desc='Evaluating'))

    tp = sum(r['tp'] for r in results)
    fp = sum(r['fp'] for r in results)
    fn = sum(r['fn'] for r in results)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    hr_errors = [r['hr_error'] for r in results if r['hr_error'] is not None]
    hr_mapes = [r['hr_mape'] for r in results if r['hr_mape'] is not None and not np.isnan(r['hr_mape'])]
    sdnn_errors = [r['hrv_sdnn_error'] for r in results if r['hrv_sdnn_error'] is not None]
    sdnn_mapes = [r['hrv_sdnn_mape'] for r in results if r['hrv_sdnn_mape'] is not None and not np.isnan(r['hrv_sdnn_mape'])]
    rmssd_errors = [r['hrv_rmssd_error'] for r in results if r['hrv_rmssd_error'] is not None]
    rmssd_mapes = [r['hrv_rmssd_mape'] for r in results if r['hrv_rmssd_mape'] is not None and not np.isnan(r['hrv_rmssd_mape'])]
    inf_times = [r['inference_time'] for r in results]

    hr_mae = float(np.mean(hr_errors)) if hr_errors else float('nan')
    hr_mape = float(np.mean(hr_mapes)) if hr_mapes else float('nan')
    sdnn_mae = float(np.mean(sdnn_errors)) if sdnn_errors else float('nan')
    sdnn_mape = float(np.mean(sdnn_mapes)) if sdnn_mapes else float('nan')
    rmssd_mae = float(np.mean(rmssd_errors)) if rmssd_errors else float('nan')
    rmssd_mape = float(np.mean(rmssd_mapes)) if rmssd_mapes else float('nan')
    total_time = sum(inf_times)
    avg_time = total_time / len(results) if results else 0.0
    throughput = 1.0 / avg_time if avg_time > 0 else 0.0

    fmt = lambda v: f"{v:.2f}" if not np.isnan(v) else "NaN"
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Heart Rate MAE (BPM): {fmt(hr_mae)}")
    print(f"Heart Rate MAPE (%): {fmt(hr_mape)}")
    print(f"HRV SDNN MAE (ms): {fmt(sdnn_mae)}")
    print(f"HRV SDNN MAPE (%): {fmt(sdnn_mape)}")
    print(f"HRV RMSSD MAE (ms): {fmt(rmssd_mae)}")
    print(f"HRV RMSSD MAPE (%): {fmt(rmssd_mape)}")
    print(f"\nTotal Inference Time: {total_time:.2f} seconds")
    print(f"Avg Inference Time per Sample: {avg_time:.6f} seconds")
    print(f"Inference Throughput: {throughput:.2f} samples/sec")
    print(f"\nTotal: TP={tp}, FP={fp}, FN={fn}")

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MITBIH_predicted_peaks_ptpp.npy')
    predicted_peaks_array = np.stack([r['binary_peaks'] for r in results]).astype(np.float32)
    np.save(output_path, predicted_peaks_array)
    print(f'\nPredicted peaks saved to: {output_path}')
    print(f'Shape: {predicted_peaks_array.shape}')


if __name__ == '__main__':
    main()
