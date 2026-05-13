#!/usr/bin/env python3
"""
Pan-Tompkins++ benchmark on all datasets with absolute and relative tolerance.

Algorithms:
  - Pan-Tompkins++ (from /path/to/workspace/Pan-Tompkins-Plus-Plus)
  - Original Pan-Tompkins (Fabrizio1994)

Tolerance modes:
  - Absolute: fixed sample window derived from 50 ms at the dataset's fs
  - Relative: 5% of median GT RR interval per sample (min 5 samples)
"""

import os
import sys
import json
import time
import numpy as np
from scipy import signal
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

sys.path.insert(0, '/path/to/workspace/Pan-Tompkins-Plus-Plus')
from algos.pan_tompkins_plus_plus import Pan_Tompkins_Plus_Plus
from algos.pan_tompkins_original import Pan_Tompkins_Original

DATA_ROOT = '/path/to/workspace/project-BCG-LLM/combined_data/combined_splitted_data'

DATASETS = [
    {
        'name': 'MIT-BIH (ECG)',
        'key': 'ECG',
        'fs': 360,
        'x_test': f'{DATA_ROOT}/ECG/total_X_test.npy',
        'y_test': f'{DATA_ROOT}/ECG/total_y_test.npy',
    },
    {
        'name': 'INCART (ECG)',
        'key': 'ECG_incart',
        'fs': 257,
        'x_test': f'{DATA_ROOT}/ECG_incart/total_X_test.npy',
        'y_test': f'{DATA_ROOT}/ECG_incart/total_y_test.npy',
    },
    {
        'name': 'BIDMC (PPG)',
        'key': 'PPG',
        'fs': 125,
        'x_test': f'{DATA_ROOT}/PPG/processed/X_test.npy',
        'y_test': f'{DATA_ROOT}/PPG/processed/Y_test.npy',
    },
    {
        'name': 'CapnoBase (PPG)',
        'key': 'PPG_capnobase',
        'fs': 300,
        'x_test': f'{DATA_ROOT}/PPG_capnobase/processed/X_test.npy',
        'y_test': f'{DATA_ROOT}/PPG_capnobase/processed/Y_test_gauss.npy',
    },
    {
        'name': 'Kansas (BCG)',
        'key': 'BCG',
        'fs': 100,
        'x_test': f'{DATA_ROOT}/BCG/X_test.npy',
        'y_test': f'{DATA_ROOT}/BCG/y_test.npy',
    },
    {
        'name': 'Arrhythmia (BCG)',
        'key': 'BCG_Arrhythmia',
        'fs': 100,
        'x_test': f'{DATA_ROOT}/BCG_Arrhythmia/X_test.npy',
        'y_test': f'{DATA_ROOT}/BCG_Arrhythmia/y_test.npy',
    },
    {
        'name': 'Hospital (BCG)',
        'key': 'BCG_hospital',
        'fs': 100,
        'x_test': f'{DATA_ROOT}/BCG_hospital/X_test.npy',
        'y_test': f'{DATA_ROOT}/BCG_hospital/y_test.npy',
    },
]

ABSOLUTE_TOL_SEC = 0.05     # 50 ms
RELATIVE_TOL_FRAC = 0.05    # 5% of median RR
RELATIVE_TOL_MIN = 5        # minimum samples for relative tolerance


# ---- Detectors ----

def detect_ptpp(ecg, fs):
    ecg = np.ascontiguousarray(ecg).astype(np.float64)
    det = Pan_Tompkins_Plus_Plus()
    try:
        peaks = det.rpeak_detection(ecg, fs)
    except Exception:
        return np.array([], dtype=int)
    if peaks is None or len(peaks) == 0:
        return np.array([], dtype=int)
    return _refractory_filter(np.asarray(peaks, dtype=int), fs)


def detect_pt_orig(ecg, fs):
    ecg = np.ascontiguousarray(ecg).astype(np.float64)
    det = Pan_Tompkins_Original()
    try:
        peaks = det.rpeak_detection(ecg, fs)
    except Exception:
        return np.array([], dtype=int)
    if peaks is None or len(peaks) == 0:
        return np.array([], dtype=int)
    return _refractory_filter(np.asarray(peaks, dtype=int), fs)


def _refractory_filter(peaks, fs):
    """Remove peaks closer than 200 ms (from reference main.py)."""
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


# ---- GT extraction ----

def extract_gt_peaks(gt_label, fs):
    refractory = int(fs * 0.2)
    peaks, _ = signal.find_peaks(gt_label, height=0.5, distance=refractory)
    return peaks.astype(int)


# ---- Metrics ----

def match_peaks_absolute(pred, gt, tol):
    if len(pred) == 0 and len(gt) == 0:
        return 0, 0, 0
    if len(pred) == 0:
        return 0, 0, len(gt)
    if len(gt) == 0:
        return 0, len(pred), 0
    tp = 0
    matched = set()
    for p in pred:
        candidates = np.where(np.abs(gt - p) <= tol)[0]
        for c in candidates:
            if c not in matched:
                matched.add(c)
                tp += 1
                break
    return tp, len(pred) - tp, len(gt) - tp


def match_peaks_relative(pred, gt, rel_frac, min_samples):
    if len(pred) == 0 and len(gt) == 0:
        return 0, 0, 0
    if len(pred) == 0:
        return 0, 0, len(gt)
    if len(gt) == 0:
        return 0, len(pred), 0

    if len(gt) >= 2:
        rr = np.diff(gt)
        median_rr = np.median(rr)
        tol = max(int(rel_frac * median_rr), min_samples)
    else:
        tol = max(min_samples, 10)

    return match_peaks_absolute(pred, gt, tol)


def hr_from_peaks(peaks, fs):
    if peaks is None or len(peaks) < 2:
        return np.nan
    rr = np.diff(peaks) / fs
    avg = np.mean(rr)
    return 60.0 / avg if avg > 0 else np.nan


def hrv_sdnn(peaks, fs):
    if peaks is None or len(peaks) < 3:
        return np.nan
    rr_ms = np.diff(peaks) / fs * 1000.0
    return float(np.std(rr_ms, ddof=1))


def hrv_rmssd(peaks, fs):
    if peaks is None or len(peaks) < 3:
        return np.nan
    rr_ms = np.diff(peaks) / fs * 1000.0
    return float(np.sqrt(np.mean(np.diff(rr_ms) ** 2)))


def safe_mape(pred_val, gt_val, min_thresh=1.0):
    if np.isnan(pred_val) or np.isnan(gt_val) or gt_val == 0 or abs(gt_val) < min_thresh:
        return np.nan
    return abs((pred_val - gt_val) / gt_val) * 100.0


# ---- Per-sample worker ----

def _worker(args):
    sig, lbl, fs, abs_tol, det_name = args
    gt = extract_gt_peaks(lbl, fs)

    t0 = time.time()
    if det_name == 'ptpp':
        pred = detect_ptpp(sig, fs)
    else:
        pred = detect_pt_orig(sig, fs)
    inf_time = time.time() - t0

    valid = pred[(pred >= 0) & (pred < len(sig))]
    binary = np.zeros(len(sig), dtype=np.float32)
    if len(valid) > 0:
        binary[valid] = 1.0

    tp_abs, fp_abs, fn_abs = match_peaks_absolute(pred, gt, abs_tol)
    tp_rel, fp_rel, fn_rel = match_peaks_relative(pred, gt, RELATIVE_TOL_FRAC, RELATIVE_TOL_MIN)

    pred_hr = hr_from_peaks(pred, fs)
    gt_hr = hr_from_peaks(gt, fs)
    hr_err = abs(pred_hr - gt_hr) if not (np.isnan(pred_hr) or np.isnan(gt_hr)) else None
    hr_mape = safe_mape(pred_hr, gt_hr, 30.0) if hr_err is not None else None

    pred_sdnn = hrv_sdnn(pred, fs)
    gt_sdnn = hrv_sdnn(gt, fs)
    sdnn_err = abs(pred_sdnn - gt_sdnn) if not (np.isnan(pred_sdnn) or np.isnan(gt_sdnn)) else None
    sdnn_mape = safe_mape(pred_sdnn, gt_sdnn, 5.0) if sdnn_err is not None else None

    pred_rmssd = hrv_rmssd(pred, fs)
    gt_rmssd = hrv_rmssd(gt, fs)
    rmssd_err = abs(pred_rmssd - gt_rmssd) if not (np.isnan(pred_rmssd) or np.isnan(gt_rmssd)) else None
    rmssd_mape = safe_mape(pred_rmssd, gt_rmssd, 5.0) if rmssd_err is not None else None

    return {
        'tp_abs': tp_abs, 'fp_abs': fp_abs, 'fn_abs': fn_abs,
        'tp_rel': tp_rel, 'fp_rel': fp_rel, 'fn_rel': fn_rel,
        'hr_err': hr_err, 'hr_mape': hr_mape,
        'sdnn_err': sdnn_err, 'sdnn_mape': sdnn_mape,
        'rmssd_err': rmssd_err, 'rmssd_mape': rmssd_mape,
        'inf_time': inf_time, 'binary': binary,
    }


def aggregate(results):
    def _prf(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f

    tp_a = sum(r['tp_abs'] for r in results)
    fp_a = sum(r['fp_abs'] for r in results)
    fn_a = sum(r['fn_abs'] for r in results)
    tp_r = sum(r['tp_rel'] for r in results)
    fp_r = sum(r['fp_rel'] for r in results)
    fn_r = sum(r['fn_rel'] for r in results)

    pa, ra, fa = _prf(tp_a, fp_a, fn_a)
    pr, rr, fr = _prf(tp_r, fp_r, fn_r)

    def _mean_valid(key):
        vals = [r[key] for r in results if r[key] is not None and not np.isnan(r[key])]
        return float(np.mean(vals)) if vals else float('nan')

    return {
        'abs': {'precision': pa, 'recall': ra, 'f1': fa, 'tp': tp_a, 'fp': fp_a, 'fn': fn_a},
        'rel': {'precision': pr, 'recall': rr, 'f1': fr, 'tp': tp_r, 'fp': fp_r, 'fn': fn_r},
        'hr_mae': _mean_valid('hr_err'),
        'hr_mape': _mean_valid('hr_mape'),
        'sdnn_mae': _mean_valid('sdnn_err'),
        'sdnn_mape': _mean_valid('sdnn_mape'),
        'rmssd_mae': _mean_valid('rmssd_err'),
        'rmssd_mape': _mean_valid('rmssd_mape'),
        'total_time': sum(r['inf_time'] for r in results),
        'avg_time': sum(r['inf_time'] for r in results) / len(results) if results else 0,
        'n_samples': len(results),
    }


# ---- Main ----

def run_dataset(ds, det_name, n_cpus):
    x_path, y_path, fs = ds['x_test'], ds['y_test'], ds['fs']
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        print(f"  SKIP: files not found for {ds['name']}")
        return None

    X = np.load(x_path)
    Y = np.load(y_path)
    assert X.shape[0] == Y.shape[0]

    abs_tol = max(1, int(ABSOLUTE_TOL_SEC * fs))

    args = [(X[i], Y[i], fs, abs_tol, det_name) for i in range(X.shape[0])]
    with Pool(n_cpus) as pool:
        results = list(tqdm(pool.imap(_worker, args), total=len(args),
                            desc=f"  {ds['name']}"))

    agg = aggregate(results)
    agg['abs_tol_samples'] = abs_tol
    agg['abs_tol_ms'] = ABSOLUTE_TOL_SEC * 1000

    binaries = np.stack([r['binary'] for r in results]).astype(np.float32)

    return agg, binaries


def fmt(v):
    return f"{v:.2f}" if not np.isnan(v) else "N/A"


def fmt4(v):
    return f"{v:.4f}"


def main():
    n_cpus = cpu_count()
    all_results = {}

    detectors = [
        ('pt_orig', 'Pan-Tompkins'),
    ]

    for det_key, det_label in detectors:
        print(f"\n{'#'*80}")
        print(f"  Detector: {det_label}")
        print(f"{'#'*80}")

        for ds in DATASETS:
            print(f"\n  Dataset: {ds['name']} (fs={ds['fs']})")
            out = run_dataset(ds, det_key, n_cpus)
            if out is None:
                continue
            agg, binaries = out

            result_key = f"{det_key}__{ds['key']}"
            all_results[result_key] = {
                'detector': det_label,
                'dataset': ds['name'],
                'dataset_key': ds['key'],
                'fs': ds['fs'],
                'n_samples': agg['n_samples'],
                **agg,
            }

            npy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    f"predicted_peaks_{det_key}_{ds['key']}.npy")
            np.save(npy_path, binaries)

            a, r = agg['abs'], agg['rel']
            print(f"    Absolute tol ({agg['abs_tol_ms']:.0f}ms = {agg['abs_tol_samples']} samples):")
            print(f"      Precision={fmt4(a['precision'])} Recall={fmt4(a['recall'])} F1={fmt4(a['f1'])}")
            print(f"      TP={a['tp']} FP={a['fp']} FN={a['fn']}")
            print(f"    Relative tol ({RELATIVE_TOL_FRAC*100:.0f}% of median RR, min {RELATIVE_TOL_MIN}):")
            print(f"      Precision={fmt4(r['precision'])} Recall={fmt4(r['recall'])} F1={fmt4(r['f1'])}")
            print(f"      TP={r['tp']} FP={r['fp']} FN={r['fn']}")
            print(f"    HR MAE={fmt(agg['hr_mae'])} BPM  MAPE={fmt(agg['hr_mape'])}%")
            print(f"    HRV SDNN MAE={fmt(agg['sdnn_mae'])} ms  MAPE={fmt(agg['sdnn_mape'])}%")
            print(f"    HRV RMSSD MAE={fmt(agg['rmssd_mae'])} ms  MAPE={fmt(agg['rmssd_mape'])}%")
            print(f"    Inference: {agg['total_time']:.2f}s total, {agg['avg_time']*1000:.3f}ms/sample")

    # Write result.md
    write_result_md(all_results)

    # Save raw JSON
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nJSON results saved to: {json_path}")


def write_result_md(all_results):
    md_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'result.md')
    lines = []
    lines.append("# Pan-Tompkins Benchmark Results\n")
    lines.append(f"**Absolute tolerance:** {ABSOLUTE_TOL_SEC*1000:.0f} ms fixed window\n")
    lines.append(f"**Relative tolerance:** {RELATIVE_TOL_FRAC*100:.0f}% of median GT RR interval (min {RELATIVE_TOL_MIN} samples)\n")
    lines.append("")

    for det_key, det_label in [('pt_orig', 'Pan-Tompkins')]:
        lines.append(f"## {det_label}\n")

        # Absolute tolerance table
        lines.append("### Absolute Tolerance (50 ms)\n")
        lines.append("| Dataset | Samples | Precision | Recall | F1 | HR MAE (BPM) | HR MAPE (%) | SDNN MAE (ms) | SDNN MAPE (%) | RMSSD MAE (ms) | RMSSD MAPE (%) | Throughput |")
        lines.append("|---------|---------|-----------|--------|----|-------------|-------------|---------------|---------------|----------------|----------------|------------|")

        for ds in DATASETS:
            key = f"{det_key}__{ds['key']}"
            if key not in all_results:
                continue
            r = all_results[key]
            a = r['abs']
            tput = f"{1.0/r['avg_time']:.0f}/s" if r['avg_time'] > 0 else "N/A"
            lines.append(
                f"| {ds['name']} | {r['n_samples']} "
                f"| {a['precision']:.4f} | {a['recall']:.4f} | {a['f1']:.4f} "
                f"| {fmt(r['hr_mae'])} | {fmt(r['hr_mape'])} "
                f"| {fmt(r['sdnn_mae'])} | {fmt(r['sdnn_mape'])} "
                f"| {fmt(r['rmssd_mae'])} | {fmt(r['rmssd_mape'])} "
                f"| {tput} |"
            )
        lines.append("")

        # Relative tolerance table
        lines.append("### Relative Tolerance (5% of median RR)\n")
        lines.append("| Dataset | Samples | Precision | Recall | F1 | HR MAE (BPM) | HR MAPE (%) | SDNN MAE (ms) | SDNN MAPE (%) | RMSSD MAE (ms) | RMSSD MAPE (%) | Throughput |")
        lines.append("|---------|---------|-----------|--------|----|-------------|-------------|---------------|---------------|----------------|----------------|------------|")

        for ds in DATASETS:
            key = f"{det_key}__{ds['key']}"
            if key not in all_results:
                continue
            r = all_results[key]
            rel = r['rel']
            tput = f"{1.0/r['avg_time']:.0f}/s" if r['avg_time'] > 0 else "N/A"
            lines.append(
                f"| {ds['name']} | {r['n_samples']} "
                f"| {rel['precision']:.4f} | {rel['recall']:.4f} | {rel['f1']:.4f} "
                f"| {fmt(r['hr_mae'])} | {fmt(r['hr_mape'])} "
                f"| {fmt(r['sdnn_mae'])} | {fmt(r['sdnn_mape'])} "
                f"| {fmt(r['rmssd_mae'])} | {fmt(r['rmssd_mape'])} "
                f"| {tput} |"
            )
        lines.append("")

    md_content = "\n".join(lines)
    with open(md_path, 'w') as f:
        f.write(md_content)
    print(f"\nResults written to: {md_path}")


if __name__ == '__main__':
    main()
