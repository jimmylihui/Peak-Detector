#!/usr/bin/env python3
"""
Test traditional Pan–Tompkins ECG R-peak detection on held-out test set and report metrics.

- Uses classical Pan–Tompkins pipeline (bandpass → derivative → squaring → MWI → threshold)
- Loads X_test.npy and y_test.npy from provided paths
- Detects R-peaks, computes HR MAE, precision, recall, F1
"""

import os
import numpy as np
from scipy import signal
from tqdm import tqdm

"""
Pan–Tompkins utilities
"""

def butter_bandpass(signal_array: np.ndarray, fs: int) -> np.ndarray:
	"""Bandpass filter around QRS (roughly 5-15 Hz for fs≈360)."""
	low_cut_hz = 1.0
	high_cut_hz = 20.0
	# Ensure valid normalized frequencies
	low = max(low_cut_hz, 0.5)
	high = min(high_cut_hz, 0.49 * fs)
	b, a = signal.butter(3, [low, high], btype='band', fs=fs)
	return signal.filtfilt(b, a, signal_array)

def five_point_derivative(signal_array: np.ndarray, fs: int) -> np.ndarray:
	"""Five-point derivative approximation used in Pan–Tompkins."""
	# Kernel corresponds to (1/8T)[-1 -2 0 2 1]
	kernel = np.array([-1, -2, 0, 2, 1], dtype=np.float64) * (fs / 8.0)
	deriv = np.convolve(signal_array, kernel, mode='same')
	return deriv

def moving_window_integration(signal_array: np.ndarray, fs: int, window_ms: float = 150.0) -> np.ndarray:
	"""Moving window integration over specified window length in ms."""
	window_samples = int(max(1, round(window_ms * 1e-3 * fs)))
	window = np.ones(window_samples, dtype=np.float64) / window_samples
	return np.convolve(signal_array, window, mode='same')

def pan_tompkins_detect(ecg: np.ndarray, fs: int = 360) -> np.ndarray:
	"""Run a Pan–Tompkins-like pipeline and return R-peak indices."""
	ecg = np.ascontiguousarray(ecg).astype(np.float64)
	# 1) Bandpass
	band = butter_bandpass(ecg, fs)
	# 2) Derivative
	deriv = five_point_derivative(band, fs)
	# 3) Squaring
	sq = deriv ** 2
	# 4) Moving window integration (~150 ms)
	mwi = moving_window_integration(sq, fs, window_ms=150.0)
	# 5) Adaptive thresholding + refractory
	refractory = int(0.2 * fs)
	mean_val = np.mean(mwi)
	std_val = np.std(mwi)
	height = max(mean_val + 1.0 * std_val, 0.0)
	peaks, _ = signal.find_peaks(mwi, height=height, distance=refractory)
	# Fallback to prominence if none
	if len(peaks) == 0:
		prom = 0.5 * std_val if std_val > 0 else None
		peaks, _ = signal.find_peaks(mwi, prominence=prom, distance=refractory)
	return peaks.astype(int)

# -----------------------------
# Utility functions
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

# -----------------------------
# Main testing routine
# -----------------------------

def main():
	# Config
	fs = 125
	window_length = 1000
	tolerance_samples = int(0.03 * fs)
	x_test_path = '/path/to/workspace/project-BCG-LLM/combined_data/combined_splitted_data/PPG/manually_clean_X_test.npy'
	y_test_path = '/path/to/workspace/project-BCG-LLM/combined_data/combined_splitted_data/PPG/manually_clean_y_test.npy'

	# Load data
	X_test = np.load(x_test_path)
	y_test = np.load(y_test_path)
	assert X_test.shape[0] == y_test.shape[0], 'Mismatched number of samples'
	assert X_test.shape[1] == window_length, f'Expected window length {window_length}'

	# Accumulators
	all_tp = 0
	all_fp = 0
	all_fn = 0
	hr_errors = []

	# Iterate samples
	for i in tqdm(range(X_test.shape[0]), desc='Testing'):
		signal_np = preprocess_signal(X_test[i])
		label_np = y_test[i]

		# GT peaks from label
		gt_peaks = extract_gt_peaks_from_label(label_np, fs=fs)

		# Pan–Tompkins detection
		pred_peaks = pan_tompkins_detect(signal_np, fs=fs)

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

	# Aggregate metrics
	overall_precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
	overall_recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
	overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
	hr_mae = float(np.mean(hr_errors)) if len(hr_errors) > 0 else float('nan')

	print('\n=== Test Metrics ===')
	print(f'Heart Rate MAE (BPM): {hr_mae:.2f}' if not np.isnan(hr_mae) else 'Heart Rate MAE (BPM): NaN')
	print(f'Precision: {overall_precision:.4f}')
	print(f'Recall:    {overall_recall:.4f}')
	print(f'F1-Score:  {overall_f1:.4f}')

if __name__ == '__main__':
	main()
