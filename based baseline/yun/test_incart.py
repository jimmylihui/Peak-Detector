#!/usr/bin/env python3
"""
Test UNet++ ECG R-peak detection model on held-out test set and report metrics.

- Loads model from the current benchmark directory
- Loads X_test.npy and y_test.npy from provided paths
- Runs inference, localizes peaks, computes HR MAE, precision, recall, F1
"""

import os
import numpy as np
import torch
import torch.nn as nn
from scipy import signal
from tqdm import tqdm
import sys

# -----------------------------
# Model definition (must match training)
# -----------------------------

class ConvBlock(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=32):
		super(ConvBlock, self).__init__()
		self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same')
		self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding='same')
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		x = self.relu(self.conv1(x))
		x = self.relu(self.conv2(x))
		return x

class UNetPlusPlus1D(nn.Module):
	def __init__(self, input_channels=1, base_filters=16):
		super(UNetPlusPlus1D, self).__init__()
		self.conv0_0 = ConvBlock(input_channels, base_filters)
		self.conv1_0 = ConvBlock(base_filters, base_filters * 2)
		self.conv2_0 = ConvBlock(base_filters * 2, base_filters * 4)
		self.conv3_0 = ConvBlock(base_filters * 4, base_filters * 8)

		self.conv0_1 = ConvBlock(base_filters + base_filters, base_filters)
		self.conv1_1 = ConvBlock(base_filters * 2 + base_filters * 2, base_filters * 2)
		self.conv2_1 = ConvBlock(base_filters * 4 + base_filters * 4, base_filters * 4)

		self.conv0_2 = ConvBlock(base_filters + base_filters + base_filters, base_filters)
		self.conv1_2 = ConvBlock(base_filters * 2 + base_filters * 2 + base_filters * 2, base_filters * 2)

		self.conv0_3 = ConvBlock(base_filters + base_filters + base_filters + base_filters, base_filters)

		self.pool = nn.MaxPool1d(2)
		self.up = nn.Upsample(scale_factor=2, mode='nearest')

		self.up_conv3_1 = nn.Conv1d(base_filters * 8, base_filters * 4, 2, padding=1)
		self.up_conv2_1 = nn.Conv1d(base_filters * 4, base_filters * 2, 2, padding=1)
		self.up_conv1_1 = nn.Conv1d(base_filters * 2, base_filters, 2, padding=1)
		self.up_conv2_2 = nn.Conv1d(base_filters * 4, base_filters * 2, 2, padding=1)
		self.up_conv1_2 = nn.Conv1d(base_filters * 2, base_filters, 2, padding=1)
		self.up_conv1_3 = nn.Conv1d(base_filters * 2, base_filters, 2, padding=1)

		self.final = nn.Conv1d(base_filters, 1, 1)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		# x: (batch, length, channels)
		x = x.transpose(1, 2)
		x0_0 = self.conv0_0(x)
		x1_0 = self.conv1_0(self.pool(x0_0))
		x2_0 = self.conv2_0(self.pool(x1_0))
		x3_0 = self.conv3_0(self.pool(x2_0))

		x2_1_up = self.up_conv3_1(self.up(x3_0))
		if x2_1_up.size(2) != x2_0.size(2):
			x2_1_up = nn.functional.interpolate(x2_1_up, size=x2_0.size(2), mode='nearest')
		x2_1 = self.conv2_1(torch.cat([x2_0, x2_1_up], 1))

		x1_1_up = self.up_conv2_1(self.up(x2_0))
		if x1_1_up.size(2) != x1_0.size(2):
			x1_1_up = nn.functional.interpolate(x1_1_up, size=x1_0.size(2), mode='nearest')
		x1_1 = self.conv1_1(torch.cat([x1_0, x1_1_up], 1))

		x1_2_up = self.up_conv2_2(self.up(x2_1))
		if x1_2_up.size(2) != x1_0.size(2):
			x1_2_up = nn.functional.interpolate(x1_2_up, size=x1_0.size(2), mode='nearest')
		x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, x1_2_up], 1))

		x0_1_up = self.up_conv1_1(self.up(x1_0))
		if x0_1_up.size(2) != x0_0.size(2):
			x0_1_up = nn.functional.interpolate(x0_1_up, size=x0_0.size(2), mode='nearest')
		x0_1 = self.conv0_1(torch.cat([x0_0, x0_1_up], 1))

		x0_2_up = self.up_conv1_2(self.up(x1_1))
		if x0_2_up.size(2) != x0_0.size(2):
			x0_2_up = nn.functional.interpolate(x0_2_up, size=x0_0.size(2), mode='nearest')
		x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, x0_2_up], 1))

		x0_3_up = self.up_conv1_3(self.up(x1_2))
		if x0_3_up.size(2) != x0_0.size(2):
			x0_3_up = nn.functional.interpolate(x0_3_up, size=x0_0.size(2), mode='nearest')
		x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, x0_3_up], 1))

		output = self.sigmoid(self.final(x0_3))
		output = output.transpose(1, 2)
		return output

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
	#rr_intervals = rr_intervals[(rr_intervals >= 0.5) & (rr_intervals <= 2.0)]
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
	fs = 257
	window_length = 2048
	tolerance_samples = int(0.03 * fs)
	model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sepconv_incart_trained_model.pt')
	x_test_path = '/path/to/workspace/project-BCG-LLM/combined_data/combined_splitted_data/ECG_incart/total_X_test.npy'
	y_test_path = '/path/to/workspace/project-BCG-LLM/combined_data/combined_splitted_data/ECG_incart/total_y_test.npy'

	# Load data
	X_test = np.load(x_test_path)
	y_test = np.load(y_test_path)
	assert X_test.shape[0] == y_test.shape[0], 'Mismatched number of samples'
	# input length will be padded/cropped to window_length (2048)

	# Device and model (SepConv as in training benchmark)
	# Import SepConv utils
	sys.path.append('/path/to/workspace/project-BCG-LLM/PPG_peaks/Benchmark/yun/utils')
	from sep_conv import Sep_conv_detector
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	model = Sep_conv_detector(n_channel=2).to(device)
	checkpoint = torch.load(model_path, map_location=device)
	state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
	model.load_state_dict(state_dict)
	model.eval()

	# Accumulators
	all_tp = 0
	all_fp = 0
	all_fn = 0
	hr_errors = []
	hrv_sdnn_errors = []
	hrv_rmssd_errors = []

	# Iterate samples
	for i in tqdm(range(X_test.shape[0]), desc='Testing'):
		signal_np = preprocess_signal(X_test[i])
		label_np = y_test[i]

		# GT peaks from label
		gt_peaks = extract_gt_peaks_from_label(label_np, fs=fs)

		# Inference
		# Prepare two-channel input (raw + first difference), pad/crop to 2048
		def make_two_channel(arr: np.ndarray, target_len: int = 2048) -> np.ndarray:
			arr = np.ascontiguousarray(arr)
			if arr.ndim == 1:
				arr = arr[None, :]
			L = arr.shape[-1]
			if L < target_len:
				pad_width = target_len - L
				arr = np.pad(arr, ((0,0),(0,pad_width)), mode='edge')
			elif L > target_len:
				arr = arr[..., :target_len]
			diff = np.diff(arr, axis=-1)
			last = arr[..., -1:]
			diff_padded = np.concatenate([diff, last], axis=-1)
			two_ch = np.stack([arr, diff_padded], axis=1)  # (N, 2, L)
			return two_ch

		with torch.no_grad():
			two_ch_np = make_two_channel(signal_np, target_len=window_length)[0]  # (2, L)
			inp = torch.from_numpy(two_ch_np).float().unsqueeze(0).to(device)  # (1, 2, L)
			logits = model(inp)
			pred = torch.sigmoid(logits).squeeze().detach().cpu().numpy()  # (L,)

		# Localize predicted peaks
		pred_peaks = localize_predicted_peaks(pred, fs=fs)

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

	print('\n=== Test Metrics ===')
	print(f'Heart Rate MAE (BPM): {hr_mae:.2f}' if not np.isnan(hr_mae) else 'Heart Rate MAE (BPM): NaN')
	print(f'HRV SDNN MAE (ms): {hrv_sdnn_mae:.2f}' if not np.isnan(hrv_sdnn_mae) else 'HRV SDNN MAE (ms): NaN')
	print(f'HRV RMSSD MAE (ms): {hrv_rmssd_mae:.2f}' if not np.isnan(hrv_rmssd_mae) else 'HRV RMSSD MAE (ms): NaN')
	print(f'Precision: {overall_precision:.4f}')
	print(f'Recall:    {overall_recall:.4f}')
	print(f'F1-Score:  {overall_f1:.4f}')

if __name__ == '__main__':
	main()
