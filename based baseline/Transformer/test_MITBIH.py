#!/usr/bin/env python3
"""
Test Transformer ECG R-peak detection model on held-out test set and report metrics.

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

# -----------------------------
# Model definition (must match training)
# -----------------------------

class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class ECGRPeakTransformer(nn.Module):
    """Standard Transformer for per-timestep R-peak probability estimation."""

    def __init__(self, input_dim=1, d_model=128, nhead=8, num_layers=6, dim_feedforward=512, max_len=1000, dropout=0.1):
        super(ECGRPeakTransformer, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_projection(x)
        x = x.transpose(0, 1)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)
        x = self.transformer_encoder(x)
        return self.output_projection(x)

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

# -----------------------------
# Main testing routine
# -----------------------------

def main():
	# Config
	fs = 360
	window_length = 1000
	tolerance_samples = int(0.01 * fs)
	model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'transformer_mitbih_trained_model.pt')
	x_test_path = '/path/to/workspace/project-BCG-LLM/combined_data/combined_splitted_data/ECG/total_X_test.npy'
	y_test_path = '/path/to/workspace/project-BCG-LLM/combined_data/combined_splitted_data/ECG/total_y_test.npy'

	# Load data
	X_test = np.load(x_test_path)
	y_test = np.load(y_test_path)
	assert X_test.shape[0] == y_test.shape[0], 'Mismatched number of samples'
	assert X_test.shape[1] == window_length, f'Expected window length {window_length}'

	# Device and model
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	model = ECGRPeakTransformer(
		input_dim=1,
		d_model=128,
		nhead=8,
		num_layers=6,
		dim_feedforward=512,
		max_len=window_length,
		dropout=0.1,
	).to(device)
	checkpoint = torch.load(model_path, map_location=device)
	state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
	model.load_state_dict(state_dict)
	model.eval()

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

		# Inference
		with torch.no_grad():
			inp = torch.from_numpy(signal_np).float().unsqueeze(0).unsqueeze(-1).to(device)  # (1, L, 1)
			pred = model(inp).squeeze().detach().cpu().numpy()  # (L,)

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
