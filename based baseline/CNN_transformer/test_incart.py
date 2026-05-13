#!/usr/bin/env python3
"""
Test FR-Net (CNN-Transformer) ECG R-peak detection model on held-out test set.

- Loads FR-Net model from the current benchmark directory
- Loads X_test.npy and y_test.npy from provided paths
- Runs inference, localizes peaks, computes HR MAE, precision, recall, F1
"""

import os
import numpy as np
import torch
import torch.nn as nn
from scipy import signal
from tqdm import tqdm
import math

# -----------------------------
# Model definition (FR-Net from Paper)
# -----------------------------

class ResidualBlock(nn.Module):
    """1D Residual block for ECG feature extraction"""
    def __init__(self, in_channels, out_channels, kernel_size=9):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.skip_bn = nn.BatchNorm1d(out_channels) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        identity = self.skip_bn(self.skip(x))
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class FRNet(nn.Module):
    """FR-Net: CNN-Transformer hybrid for ECG R-peak detection"""
    def __init__(self, in_channels=1, base_channels=64, num_transformer_layers=6, nhead=8):
        super().__init__()
        
        # CNN Encoder with residual blocks
        self.res_block1 = ResidualBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool1d(2)
        self.res_block2 = ResidualBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool1d(2)
        self.res_block3 = ResidualBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool1d(2)
        
        # Transformer Encoder
        self.pos_encoding = PositionalEncoding(base_channels * 4)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=base_channels * 4,
            nhead=nhead,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_transformer_layers)
        
        # Decoder with upsampling
        self.up1 = nn.Sequential(
            nn.Conv1d(base_channels * 4, base_channels * 2, 9, padding=4),
            nn.ReLU(),
            nn.Conv1d(base_channels * 2, base_channels * 2, 9, padding=4),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        )
        
        self.up2 = nn.Sequential(
            nn.Conv1d(base_channels * 6, base_channels, 9, padding=4),
            nn.ReLU(),
            nn.Conv1d(base_channels, base_channels, 9, padding=4),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        )
        
        self.up3 = nn.Sequential(
            nn.Conv1d(base_channels * 3, base_channels // 2, 9, padding=4),
            nn.ReLU(),
            nn.Conv1d(base_channels // 2, base_channels // 2, 9, padding=4),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        )
        
        self.final = nn.Sequential(
            nn.Conv1d(base_channels + base_channels // 2, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: (batch, seq_len, channels) -> (batch, channels, seq_len)
        if x.dim() == 3:
            x = x.transpose(1, 2)
        
        # CNN encoder with skip connections
        x1 = self.res_block1(x)
        x = self.pool1(x1)
        
        x2 = self.res_block2(x)
        x = self.pool2(x2)
        
        x3 = self.res_block3(x)
        x = self.pool3(x3)
        
        # CNN output
        f1 = x
        
        # Transformer
        x = x.transpose(1, 2)
        x = self.pos_encoding(x)
        f2 = self.transformer(x)
        f2 = f2.transpose(1, 2)
        
        # Combine CNN and Transformer features
        fe = f1 + f2
        
        # Decoder with skip connections
        x = self.up1(fe)
        x = torch.cat([x, x3], dim=1)
        
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        
        x = self.final(x)
        
        return x.transpose(1, 2)  # (batch, seq_len, 1)


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
    fs = 257
    window_length = 1000
    tolerance_samples = int(0.03 * fs)
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frnet_incart_trained_model.pt')
    x_test_path = '/path/to/workspace/project-BCG-LLM/combined_data/combined_splitted_data/ECG_incart/total_X_test.npy'
    y_test_path = '/path/to/workspace/project-BCG-LLM/combined_data/combined_splitted_data/ECG_incart/total_y_test.npy'

    print("="*80)
    print("FR-NET (CNN-TRANSFORMER) ECG R-PEAK DETECTION - TESTING")
    print("="*80)
    
    # Load data
    print(f"\nLoading test data...")
    X_test = np.load(x_test_path)
    y_test = np.load(y_test_path)
    print(f"Test data: X={X_test.shape}, y={y_test.shape}")
    
    assert X_test.shape[0] == y_test.shape[0], 'Mismatched number of samples'
    assert X_test.shape[1] == window_length, f'Expected window length {window_length}'

    # Device and model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = FRNet(in_channels=1, base_channels=64).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load weights
    if not os.path.exists(model_path):
        print(f"WARNING: Model file not found at {model_path}")
        print("Please train the model first!")
        return
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded from: {model_path}\n")

    # Accumulators
    all_tp = 0
    all_fp = 0
    all_fn = 0
    hr_errors = []
    hrv_sdnn_errors = []
    hrv_rmssd_errors = []

    # Iterate samples
    print("Running inference on test set...")
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

    print('\n' + "="*80)
    print('FR-NET TEST RESULTS')
    print("="*80)
    print(f'Total Test Samples:    {X_test.shape[0]}')
    print(f'True Positives (TP):   {all_tp}')
    print(f'False Positives (FP):  {all_fp}')
    print(f'False Negatives (FN):  {all_fn}')
    print('-'*80)
    print(f'Heart Rate MAE (BPM):  {hr_mae:.2f}' if not np.isnan(hr_mae) else 'Heart Rate MAE (BPM): NaN')
    print(f'HRV SDNN MAE (ms):     {hrv_sdnn_mae:.2f}' if not np.isnan(hrv_sdnn_mae) else 'HRV SDNN MAE (ms): NaN')
    print(f'HRV RMSSD MAE (ms):    {hrv_rmssd_mae:.2f}' if not np.isnan(hrv_rmssd_mae) else 'HRV RMSSD MAE (ms): NaN')
    print(f'Precision:             {overall_precision:.4f}')
    print(f'Recall:                {overall_recall:.4f}')
    print(f'F1-Score:              {overall_f1:.4f}')
    print("="*80)


if __name__ == '__main__':
    main()