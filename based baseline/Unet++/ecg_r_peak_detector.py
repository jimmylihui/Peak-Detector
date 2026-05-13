#!/usr/bin/env python3
"""
ECG R-peak Detection using Trained UNet++ Model

This module provides a comprehensive ECG R-peak detection system using a pre-trained UNet++ model.
It can be used for both batch processing and real-time detection.

Features:
- Loads pre-trained UNet++ model for ECG R-peak detection
- Supports both single signal and batch processing
- Provides peak localization with adaptive thresholding
- Calculates heart rate and detection metrics
- Includes signal preprocessing and filtering
"""

import os
import numpy as np
import torch
import torch.nn as nn
from scipy import signal
from typing import Union, List, Tuple, Optional, Dict
import warnings

# -----------------------------
# Model Definition (must match training)
# -----------------------------

class ConvBlock(nn.Module):
    """Convolutional block with two 1D convolutions and ReLU activation."""
    
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
    """UNet++ 1D architecture for ECG R-peak detection."""
    
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
# ECG R-Peak Detector Class
# -----------------------------

class ECGRPeakDetector:
    """
    ECG R-peak detector using trained UNet++ model.
    
    This class provides a complete ECG R-peak detection pipeline including:
    - Signal preprocessing and filtering
    - Model inference
    - Peak localization
    - Heart rate calculation
    - Performance metrics
    """
    
    def __init__(self, 
                 model_path: str,
                 device: Optional[str] = None,
                 fs: int = 360,
                 window_length: int = 1000):
        """
        Initialize the ECG R-peak detector.
        
        Parameters:
        -----------
        model_path : str
            Path to the trained model file (.pt)
        device : str, optional
            Device to run inference on ('cuda', 'cpu', or None for auto-detection)
        fs : int
            Sampling frequency in Hz (default: 360)
        window_length : int
            Expected window length for the model (default: 1000)
        """
        self.fs = fs
        self.window_length = window_length
        self.tolerance_samples = int(0.01 * fs)  # 10ms tolerance
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Load model
        self.model = self._load_model(model_path)
        
    def _load_model(self, model_path: str) -> UNetPlusPlus1D:
        """Load the trained model from file."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        model = UNetPlusPlus1D(input_channels=1, base_filters=16).to(self.device)
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")
            
        model.eval()
        return model
    
    def preprocess_signal(self, signal_array: np.ndarray) -> np.ndarray:
        """
        Preprocess ECG signal by normalization.
        
        Parameters:
        -----------
        signal_array : np.ndarray
            Input ECG signal
            
        Returns:
        --------
        np.ndarray
            Preprocessed signal
        """
        signal_array = np.ascontiguousarray(signal_array)
        return (signal_array - np.mean(signal_array)) / (np.std(signal_array) + 1e-8)
    
    def bandpass_filter(self, data: np.ndarray, lowcut: float = 0.5, highcut: float = 40.0, order: int = 4) -> np.ndarray:
        """
        Apply Butterworth bandpass filter to ECG signal.
        
        Parameters:
        -----------
        data : np.ndarray
            Input ECG signal
        lowcut : float
            Low cutoff frequency in Hz
        highcut : float
            High cutoff frequency in Hz
        order : int
            Filter order
            
        Returns:
        --------
        np.ndarray
            Filtered signal
        """
        nyquist = 0.5 * self.fs
        low = lowcut / nyquist
        high = highcut / nyquist
        
        b, a = signal.butter(order, [low, high], btype='band')
        return signal.filtfilt(b, a, data)
    
    def localize_peaks(self, pred: np.ndarray) -> np.ndarray:
        """
        Localize R-peaks from probability sequence using adaptive thresholding.
        
        Parameters:
        -----------
        pred : np.ndarray
            Model output probability sequence
            
        Returns:
        --------
        np.ndarray
            Array of peak indices
        """
        conv_window = int(self.fs * 0.075)
        refractory = int(self.fs * 0.2)
        
        if conv_window < 1:
            conv_window = 1
            
        # Smooth the prediction
        c_pred = np.convolve(pred, np.ones(conv_window) / conv_window, mode='same')
        
        # Adaptive thresholding
        pred_mean = np.mean(c_pred)
        pred_std = np.std(c_pred)
        threshold = max(pred_mean + 2 * pred_std, 0.1)
        threshold = min(threshold, 0.8)
        
        # Find peaks
        peaks, _ = signal.find_peaks(c_pred, height=threshold, distance=refractory)
        
        # Fallback with prominence-based detection
        if len(peaks) == 0:
            alt_prom = pred_std * 0.5
            peaks, _ = signal.find_peaks(c_pred, prominence=alt_prom, distance=refractory)
            
        return peaks.astype(int)
    
    def calculate_heart_rate(self, peaks: np.ndarray) -> float:
        """
        Calculate heart rate from R-peak locations.
        
        Parameters:
        -----------
        peaks : np.ndarray
            Array of peak indices
            
        Returns:
        --------
        float
            Heart rate in BPM
        """
        if peaks is None or len(peaks) < 2:
            return np.nan
            
        rr_intervals = np.diff(peaks) / self.fs
        
        # Filter physiologically reasonable RR intervals
        rr_intervals = rr_intervals[(rr_intervals >= 0.3) & (rr_intervals <= 3.0)]
        
        if len(rr_intervals) == 0:
            return np.nan
            
        avg_rr = np.mean(rr_intervals)
        if avg_rr <= 0:
            return np.nan
            
        return 60.0 / avg_rr
    
    def detect_peaks(self, 
                    ecg_signal: np.ndarray, 
                    apply_filter: bool = True,
                    return_probabilities: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Detect R-peaks in ECG signal.
        
        Parameters:
        -----------
        ecg_signal : np.ndarray
            Input ECG signal
        apply_filter : bool
            Whether to apply bandpass filter
        return_probabilities : bool
            Whether to return probability sequence
            
        Returns:
        --------
        np.ndarray or tuple
            Peak indices, optionally with probability sequence
        """
        # Validate input
        if len(ecg_signal) != self.window_length:
            raise ValueError(f"Signal length {len(ecg_signal)} doesn't match expected window length {self.window_length}")
        
        # Preprocess signal
        processed_signal = self.preprocess_signal(ecg_signal)
        
        # Apply filter if requested
        if apply_filter:
            processed_signal = self.bandpass_filter(processed_signal)
        
        # Model inference
        with torch.no_grad():
            inp = torch.from_numpy(processed_signal).float().unsqueeze(0).unsqueeze(-1).to(self.device)
            pred = self.model(inp).squeeze().detach().cpu().numpy()
        
        # Localize peaks
        peaks = self.localize_peaks(pred)
        
        if return_probabilities:
            return peaks, pred
        else:
            return peaks
    
    def detect_peaks_batch(self, 
                          ecg_signals: np.ndarray, 
                          apply_filter: bool = True,
                          return_probabilities: bool = False) -> Union[List[np.ndarray], Tuple[List[np.ndarray], List[np.ndarray]]]:
        """
        Detect R-peaks in batch of ECG signals.
        
        Parameters:
        -----------
        ecg_signals : np.ndarray
            Batch of ECG signals (N, window_length)
        apply_filter : bool
            Whether to apply bandpass filter
        return_probabilities : bool
            Whether to return probability sequences
            
        Returns:
        --------
        List[np.ndarray] or tuple
            List of peak arrays, optionally with probability sequences
        """
        all_peaks = []
        all_probs = [] if return_probabilities else None
        
        for i in range(ecg_signals.shape[0]):
            peaks, probs = self.detect_peaks(
                ecg_signals[i], 
                apply_filter=apply_filter, 
                return_probabilities=True
            )
            all_peaks.append(peaks)
            if return_probabilities:
                all_probs.append(probs)
        
        if return_probabilities:
            return all_peaks, all_probs
        else:
            return all_peaks
    
    def calculate_metrics(self, 
                         pred_peaks: np.ndarray, 
                         gt_peaks: np.ndarray) -> Dict[str, float]:
        """
        Calculate detection metrics (precision, recall, F1).
        
        Parameters:
        -----------
        pred_peaks : np.ndarray
            Predicted peak indices
        gt_peaks : np.ndarray
            Ground truth peak indices
            
        Returns:
        --------
        Dict[str, float]
            Dictionary containing precision, recall, F1, TP, FP, FN
        """
        if len(pred_peaks) == 0 and len(gt_peaks) == 0:
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'tp': 0, 'fp': 0, 'fn': 0}
        if len(pred_peaks) == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': 0, 'fn': len(gt_peaks)}
        if len(gt_peaks) == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': len(pred_peaks), 'fn': 0}
        
        tp = 0
        matched_gt = set()
        
        for pred_idx in pred_peaks:
            candidates = np.where(np.abs(gt_peaks - pred_idx) <= self.tolerance_samples)[0]
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
        
        return {
            'precision': precision, 
            'recall': recall, 
            'f1': f1, 
            'tp': tp, 
            'fp': fp, 
            'fn': fn
        }
    
    def extract_gt_peaks_from_label(self, gt_label: np.ndarray) -> np.ndarray:
        """
        Extract ground truth peaks from label sequence.
        
        Parameters:
        -----------
        gt_label : np.ndarray
            Ground truth label sequence
            
        Returns:
        --------
        np.ndarray
            Ground truth peak indices
        """
        refractory = int(self.fs * 0.2)
        peaks, _ = signal.find_peaks(gt_label, height=0.5, distance=refractory)
        return peaks.astype(int)

# -----------------------------
# Utility Functions
# -----------------------------

def load_test_data(x_path: str, y_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load test data from numpy files.
    
    Parameters:
    -----------
    x_path : str
        Path to X_test.npy
    y_path : str
        Path to y_test.npy
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        X_test and y_test arrays
    """
    X_test = np.load(x_path)
    y_test = np.load(y_path)
    
    if X_test.shape[0] != y_test.shape[0]:
        raise ValueError(f"Mismatched number of samples: X={X_test.shape[0]}, y={y_test.shape[0]}")
    
    return X_test, y_test

def evaluate_model(detector: ECGRPeakDetector, 
                  X_test: np.ndarray, 
                  y_test: np.ndarray) -> Dict[str, float]:
    """
    Evaluate model performance on test data.
    
    Parameters:
    -----------
    detector : ECGRPeakDetector
        Initialized detector
    X_test : np.ndarray
        Test ECG signals
    y_test : np.ndarray
        Test labels
        
    Returns:
    --------
    Dict[str, float]
        Evaluation metrics
    """
    all_tp = 0
    all_fp = 0
    all_fn = 0
    hr_errors = []
    
    for i in range(X_test.shape[0]):
        # Get ground truth peaks
        gt_peaks = detector.extract_gt_peaks_from_label(y_test[i])
        
        # Detect peaks
        pred_peaks = detector.detect_peaks(X_test[i])
        
        # Calculate metrics
        metrics = detector.calculate_metrics(pred_peaks, gt_peaks)
        all_tp += metrics['tp']
        all_fp += metrics['fp']
        all_fn += metrics['fn']
        
        # Heart rate calculation
        pred_hr = detector.calculate_heart_rate(pred_peaks)
        gt_hr = detector.calculate_heart_rate(gt_peaks)
        
        if not (np.isnan(pred_hr) or np.isnan(gt_hr)):
            hr_errors.append(abs(pred_hr - gt_hr))
    
    # Aggregate metrics
    overall_precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    overall_recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
    hr_mae = float(np.mean(hr_errors)) if len(hr_errors) > 0 else float('nan')
    
    return {
        'precision': overall_precision,
        'recall': overall_recall,
        'f1': overall_f1,
        'hr_mae': hr_mae,
        'total_samples': X_test.shape[0]
    }
