#!/usr/bin/env python3
"""
Deep Learning ECG R-peak Detection Benchmark for MIT-BIH Database
Applies the trained separable convolution model to MIT-BIH data split into 1000-sample windows
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import json
from datetime import datetime
import warnings
from scipy import signal
import time
warnings.filterwarnings('ignore')

# Attempt to import shared ECG dataloader
import sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))
try:
    from combined_data.benchmark.dataloader import load_mitbih_dataset as shared_load_mitbih_dataset
except Exception:
    shared_load_mitbih_dataset = None

sys.path.append('/path/to/workspace/project-BCG-LLM/PPG_peaks/Benchmark/yun/utils')
from sep_conv import *
# Using SepConv-based detector from utils

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        bce_loss = nn.BCELoss(reduction='none')(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
            
def calculate_metrics(pred_peaks, gt_peaks, tolerance=0.03, fs=360):
    """
    Calculate detection metrics
    
    Args:
        pred_peaks (array): Predicted peak indices
        gt_peaks (array): Ground truth peak indices
        tolerance (float): Tolerance in seconds
        fs (int): Sampling frequency
        
    Returns:
        dict: Metrics dictionary
    """
    # Consider only the first 1000 samples
    try:
        pred_peaks = np.asarray(pred_peaks)
        gt_peaks = np.asarray(gt_peaks)
        pred_peaks = pred_peaks[pred_peaks < 1000]
        gt_peaks = gt_peaks[gt_peaks < 1000]
    except Exception:
        pass
    if len(pred_peaks) == 0 and len(gt_peaks) == 0:
        return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'tp': 0, 'fp': 0, 'fn': 0}
    
    if len(pred_peaks) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': 0, 'fn': len(gt_peaks)}
    
    if len(gt_peaks) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': len(pred_peaks), 'fn': 0}
    
    # Convert tolerance to samples
    tolerance_samples = int(tolerance * fs)
    
    # Calculate True Positives (TP)
    tp = 0
    matched_gt = set()
    matched_pred = set()
    
    for i, pred_peak in enumerate(pred_peaks):
        for j, gt_peak in enumerate(gt_peaks):
            if j not in matched_gt and abs(pred_peak - gt_peak) <= tolerance_samples:
                tp += 1
                matched_gt.add(j)
                matched_pred.add(i)
                break
    
    # Calculate False Positives (FP) and False Negatives (FN)
    fp = len(pred_peaks) - tp
    fn = len(gt_peaks) - tp
    
    # Calculate metrics
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

def calculate_mape(predicted: float, actual: float, min_threshold: float = 1.0) -> float:
    """Calculate Mean Absolute Percentage Error (MAPE) for a single value."""
    if np.isnan(predicted) or np.isnan(actual) or actual == 0:
        return np.nan
    if abs(actual) < min_threshold:
        return np.nan
    return abs((predicted - actual) / actual) * 100.0





class ECGDataset(Dataset):
    """Dataset for ECG signals"""

    def __init__(self, signals, labels):
        # Ensure arrays are contiguous to avoid negative stride issues
        signals = np.ascontiguousarray(signals)
        labels = np.ascontiguousarray(labels)
        
        # Expect signals as (N, C, L). If (N, L), add channel dim.
        if signals.ndim == 2:
            signals = signals[:, None, :]
        self.signals = torch.FloatTensor(signals)

        # Expect labels as (N, 1, L). If (N, L), add channel dim. If (L,), make (1,1,L)
        if labels.ndim == 1:
            labels = labels[None, None, :]
        elif labels.ndim == 2:
            labels = labels[:, None, :]
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]





class DeepLearningMitbihBenchmark:
    """
    Deep Learning ECG R-peak detection benchmark for MIT-BIH database
    """
    
    def __init__(self, model_path=None, window_length=2048, tolerance=3):
        """
        Initialize the benchmark
        
        Args:
            model_path (str): Path to trained model file
            window_length (int): Length of signal windows in samples (1000 for UNet++)
            tolerance (int): Tolerance for peak matching in samples
        """
        self.window_length = window_length
        self.tolerance = tolerance
        self.fs = 300  # CapnoBase PPG sampling frequency
        
        # Set up device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        

        self.model = Sep_conv_detector(n_channel=2).to(self.device)
        
        # Load trained weights if available
        if model_path is None:
            # Use default model path
            path_base = os.path.dirname(os.path.abspath(__file__))
            self.model_path = os.path.join(path_base, 'model', 'sepconv_model.pt')
        else:
            self.model_path = model_path
            
        model_loaded = False
        if os.path.exists(self.model_path):
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                # Handle both full checkpoints and direct state_dict files
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint

                self.model.load_state_dict(state_dict)
                self.model.eval()
                print(f"SepConv model loaded from: {self.model_path}")
                model_loaded = True
            except Exception as e:
                print(f"Failed to load model from {self.model_path}: {e}")
                print("Initializing with random weights...")
                model_loaded = False
        else:
            print(f"Model file not found at {self.model_path}")
            print("Initializing with random weights...")
            model_loaded = False

        if not model_loaded:
            print("WARNING: Using randomly initialized SepConv model")
            print("This model will not perform well for peak detection!")
            print("Please train the model first using the training pipeline.")
            self.model.eval()
        
        # Initialize preprocessing parameters (kept minimal; no resampling/WT params needed)
    
    def preprocess_signal(self, signal_data):
        """
        Preprocess signal for SepConv model
        
        Args:
            signal_data (array): Raw ECG signal
            
        Returns:
            array: Preprocessed signal array
        """
        # Ensure input is contiguous
        signal_data = np.ascontiguousarray(signal_data)
        
        # Simple preprocessing for SepConv
        # 1. Normalize the signal
        signal_normalized = (signal_data - np.mean(signal_data)) / (np.std(signal_data) + 1e-8)
        
        # 2. Apply light filtering (disabled; use normalized signal directly)
        filtered = signal_normalized
        
        # Ensure output is contiguous
        return np.ascontiguousarray(filtered)
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, learning_rate=0.001):
        """
        Train the SepConv model
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            learning_rate (float): Learning rate for optimizer
            
        Returns:
            dict: Training history
        """
        print("Starting SepConv model training...")
        
        # Create datasets and dataloaders
        train_dataset = ECGDataset(X_train, y_train)
        val_dataset = ECGDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Calculate class weights for handling imbalance
        positive_samples = np.sum(y_train)
        negative_samples = y_train.size - positive_samples
        pos_weight = negative_samples / (positive_samples + 1e-8)
        
        print(f"Class balance: {positive_samples:.0f} positive, {negative_samples:.0f} negative")
        print(f"Positive weight: {pos_weight:.2f}")
        
        # Initialize optimizer and criterion
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        dice_criterion = DiceLoss()
        bce_logits_criterion = nn.BCEWithLogitsLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, min_lr=1e-6)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_f1': [],
            'val_f1': []
        }
        
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        early_stopping_patience = 15
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_f1_scores = []
            
            for batch_signals, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                batch_signals = batch_signals.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_signals)
                loss = 0.7 * dice_criterion(outputs, batch_labels) + 0.3 * bce_logits_criterion(outputs, batch_labels)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
                
                # Calculate F1 score for this batch
                with torch.no_grad():
                    pred_binary = (torch.sigmoid(outputs) > 0.5).float()
                    f1 = self.calculate_batch_f1(pred_binary, batch_labels)
                    train_f1_scores.append(f1)
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_f1_scores = []
            
            with torch.no_grad():
                for batch_signals, batch_labels in val_loader:
                    batch_signals = batch_signals.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    outputs = self.model(batch_signals)
                    loss = 0.7 * dice_criterion(outputs, batch_labels) + 0.3 * bce_logits_criterion(outputs, batch_labels)
                    val_loss += loss.item()
                    
                    # Calculate F1 score for this batch
                    pred_binary = (torch.sigmoid(outputs) > 0.5).float()
                    f1 = self.calculate_batch_f1(pred_binary, batch_labels)
                    val_f1_scores.append(f1)
            
            # Calculate average metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            avg_train_f1 = np.mean(train_f1_scores)
            avg_val_f1 = np.mean(val_f1_scores)
            
            # Update history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['train_f1'].append(avg_train_f1)
            history['val_f1'].append(avg_val_f1)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                  f"Train F1: {avg_train_f1:.4f}, Val F1: {avg_val_f1:.4f}")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print("Loaded best model weights")
        
        return history
    
    def calculate_batch_f1(self, pred_binary, targets):
        """Calculate F1 score for a batch"""
        pred_flat = pred_binary.flatten()
        target_flat = targets.flatten()
        
        tp = torch.sum(pred_flat * target_flat).item()
        fp = torch.sum(pred_flat * (1 - target_flat)).item()
        fn = torch.sum((1 - pred_flat) * target_flat).item()
        
        if tp + fp == 0:
            precision = 0
        else:
            precision = tp / (tp + fp)
        
        if tp + fn == 0:
            recall = 0
        else:
            recall = tp / (tp + fn)
        
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        return f1
    
    def save_model(self, filepath):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'window_length': self.window_length,
            'tolerance': self.tolerance,
            'fs': self.fs
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"Model loaded from {filepath}")
    
    def plot_training_history(self, history, save_path=None):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        axes[0].plot(history['train_loss'], label='Training Loss', color='blue')
        axes[0].plot(history['val_loss'], label='Validation Loss', color='red')
        axes[0].set_title('Training and Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot F1 score
        axes[1].plot(history['train_f1'], label='Training F1', color='blue')
        axes[1].plot(history['val_f1'], label='Validation F1', color='red')
        axes[1].set_title('Training and Validation F1 Score')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('F1 Score')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.show()
        
        
        
        
        
        
    def _upsample_to_size(self, X, y, target_size, seed=42):
        """
        Upsample (with replacement) arrays X and y to target_size along axis 0.
        If current size equals or exceeds target_size, returns inputs unchanged.
        """
        current_size = X.shape[0]
        if current_size >= target_size:
            return X, y
        rng = np.random.RandomState(seed)
        extra_indices = rng.choice(current_size, size=target_size - current_size, replace=True)
        X_extra = X[extra_indices]
        y_extra = y[extra_indices]
        X_balanced = np.concatenate([X, X_extra], axis=0)
        y_balanced = np.concatenate([y, y_extra], axis=0)
        return X_balanced, y_balanced

    def load_dataset(self):
        
        X_train = np.load('/path/to/workspace/project-BCG-LLM/combined_data/combined_splitted_data/PPG_capnobase/processed/X_train.npy')
        
        y_train   = np.load('/path/to/workspace/project-BCG-LLM/combined_data/combined_splitted_data/PPG_capnobase/processed/Y_train_gauss.npy')
        
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

        # Ensure 2-channel features: raw + first difference
        def make_two_channel(arr: np.ndarray) -> np.ndarray:
            arr = np.ascontiguousarray(arr)
            if arr.ndim == 1:
                arr = arr[None, :]
            # Ensure fixed length 2048 for SepConv
            target_len = 2048
            L = arr.shape[-1]
            if L < target_len:
                pad_width = target_len - L
                arr = np.pad(arr, ((0,0),(0,pad_width)), mode='edge')
            elif L > target_len:
                arr = arr[..., :target_len]
            # Compute diff along last axis; pad last value to keep length
            diff = np.diff(arr, axis=-1)
            last = arr[..., -1:]
            diff_padded = np.concatenate([diff, last], axis=-1)
            two_ch = np.stack([arr, diff_padded], axis=1)  # (N, 2, L)
            return two_ch

        X_train = make_two_channel(X_train)
        X_val   = make_two_channel(X_val)

        # Ensure labels length matches 2048 and are binary floats, return as (N, L)
        def make_labels(arr: np.ndarray) -> np.ndarray:
            arr = np.ascontiguousarray(arr)
            if arr.ndim == 1:
                arr = arr[None, :]
            target_len = 2048
            L = arr.shape[-1]
            if L < target_len:
                pad_width = target_len - L
                arr = np.pad(arr, ((0,0),(0,pad_width)), mode='constant')
            elif L > target_len:
                arr = arr[:, :target_len]
            # binarize if not already
            if arr.dtype != np.float32 and arr.dtype != np.float64:
                arr = arr.astype(np.float32)
            arr = (arr > 0.5).astype(np.float32)
            return arr

        y_train = make_labels(y_train)
        y_val   = make_labels(y_val)

        return X_train, X_val, y_train, y_val
    
    def calculate_ibi_from_peaks(self, peaks, fs=300):
        """Calculate Inter-Beat Intervals (IBI) in seconds from peak indices"""
        if peaks is None or len(peaks) < 2:
            return np.array([])
        ibi = np.diff(peaks) / fs
        return ibi
    
    def calculate_hr_from_peaks(self, peaks, fs=300):
        """Calculate Heart Rate in BPM from peak indices"""
        if peaks is None or len(peaks) < 2:
            return np.nan
        ibi = self.calculate_ibi_from_peaks(peaks, fs)
        if len(ibi) == 0:
            return np.nan
        avg_ibi = np.mean(ibi)
        if avg_ibi <= 0:
            return np.nan
        return 60.0 / avg_ibi
    
    def calculate_hrv_from_peaks(self, peaks, fs=300, metric='sdnn'):
        """Calculate Heart Rate Variability (SDNN or RMSSD) in milliseconds from peak indices"""
        if peaks is None or len(peaks) < 2:
            return np.nan
        ibi_ms = self.calculate_ibi_from_peaks(peaks, fs) * 1000.0  # Convert to milliseconds
        if len(ibi_ms) == 0:
            return np.nan
        if metric.lower() == 'sdnn':
            return float(np.std(ibi_ms, ddof=1))
        elif metric.lower() == 'rmssd':
            if len(ibi_ms) < 2:
                return np.nan
            return float(np.sqrt(np.mean(np.diff(ibi_ms) ** 2)))
        return np.nan
    
    def extract_peaks_from_labels(self, y_labels):
        """Extract peak indices from label arrays (handles both binary and Gaussian-smoothed labels)"""
        all_peaks = []
        for i in range(len(y_labels)):
            signal_flat = y_labels[i].flatten()
            refractory = int(self.fs * 0.2)
            peaks, _ = signal.find_peaks(signal_flat, height=0.5, distance=refractory)
            all_peaks.append(peaks)
        return all_peaks
    
    def localize_predicted_peaks(self, pred_probs, fs=300):
        """Localize peaks from probability sequence using adaptive thresholding and refractory."""
        conv_window = int(fs * 0.075)
        refractory = int(fs * 0.2)
        if conv_window < 1:
            conv_window = 1
        c_pred = np.convolve(pred_probs, np.ones(conv_window) / conv_window, mode='same')
        pred_mean = np.mean(c_pred)
        pred_std = np.std(c_pred)
        threshold = max(pred_mean + 2 * pred_std, 0.1)
        threshold = min(threshold, 0.8)
        peaks, _ = signal.find_peaks(c_pred, height=threshold, distance=refractory)
        if len(peaks) == 0:
            alt_prom = pred_std * 0.5
            peaks, _ = signal.find_peaks(c_pred, prominence=alt_prom, distance=refractory)
        return peaks.astype(int)
    
    def evaluate_fold(self, X_test, y_test, batch_size=32):
        """Evaluate model on a test fold and return metrics"""
        test_dataset = ECGDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        self.model.eval()
        
        all_tp = 0
        all_fp = 0
        all_fn = 0
        hr_errors = []
        hrv_sdnn_errors = []
        hrv_rmssd_errors = []
        hr_mapes = []
        hrv_sdnn_mapes = []
        hrv_rmssd_mapes = []
        inference_times = []
        all_predicted_peaks_binary = []
        
        gt_peaks_list = self.extract_peaks_from_labels(y_test)
        
        with torch.no_grad():
            inference_start = time.time()
            for batch_idx, (batch_signals, batch_labels) in enumerate(test_loader):
                batch_signals = batch_signals.to(self.device)
                
                batch_start = time.time()
                outputs = self.model(batch_signals)
                batch_inference_time = time.time() - batch_start
                
                outputs_np = torch.sigmoid(outputs).cpu().numpy()
                
                batch_start_idx = batch_idx * batch_size
                for i in range(outputs_np.shape[0]):
                    sample_idx = batch_start_idx + i
                    if sample_idx >= len(X_test):
                        break
                    
                    if outputs_np.ndim == 3:
                        pred_probs = outputs_np[i, 0, :].flatten()
                    else:
                        pred_probs = outputs_np[i].flatten()
                    
                    pred_peaks = self.localize_predicted_peaks(pred_probs, fs=self.fs)
                    gt_peaks = gt_peaks_list[sample_idx]
                    
                    if len(pred_peaks) > 0 or len(gt_peaks) > 0:
                        metrics = calculate_metrics(pred_peaks, gt_peaks, tolerance=0.03, fs=self.fs)
                        all_tp += metrics['tp']
                        all_fp += metrics['fp']
                        all_fn += metrics['fn']
                        
                        pred_hr = self.calculate_hr_from_peaks(pred_peaks, fs=self.fs)
                        gt_hr = self.calculate_hr_from_peaks(gt_peaks, fs=self.fs)
                        if not (np.isnan(pred_hr) or np.isnan(gt_hr)):
                            hr_errors.append(abs(pred_hr - gt_hr))
                            hr_mape = calculate_mape(pred_hr, gt_hr, min_threshold=30.0)
                            if not np.isnan(hr_mape):
                                hr_mapes.append(hr_mape)
                        
                        pred_hrv_sdnn = self.calculate_hrv_from_peaks(pred_peaks, fs=self.fs, metric='sdnn')
                        gt_hrv_sdnn = self.calculate_hrv_from_peaks(gt_peaks, fs=self.fs, metric='sdnn')
                        if not (np.isnan(pred_hrv_sdnn) or np.isnan(gt_hrv_sdnn)):
                            hrv_sdnn_errors.append(abs(pred_hrv_sdnn - gt_hrv_sdnn))
                            hrv_sdnn_mape = calculate_mape(pred_hrv_sdnn, gt_hrv_sdnn, min_threshold=5.0)
                            if not np.isnan(hrv_sdnn_mape):
                                hrv_sdnn_mapes.append(hrv_sdnn_mape)
                        
                        pred_hrv_rmssd = self.calculate_hrv_from_peaks(pred_peaks, fs=self.fs, metric='rmssd')
                        gt_hrv_rmssd = self.calculate_hrv_from_peaks(gt_peaks, fs=self.fs, metric='rmssd')
                        if not (np.isnan(pred_hrv_rmssd) or np.isnan(gt_hrv_rmssd)):
                            hrv_rmssd_errors.append(abs(pred_hrv_rmssd - gt_hrv_rmssd))
                            hrv_rmssd_mape = calculate_mape(pred_hrv_rmssd, gt_hrv_rmssd, min_threshold=5.0)
                            if not np.isnan(hrv_rmssd_mape):
                                hrv_rmssd_mapes.append(hrv_rmssd_mape)
                    
                    binary_peaks = np.zeros(self.window_length, dtype=np.float32)
                    valid_peaks = pred_peaks[pred_peaks < self.window_length]
                    if len(valid_peaks) > 0:
                        binary_peaks[valid_peaks] = 1.0
                    all_predicted_peaks_binary.append(binary_peaks)
                    
                    inference_times.append(batch_inference_time / batch_size)
        
        total_inference_time = time.time() - inference_start
        
        overall_precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
        overall_recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
        hr_mae = float(np.mean(hr_errors)) if len(hr_errors) > 0 else float('nan')
        hrv_sdnn_mae = float(np.mean(hrv_sdnn_errors)) if len(hrv_sdnn_errors) > 0 else float('nan')
        hrv_rmssd_mae = float(np.mean(hrv_rmssd_errors)) if len(hrv_rmssd_errors) > 0 else float('nan')
        hr_mape = float(np.mean(hr_mapes)) if len(hr_mapes) > 0 else float('nan')
        hrv_sdnn_mape = float(np.mean(hrv_sdnn_mapes)) if len(hrv_sdnn_mapes) > 0 else float('nan')
        hrv_rmssd_mape = float(np.mean(hrv_rmssd_mapes)) if len(hrv_rmssd_mapes) > 0 else float('nan')
        avg_inference_time_per_sample = total_inference_time / len(X_test) if len(X_test) > 0 else 0.0
        inference_throughput = 1 / avg_inference_time_per_sample if avg_inference_time_per_sample > 0 else 0.0
        
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
            'binary_peaks': all_predicted_peaks_binary,
            'total_inference_time': total_inference_time,
            'avg_inference_time_per_sample': avg_inference_time_per_sample,
            'inference_throughput': inference_throughput
        }
    
    def cross_validate(self, epochs=5, batch_size=32, learning_rate=0.001):
        """Perform k-fold cross-validation"""
        print("="*80)
        print("SepConv ECG R-PEAK DETECTION - CROSS-VALIDATION")
        print("="*80)
        
        print("Loading train and test data...")
        X_train, X_val, y_train, y_val = self.load_dataset()
        X_test = np.load('/path/to/workspace/project-BCG-LLM/combined_data/combined_splitted_data/PPG_capnobase/processed/X_test.npy')
        y_test = np.load('/path/to/workspace/project-BCG-LLM/combined_data/combined_splitted_data/PPG_capnobase/processed/Y_test_gauss.npy')
        
        def make_two_channel(arr: np.ndarray) -> np.ndarray:
            arr = np.ascontiguousarray(arr)
            if arr.ndim == 1:
                arr = arr[None, :]
            target_len = 2048
            L = arr.shape[-1]
            if L < target_len:
                pad_width = target_len - L
                arr = np.pad(arr, ((0,0),(0,pad_width)), mode='edge')
            elif L > target_len:
                arr = arr[..., :target_len]
            diff = np.diff(arr, axis=-1)
            last = arr[..., -1:]
            diff_padded = np.concatenate([diff, last], axis=-1)
            two_ch = np.stack([arr, diff_padded], axis=1)
            return two_ch
        
        def make_labels(arr: np.ndarray) -> np.ndarray:
            arr = np.ascontiguousarray(arr)
            if arr.ndim == 1:
                arr = arr[None, :]
            target_len = 2048
            L = arr.shape[-1]
            if L < target_len:
                pad_width = target_len - L
                arr = np.pad(arr, ((0,0),(0,pad_width)), mode='constant')
            elif L > target_len:
                arr = arr[:, :target_len]
            if arr.dtype != np.float32 and arr.dtype != np.float64:
                arr = arr.astype(np.float32)
            arr = (arr > 0.5).astype(np.float32)
            return arr
        
        X_test = make_two_channel(X_test)
        y_test = make_labels(y_test)
        
        original_test_size = X_test.shape[0]
        X_combined = np.concatenate([X_train, X_test], axis=0)
        y_combined = np.concatenate([y_train, y_test], axis=0)
        
        print(f"Combined dataset: {X_combined.shape[0]} samples")
        print(f"Original test size: {original_test_size} samples")
        
        n_folds = X_combined.shape[0] // original_test_size
        print(f"Number of folds: {n_folds}")
        
        fold_metrics = []
        all_predicted_peaks_binary = []
        
        for fold_idx in range(n_folds):
            print(f'\n{"="*80}')
            print(f'Fold {fold_idx + 1}/{n_folds}')
            print(f'{"="*80}')
            
            fold_start = fold_idx * original_test_size
            fold_end = (fold_idx + 1) * original_test_size
            
            X_fold_test = X_combined[fold_start:fold_end]
            y_fold_test = y_combined[fold_start:fold_end]
            
            X_fold_train = np.concatenate([X_combined[:fold_start], X_combined[fold_end:]], axis=0)
            y_fold_train = np.concatenate([y_combined[:fold_start], y_combined[fold_end:]], axis=0)
            
            X_fold_train_split, X_fold_val_split, y_fold_train_split, y_fold_val_split = train_test_split(
                X_fold_train, y_fold_train, test_size=0.2, random_state=42
            )
            
            print(f"Fold train size: {X_fold_train_split.shape[0]} samples")
            print(f"Fold val size: {X_fold_val_split.shape[0]} samples")
            print(f"Fold test size: {X_fold_test.shape[0]} samples")
            
            self.model = Sep_conv_detector(n_channel=2).to(self.device)
            
            print(f"\nTraining model for fold {fold_idx + 1}...")
            self.train_model(
                X_fold_train_split, y_fold_train_split,
                X_fold_val_split, y_fold_val_split,
                epochs=epochs, batch_size=batch_size, learning_rate=learning_rate
            )
            
            print(f"\nEvaluating fold {fold_idx + 1}...")
            fold_result = self.evaluate_fold(X_fold_test, y_fold_test, batch_size=batch_size)
            fold_metrics.append(fold_result)
            
            all_predicted_peaks_binary.extend(fold_result['binary_peaks'])
            
            print(f"\nFold {fold_idx + 1} Results:")
            print(f"  Precision: {fold_result['precision']:.4f}")
            print(f"  Recall: {fold_result['recall']:.4f}")
            print(f"  F1-Score: {fold_result['f1']:.4f}")
            print(f"  HR MAE: {fold_result['hr_mae']:.2f}" if not np.isnan(fold_result['hr_mae']) else f"  HR MAE: NaN")
            print(f"  HR MAPE: {fold_result['hr_mape']:.2f}" if not np.isnan(fold_result['hr_mape']) else f"  HR MAPE: NaN")
        
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
        
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SepConv_capnobase_predicted_peaks.npy')
        predicted_peaks_array = np.array(all_predicted_peaks_binary, dtype=np.float32)
        np.save(output_path, predicted_peaks_array)
        print(f'\nPredicted peaks saved to: {output_path}')
        print(f'Shape: {predicted_peaks_array.shape}')
        
        return {
            'fold_metrics': fold_metrics,
            'aggregated': aggregated,
            'n_folds': n_folds
        }
    
    def train_and_evaluate(self, model_save_path=None, epochs=100, batch_size=32, 
                          learning_rate=0.001):
        """
        Complete training and evaluation pipeline
        
        Args:
            model_save_path (str): Path to save trained model
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            learning_rate (float): Learning rate
            
        Returns:
            tuple: (training_history, evaluation_results)
        """
        print("="*80)
        print("SepConv ECG R-PEAK DETECTION - TRAINING AND EVALUATION")
        print("="*80)
        
        X_train, X_val, y_train, y_val = self.load_dataset()
        
       
        
        
        
        print(f"Training data shapes:")
        print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")
        
        history = self.train_model(X_train, y_train, X_val, y_val, 
                                 epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
        
        # Plot training history
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_plot_path = f'sepconv_training_history_{timestamp}.png'
        self.plot_training_history(history, history_plot_path)
        
        # Save model
        if model_save_path is None:
            model_save_path = f'sepconv_mitbih_model_{timestamp}.pt'
        
        self.save_model(model_save_path)
        
        return history
    

def main():
    """Main function to run the training pipeline"""
    
    # Configuration
    window_length = 2048  # SepConv uses 2048-sample windows by default
    tolerance = int(0.03*360)  # 3 positions tolerance in samples
    
    # Training configuration
    epochs = 50  # Reduced for faster training
    batch_size = 32
    learning_rate = 0.001
    
    print("SepConv ECG R-peak Detection - Training Pipeline")
    print("="*70)
    print(f"Window length: {window_length} samples")
    print(f"Tolerance: {tolerance} samples")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, Learning rate: {learning_rate}")
    print("="*70)
    
    # Initialize benchmark
    benchmark = DeepLearningMitbihBenchmark(
        window_length=window_length, 
        tolerance=tolerance
    )
    
    # Train model
    history = benchmark.train_and_evaluate(
        model_save_path='sepconv_capnobase_trained_model.pt',
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    if history is not None:
        print("\n" + "="*80)
        print("TRAINING COMPLETE!")
        print("="*80)
    else:
        print("Training failed!")
            
    return history

def main_cross_validation():
    """Main function to run cross-validation"""
    
    window_length = 2048
    tolerance = int(0.03*300)
    
    epochs = 50
    batch_size = 32
    learning_rate = 0.001
    
    print("SepConv ECG R-peak Detection - Cross-Validation Pipeline (CapnoBase)")
    print("="*70)
    print(f"Window length: {window_length} samples")
    print(f"Tolerance: {tolerance} samples")
    print(f"Epochs per fold: {epochs}, Batch size: {batch_size}, Learning rate: {learning_rate}")
    print("="*70)
    
    benchmark = DeepLearningMitbihBenchmark(
        window_length=window_length, 
        tolerance=tolerance
    )
    
    cv_results = benchmark.cross_validate(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    if cv_results is not None:
        print("\n" + "="*80)
        print("CROSS-VALIDATION COMPLETE!")
        print("="*80)
    else:
        print("Cross-validation failed!")
            
    return cv_results

if __name__ == "__main__":
    cv_results = main_cross_validation()
    # results_df = main()
