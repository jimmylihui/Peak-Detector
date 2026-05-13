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
        self.fs = 360  # MIT-BIH sampling frequency
        
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
        
        X_train_PPG = np.load('/path/to/workspace/project-BCG-LLM/combined_data/combined_splitted_data/PPG/X_train.npy')
        X_val_PPG   = np.load('/path/to/workspace/project-BCG-LLM/combined_data/combined_splitted_data/PPG/X_val.npy')
        y_train_PPG   = np.load('/path/to/workspace/project-BCG-LLM/combined_data/combined_splitted_data/PPG/y_train.npy')
        y_val_PPG     = np.load('/path/to/workspace/project-BCG-LLM/combined_data/combined_splitted_data/PPG/y_val.npy')
        X_train_ECG = np.load('/path/to/workspace/project-BCG-LLM/combined_data/combined_splitted_data/ECG/X_train.npy')

        y_train_ECG   = np.load('/path/to/workspace/project-BCG-LLM/combined_data/combined_splitted_data/ECG/y_train.npy')
       
        X_train_BCG = np.load('/path/to/workspace/project-BCG-LLM/combined_data/combined_splitted_data/BCG/X_train.npy')
        y_train_BCG   = np.load('/path/to/workspace/project-BCG-LLM/combined_data/combined_splitted_data/BCG/y_train.npy')
        
        from sklearn.model_selection import train_test_split
        X_train_BCG, X_val_BCG, y_train_BCG, y_val_BCG = train_test_split(
            X_train_BCG, y_train_BCG, test_size=0.2, random_state=42
        )
        X_train_ECG, X_val_ECG, y_train_ECG, y_val_ECG = train_test_split(
            X_train_ECG, y_train_ECG, test_size=0.2, random_state=42
        )
       

        # Upsample each modality to match the largest split size
        target_train = max(X_train_ECG.shape[0], X_train_BCG.shape[0], X_train_PPG.shape[0])
        target_val = max(X_val_ECG.shape[0], X_val_BCG.shape[0], X_val_PPG.shape[0])

        X_train_ECG, y_train_ECG = self._upsample_to_size(X_train_ECG, y_train_ECG, target_train)
        X_train_BCG, y_train_BCG = self._upsample_to_size(X_train_BCG, y_train_BCG, target_train)
        X_train_PPG, y_train_PPG = self._upsample_to_size(X_train_PPG, y_train_PPG, target_train)

        X_val_ECG, y_val_ECG = self._upsample_to_size(X_val_ECG, y_val_ECG, target_val)
        X_val_BCG, y_val_BCG = self._upsample_to_size(X_val_BCG, y_val_BCG, target_val)
        X_val_PPG, y_val_PPG = self._upsample_to_size(X_val_PPG, y_val_PPG, target_val)

        X_train=np.concatenate((X_train_ECG, X_train_BCG, X_train_PPG), axis=0)
        X_val=np.concatenate((X_val_ECG, X_val_BCG, X_val_PPG), axis=0)
        y_train=np.concatenate((y_train_ECG, y_train_BCG, y_train_PPG), axis=0)
        y_val=np.concatenate((y_val_ECG, y_val_BCG, y_val_PPG), axis=0)

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
            model_save_path = f'sepconv_combined_model_{timestamp}.pt'
        
        self.save_model(model_save_path)
        
        return history
    

def main():
    """Main function to run the training pipeline"""
    
    # Configuration
    window_length = 2048  # SepConv uses 2048-sample windows by default
    tolerance = int(0.03*360)  # 3 positions tolerance in samples
    
    # Training configuration
    epochs = 5  # Reduced for faster training
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
        model_save_path='sepconv_combined_trained_model.pt',
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

if __name__ == "__main__":
    results_df = main()
