#!/usr/bin/env python3
"""
FR-Net ECG R-peak Detection Training for MIT-BIH Database
CNN-Transformer hybrid model for sample point classification
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
from datetime import datetime
import matplotlib.pyplot as plt
import math
import time
from scipy import signal

# ============================================================================
# MODEL ARCHITECTURE (FR-Net from Paper)
# ============================================================================

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
        
        # Decoder with upsampling - FIXED CHANNEL DIMENSIONS
        # After concat with skip3 (256 channels): 256 + 256 = 512 channels
        self.up1 = nn.Sequential(
            nn.Conv1d(base_channels * 4, base_channels * 2, 9, padding=4),
            nn.ReLU(),
            nn.Conv1d(base_channels * 2, base_channels * 2, 9, padding=4),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        )
        
        # After concat with skip2 (128 channels): 128 + 256 = 384 channels
        self.up2 = nn.Sequential(
            nn.Conv1d(base_channels * 6, base_channels, 9, padding=4),  # 384 -> 64
            nn.ReLU(),
            nn.Conv1d(base_channels, base_channels, 9, padding=4),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        )
        
        # After concat with skip1 (64 channels): 64 + 64 = 128 channels
        self.up3 = nn.Sequential(
            nn.Conv1d(base_channels * 2, base_channels // 2, 9, padding=4),  # 128 -> 32
            nn.ReLU(),
            nn.Conv1d(base_channels // 2, base_channels // 2, 9, padding=4),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        )
        
        # Final convolution
        self.final = nn.Sequential(
            nn.Conv1d(base_channels // 2, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: (batch, seq_len, channels) -> (batch, channels, seq_len)
        if x.dim() == 3:
            x = x.transpose(1, 2)
        
        # CNN encoder with skip connections
        x1 = self.res_block1(x)  # 64 channels, 1000 length
        x = self.pool1(x1)       # 64 channels, 500 length
        
        x2 = self.res_block2(x)  # 128 channels, 500 length
        x = self.pool2(x2)       # 128 channels, 250 length
        
        x3 = self.res_block3(x)  # 256 channels, 250 length
        x = self.pool3(x3)       # 256 channels, 125 length
        
        # CNN output
        f1 = x
        
        # Transformer: (batch, channels, seq_len) -> (batch, seq_len, channels)
        x = x.transpose(1, 2)
        x = self.pos_encoding(x)
        f2 = self.transformer(x)
        f2 = f2.transpose(1, 2)  # Back to (batch, channels, seq_len)
        
        # Combine CNN and Transformer features
        fe = f1 + f2  # 256 channels, 125 length
        
        # Decoder with skip connections
        x = self.up1(fe)  # 128 channels, 250 length
        x = torch.cat([x, x3], dim=1)  # 128 + 256 = 384 channels
        
        x = self.up2(x)  # 64 channels, 500 length
        x = torch.cat([x, x2], dim=1)  # 64 + 128 = 192 channels (FIXED)
        
        # Update up3 for correct input channels
        x = self.up3_fixed(x)  # Handle 192 channels properly
        x = torch.cat([x, x1], dim=1)  # 32 + 64 = 96 channels (FIXED)
        
        x = self.final_fixed(x)  # Handle 96 channels
        
        return x.transpose(1, 2)  # (batch, seq_len, 1)
    
    def up3_fixed(self, x):
        """Fixed up3 to handle 192 input channels"""
        # 192 -> 32
        x = nn.functional.conv1d(x, 
                                 weight=torch.randn(32, 192, 9, device=x.device) if not hasattr(self, 'up3_conv1_weight') else self.up3_conv1_weight,
                                 padding=4)
        return x
    
    def final_fixed(self, x):
        """Fixed final to handle 96 input channels"""
        # 96 -> 1
        x = nn.functional.conv1d(x,
                                 weight=torch.randn(1, 96, 1, device=x.device) if not hasattr(self, 'final_conv_weight') else self.final_conv_weight)
        return torch.sigmoid(x)


# Let me provide a cleaner fix:

class FRNet(nn.Module):
    """FR-Net: CNN-Transformer hybrid for ECG R-peak detection - FIXED VERSION"""
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
        
        # Decoder with upsampling - CORRECTED CHANNEL CALCULATIONS
        # Stage 1: 256 channels input
        self.up1 = nn.Sequential(
            nn.Conv1d(base_channels * 4, base_channels * 2, 9, padding=4),  # 256 -> 128
            nn.ReLU(),
            nn.Conv1d(base_channels * 2, base_channels * 2, 9, padding=4),  # 128 -> 128
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        )
        
        # Stage 2: 128 (from up1) + 256 (skip3) = 384 channels input
        self.up2 = nn.Sequential(
            nn.Conv1d(base_channels * 6, base_channels, 9, padding=4),  # 384 -> 64
            nn.ReLU(),
            nn.Conv1d(base_channels, base_channels, 9, padding=4),  # 64 -> 64
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        )
        
        # Stage 3: 64 (from up2) + 128 (skip2) = 192 channels input
        self.up3 = nn.Sequential(
            nn.Conv1d(base_channels * 3, base_channels // 2, 9, padding=4),  # 192 -> 32
            nn.ReLU(),
            nn.Conv1d(base_channels // 2, base_channels // 2, 9, padding=4),  # 32 -> 32
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        )
        
        # Final: 32 (from up3) + 64 (skip1) = 96 channels input
        self.final = nn.Sequential(
            nn.Conv1d(base_channels + base_channels // 2, 1, 1),  # 96 -> 1
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: (batch, seq_len, channels) -> (batch, channels, seq_len)
        if x.dim() == 3:
            x = x.transpose(1, 2)
        
        # CNN encoder with skip connections
        x1 = self.res_block1(x)  # 64 channels, 1000 length
        x = self.pool1(x1)       # 64 channels, 500 length
        
        x2 = self.res_block2(x)  # 128 channels, 500 length
        x = self.pool2(x2)       # 128 channels, 250 length
        
        x3 = self.res_block3(x)  # 256 channels, 250 length
        x = self.pool3(x3)       # 256 channels, 125 length
        
        # CNN output
        f1 = x
        
        # Transformer: (batch, channels, seq_len) -> (batch, seq_len, channels)
        x = x.transpose(1, 2)
        x = self.pos_encoding(x)
        f2 = self.transformer(x)
        f2 = f2.transpose(1, 2)  # Back to (batch, channels, seq_len)
        
        # Combine CNN and Transformer features
        fe = f1 + f2  # 256 channels, 125 length
        
        # Decoder with skip connections
        x = self.up1(fe)  # 128 channels, 250 length
        x = torch.cat([x, x3], dim=1)  # 128 + 256 = 384 channels
        
        x = self.up2(x)  # 64 channels, 500 length
        x = torch.cat([x, x2], dim=1)  # 64 + 128 = 192 channels
        
        x = self.up3(x)  # 32 channels, 1000 length
        x = torch.cat([x, x1], dim=1)  # 32 + 64 = 96 channels
        
        x = self.final(x)  # 1 channel, 1000 length
        
        return x.transpose(1, 2)  # (batch, seq_len, 1)


# ============================================================================
# METRICS AND UTILITIES
# ============================================================================

def calculate_metrics(pred_peaks, gt_peaks, tolerance=0.03, fs=100):
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


# ============================================================================
# FOCAL LOSS
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


# ============================================================================
# DATASET
# ============================================================================

class ECGDataset(Dataset):
    """ECG Dataset for R-peak detection"""
    def __init__(self, signals, labels):
        self.signals = torch.FloatTensor(np.ascontiguousarray(signals)).unsqueeze(-1)
        self.labels = torch.FloatTensor(np.ascontiguousarray(labels)).unsqueeze(-1)
    
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]


# ============================================================================
# TRAINER
# ============================================================================

class FRNetTrainer:
    """FR-Net training pipeline"""
    def __init__(self, window_length=1000):
        self.window_length = window_length
        self.fs = 100  # BCG sampling frequency
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        self.model = FRNet(in_channels=1, base_channels=64).to(self.device)
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def load_dataset(self):
        """Load and split dataset"""
        X_train=np.load('/path/to/workspace/project-BCG-LLM/combined_data/combined_splitted_data/BCG/X_train.npy')
         
        y_train=np.load('/path/to/workspace/project-BCG-LLM/combined_data/combined_splitted_data/BCG/y_train.npy')
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        print(f"Training: X={X_train.shape}, y={y_train.shape}")
        print(f"Validation: X={X_val.shape}, y={y_val.shape}")
        
        return X_train, X_val, y_train, y_val
    
    def calculate_f1(self, pred, target):
        """Calculate F1 score"""
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        tp = torch.sum(pred_flat * target_flat).item()
        fp = torch.sum(pred_flat * (1 - target_flat)).item()
        fn = torch.sum((1 - pred_flat) * target_flat).item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return f1
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, lr=0.001):
        """Training loop"""
        train_loader = DataLoader(ECGDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(ECGDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = FocalLoss(alpha=0.25, gamma=2)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        history = {'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': []}
        best_val_loss = float('inf')
        patience = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss, train_f1 = 0, []
            for signals, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                signals, labels = signals.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(signals)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
                train_f1.append(self.calculate_f1((outputs > 0.5).float(), labels))
            
            # Validation
            self.model.eval()
            val_loss, val_f1 = 0, []
            with torch.no_grad():
                for signals, labels in val_loader:
                    signals, labels = signals.to(self.device), labels.to(self.device)
                    outputs = self.model(signals)
                    val_loss += criterion(outputs, labels).item()
                    val_f1.append(self.calculate_f1((outputs > 0.5).float(), labels))
            
            # Metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            avg_train_f1 = np.mean(train_f1)
            avg_val_f1 = np.mean(val_f1)
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['train_f1'].append(avg_train_f1)
            history['val_f1'].append(avg_val_f1)
            
            scheduler.step(avg_val_loss)
            
            print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, "
                  f"Train F1={avg_train_f1:.4f}, Val F1={avg_val_f1:.4f}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.model.state_dict(), 'best_frnet_kansas_model.pt')
                patience = 0
            else:
                patience += 1
                if patience >= 15:
                    print("Early stopping!")
                    break
        
        self.model.load_state_dict(torch.load('best_frnet_kansas_model.pt'))
        return history
    
    def plot_history(self, history):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes[0].plot(history['train_loss'], label='Train')
        axes[0].plot(history['val_loss'], label='Val')
        axes[0].set_title('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(history['train_f1'], label='Train')
        axes[1].plot(history['val_f1'], label='Val')
        axes[1].set_title('F1 Score')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('frnet_training_history.png', dpi=300)
        plt.show()
    
    def calculate_ibi_from_peaks(self, peaks, fs=100):
        """Calculate Inter-Beat Intervals (IBI) in seconds from peak indices"""
        if peaks is None or len(peaks) < 2:
            return np.array([])
        ibi = np.diff(peaks) / fs
        return ibi
    
    def calculate_hr_from_peaks(self, peaks, fs=100):
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
    
    def calculate_hrv_from_peaks(self, peaks, fs=100, metric='sdnn'):
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
    
    def localize_predicted_peaks(self, pred_probs, fs=100):
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
                
                outputs_np = outputs.cpu().numpy()
                
                batch_start_idx = batch_idx * batch_size
                for i in range(outputs_np.shape[0]):
                    sample_idx = batch_start_idx + i
                    if sample_idx >= len(X_test):
                        break
                    
                    if outputs_np.ndim == 3:
                        pred_probs = outputs_np[i, :, 0].flatten()
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
        print("FR-NET ECG R-PEAK DETECTION - CROSS-VALIDATION")
        print("="*80)
        
        print("Loading train and test data...")
        X_train, X_val, y_train, y_val = self.load_dataset()
        X_test = np.load('/path/to/workspace/project-BCG-LLM/combined_data/combined_splitted_data/BCG/X_test.npy')
        y_test = np.load('/path/to/workspace/project-BCG-LLM/combined_data/combined_splitted_data/BCG/y_test.npy')
        
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
            
            self.model = FRNet(in_channels=1, base_channels=64).to(self.device)
            
            print(f"\nTraining model for fold {fold_idx + 1}...")
            self.train(
                X_fold_train_split, y_fold_train_split,
                X_fold_val_split, y_fold_val_split,
                epochs=epochs, batch_size=batch_size, lr=learning_rate
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
        
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'FRNet_kansas_predicted_peaks.npy')
        predicted_peaks_array = np.array(all_predicted_peaks_binary, dtype=np.float32)
        np.save(output_path, predicted_peaks_array)
        print(f'\nPredicted peaks saved to: {output_path}')
        print(f'Shape: {predicted_peaks_array.shape}')
        
        return {
            'fold_metrics': fold_metrics,
            'aggregated': aggregated,
            'n_folds': n_folds
        }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("FR-NET ECG R-PEAK DETECTION TRAINING")
    print("="*80)
    
    trainer = FRNetTrainer(window_length=1000)
    X_train, X_val, y_train, y_val = trainer.load_dataset()
    
    history = trainer.train(X_train, y_train, X_val, y_val, 
                           epochs=10, batch_size=32, lr=0.001)
    
    trainer.plot_history(history)
    
    print("\nTraining Complete!")
    print(f"Best model saved to: best_frnet_model.pt")
    return history


def main_cross_validation():
    """Main function to run cross-validation"""
    
    window_length = 1000
    epochs = 10
    batch_size = 32
    learning_rate = 0.001
    
    print("FR-Net ECG R-peak Detection - Cross-Validation Pipeline (Kansas/BCG)")
    print("="*70)
    print(f"Window length: {window_length} samples")
    print(f"Epochs per fold: {epochs}, Batch size: {batch_size}, Learning rate: {learning_rate}")
    print("="*70)
    
    trainer = FRNetTrainer(window_length=window_length)
    
    cv_results = trainer.cross_validate(
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
    # results = main()