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
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        self.model = FRNet(in_channels=1, base_channels=64).to(self.device)
        params = self.count_parameters()
        print(f"Model parameters: {params['total']:,}")
    
    def count_parameters(self):
        """Count total and trainable parameters"""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}
    
    def load_dataset(self):
        """Load and split dataset"""
        X_train = np.load('/path/to/workspace/project-BCG-LLM/combined_data/combined_splitted_data/ECG/total_X_train.npy')
        y_train = np.load('/path/to/workspace/project-BCG-LLM/combined_data/combined_splitted_data/ECG/total_y_train.npy')
        X_train_2=np.load('/path/to/workspace/project-BCG-LLM/combined_data/combined_splitted_data/PPG/processed/X_train.npy')
        y_train_2=np.load('/path/to/workspace/project-BCG-LLM/combined_data/combined_splitted_data/PPG/processed/Y_train.npy')
        X_train_3=np.load('/path/to/workspace/project-BCG-LLM/combined_data/combined_splitted_data/BCG/X_train.npy')
        y_train_3=np.load('/path/to/workspace/project-BCG-LLM/combined_data/combined_splitted_data/BCG/y_train.npy')
        
        len_train=len(X_train)
        train_indices=np.random.choice(len_train, int(len_train*0.1), replace=False)
        X_train=X_train[train_indices]
        y_train=y_train[train_indices]
        
        X_train=np.concatenate([X_train, X_train_2, X_train_3], axis=0)
        y_train=np.concatenate([y_train, y_train_2, y_train_3], axis=0)
        
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
        train_start_time = time.time()
        
        train_loader = DataLoader(ECGDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(ECGDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
        
        total_train_samples = len(train_loader.dataset)
        
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
                torch.save(self.model.state_dict(), 'best_frnet_combined_model.pt')
                patience = 0
            else:
                patience += 1
                if patience >= 15:
                    print("Early stopping!")
                    break
        
        self.model.load_state_dict(torch.load('best_frnet_combined_model.pt'))
        
        # Calculate training throughput
        train_time = time.time() - train_start_time
        epochs_completed = len(history['train_loss'])
        train_throughput = (total_train_samples * epochs_completed) / train_time
        history['train_time'] = train_time
        history['train_throughput'] = train_throughput
        print(f"\nTraining throughput: {train_throughput:.2f} samples/sec")
        
        return history
    
    def test_throughput(self, X_test, y_test, batch_size=32):
        """Measure testing throughput"""
        test_dataset = ECGDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        self.model.eval()
        test_start = time.time()
        with torch.no_grad():
            for batch_signals, _ in test_loader:
                _ = self.model(batch_signals.to(self.device))
        test_time = time.time() - test_start
        test_throughput = len(test_dataset) / test_time
        return {'test_time': test_time, 'test_throughput': test_throughput}
    
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
        plt.savefig('frnet_combined_training_history.png', dpi=300)
        plt.show()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("FR-NET ECG R-PEAK DETECTION TRAINING")
    print("="*80)
    
    trainer = FRNetTrainer(window_length=1000)
    
    # Print model parameters
    params = trainer.count_parameters()
    print(f"\nModel Parameters:")
    print(f"  Total: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")
    
    X_train, X_val, y_train, y_val = trainer.load_dataset()
    
    history = trainer.train(X_train, y_train, X_val, y_val, 
                           epochs=20, batch_size=32, lr=0.001)
    
    # Measure testing throughput
    print("\nMeasuring testing throughput...")
    test_metrics = trainer.test_throughput(X_val, y_val, batch_size=32)
    print(f"Testing throughput: {test_metrics['test_throughput']:.2f} samples/sec")
    
    # Performance summary
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    print(f"Model Parameters: {params['total']:,}")
    print(f"Training Throughput: {history['train_throughput']:.2f} samples/sec")
    print(f"Testing Throughput: {test_metrics['test_throughput']:.2f} samples/sec")
    print("="*80)
    
    trainer.plot_history(history)
    
    print("\nTraining Complete!")
    print(f"Best model saved to: best_frnet_combined_model.pt")
    return history


if __name__ == "__main__":
    results = main()