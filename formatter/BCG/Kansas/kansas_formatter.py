import numpy as np
import sys
import os
import json
import pandas as pd
from scipy import signal
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from typing import List, Dict, Any
from pathlib import Path
import random

# Note: We use processed fetal ECG data with gaussian peak annotations

class KansasBCGDatasetFormatter:
    """
    Formats Kansas BCG signal data for instruction finetuning.
    Converts processed BCG signals to J-peak detection tasks with length 10000.
    """
    
    def __init__(self, segment_length: int = 1000, sampling_rate: int = 100, use_special_tokens: bool = True):
        self.segment_length = segment_length  # Length of each segment (2000 samples)
        self.sampling_rate = sampling_rate    # Kansas sampling rate is 100 Hz
        self.use_special_tokens = use_special_tokens
        
        # Special tokens for time series formatting
        self.start_token = "<TS_START>"
        self.end_token = "<TS_END>"
        self.sep_token = "<TS_SEP>"
        
    



    def detect_input_peaks(self, signal: np.ndarray) -> np.ndarray:
        """Detect both positive and negative peaks for input formatting."""
        signal = np.mean(signal, axis=1) if len(signal.shape) > 1 else signal
        try:
            pos_peaks, _ = find_peaks(signal, distance=10)
            neg_peaks, _ = find_peaks(-signal, distance=10)
            return np.sort(np.concatenate([pos_peaks, neg_peaks])) if len(pos_peaks) + len(neg_peaks) > 0 else np.array([])
        except:
            return np.array([])

    def normalize_signal(self, signal: np.ndarray) -> np.ndarray:
        """Normalize signal to zero mean and unit variance."""
        signal_std = np.std(signal)
        return (signal - np.mean(signal)) / signal_std if signal_std > 0 else np.zeros_like(signal)

    def reconstruct_signal_from_peaks(self, peaks: np.ndarray, signal: np.ndarray) -> np.ndarray:
        """Reconstruct signal by interpolating from peak values."""
        peak_indices = peaks[peaks < len(signal)]
        if len(peak_indices) < 2:
            return np.zeros_like(signal)
        f = interp1d(peak_indices, signal[peak_indices], kind='linear', fill_value='extrapolate')
        return f(np.arange(len(signal)))
    
    def calculate_error_metrics(self, original: np.ndarray, reconstructed: np.ndarray, peaks: np.ndarray = None) -> Dict[str, float]:
        """Calculate MAE, RMSE, and MAPE between original and reconstructed signals."""
        if peaks is not None and len(peaks) >= 2:
            original = original[int(peaks[0]):int(peaks[-1])]
            reconstructed = reconstructed[int(peaks[0]):int(peaks[-1])]
        
        diff = original - reconstructed
        mae = np.mean(np.abs(diff))
        rmse = np.sqrt(np.mean(diff ** 2))
        
        non_zero_mask = np.abs(original) > 1e-10
        mape = np.mean(np.abs(diff[non_zero_mask] / original[non_zero_mask])) * 100 if np.sum(non_zero_mask) > 0 else 0.0
        
        corr_coef = np.corrcoef(original, reconstructed)[0, 1] if len(original) > 1 and np.std(original) > 0 and np.std(reconstructed) > 0 else 0.0
        
        return {'mae': mae, 'rmse': rmse, 'mape': mape, 'corr_coef': corr_coef}

    def calculate_r_peak_recall(self, true_r_peaks: np.ndarray, detected_peaks: np.ndarray, tolerance: int = 30) -> float:
        """Calculate R-peak recall with tolerance."""
        if len(true_r_peaks) == 0 or len(detected_peaks) == 0:
            return 0.0
        
        true_positives = 0
        matched_detected = set()
        
        for true_peak in true_r_peaks:
            distances = np.abs(detected_peaks - true_peak)
            min_distance_idx = np.argmin(distances)
            min_distance = distances[min_distance_idx]
            
            if min_distance <= tolerance and min_distance_idx not in matched_detected:
                true_positives += 1
                matched_detected.add(min_distance_idx)
        
        return true_positives / len(true_r_peaks)

    

    def format_peaks_for_input(self, peaks: np.ndarray, signal: np.ndarray) -> str:
        """Format peaks for input text."""
        if len(peaks) == 0:
            return "No significant peaks detected in the signal."
        
        from datetime import datetime, timedelta
        base_time = datetime(2020, 1, 1, 0, 0, 0)
        
        peak_lines = []
        for peak_idx in peaks:
            if 0 <= peak_idx < len(signal):
                # Use peak index directly as seconds (synthetic timestamp like BCG formatter)
                timestamp = base_time + timedelta(seconds=int(peak_idx))
                timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                peak_lines.append(f"Date: {timestamp_str}, Value: {signal[peak_idx]:.6f}")
        
        peaks_block = "\\n".join(peak_lines)
        return f"<TS_START>\\n{peaks_block}\\n<TS_END>"

    def format_j_peaks_output(self, j_peaks: np.ndarray) -> str:
        """Format J peaks for structured output."""
        if len(j_peaks) == 0:
            return "J: []"
        
        from datetime import datetime, timedelta
        base_time = datetime(2020, 1, 1, 0, 0, 0)
        
        j_timestamps = [(base_time + timedelta(seconds=int(idx))).strftime("%Y-%m-%d %H:%M:%S") for idx in j_peaks]
        return f"J: [{','.join(j_timestamps)}]"

    def create_instruction_entry(self, segment_data: Dict[str, Any]) -> Dict[str, str]:
        """Create instruction finetuning entry for BCG data."""
        signal = np.array(segment_data['bcg_signal'])
        true_j_peaks = np.array(segment_data['j_peaks'])
        input_peaks = self.detect_input_peaks(signal)
        
        # Calculate compression and error metrics
        compression_ratio = len(input_peaks) / len(signal) if len(signal) > 0 else 0.0
        reconstructed_signal = self.reconstruct_signal_from_peaks(input_peaks, signal)
        error_metrics = self.calculate_error_metrics(signal, reconstructed_signal, peaks=input_peaks)
        
        # Calculate R-peak recall
        recall = self.calculate_r_peak_recall(true_j_peaks, input_peaks, tolerance=30)
        
        # Update segment data with metrics
        segment_data.update({'compression_ratio': compression_ratio, 'recall': recall, **error_metrics})
        
        instruction = ("You are a specialized assistant for biomedical signal analysis, specifically trained in BCG J-peak detection. "
                      "Analyze the following BCG signal peaks using J-peak detection methodology. "
                      "Peak detection guidance: J-peaks are the prominent positive deflections in BCG signals corresponding to ventricular depolarization. "
                      "Pay attention to the timestamp, the time interval is stable for each peaks. "
                      "Output format: Structured format with J-peak positions in brackets.")
        
        duration_seconds = self.segment_length / self.sampling_rate
        input_text = f"Kansas BCG signal sampled at {self.sampling_rate} Hz with duration of {duration_seconds:.2f} seconds. Detected peaks in signal: {self.format_peaks_for_input(input_peaks, signal)}"
        
        return {
            "instruction": instruction,
            "input": input_text,
            "output": self.format_j_peaks_output(true_j_peaks),
            "compression_ratio": compression_ratio,
            "recall": recall,
            **error_metrics
        }

    

    

    def process_processed_data(self, segments: np.ndarray, labels: np.ndarray, data_type: str = "train") -> List[Dict[str, str]]:
        """Process pre-processed numpy arrays into instruction finetuning examples."""
        print(f"Processing {data_type} data: {segments.shape[0]} segments")
        
        examples = []
        for i, (segment_signal, segment_labels) in enumerate(zip(segments, labels)):
            normalized_segment = self.normalize_signal(segment_signal)
            j_peaks = np.where(segment_labels == 1)[0]
            
            if len(j_peaks) == 0:
                continue
            
            segment_data = {
                'segment_id': i,
                'start_sample': 0,
                'end_sample': len(normalized_segment),
                'start_time': 0,
                'end_time': len(normalized_segment) / self.sampling_rate,
                'bcg_signal': normalized_segment.tolist(),
                'j_peaks': j_peaks.tolist(),
                'beat_types': ['N'] * len(j_peaks),
                'lead': 'MLII',
                'data_type': data_type
            }
            
            examples.append(self.create_instruction_entry(segment_data))
        
        print(f"  - Generated {len(examples)} examples from {segments.shape[0]} segments")
        return examples

def main():
    """Main function to run the BCG dataset formatting using processed data."""
    base_path = '/path/to/workspace/project-BCG-LLM/combined_data/combined_splitted_data/BCG'
    
    # Load processed data
    training_segments = np.load(f'{base_path}/X_train.npy')
    test_segments = np.load(f'{base_path}/X_test.npy')
    training_labels = np.load(f'{base_path}/y_train.npy')
    test_labels = np.load(f'{base_path}/y_test.npy')

    print(f"Loaded processed data:")
    print(f"  - Training segments: {training_segments.shape}")
    print(f"  - Test segments: {test_segments.shape}")
    print(f"  - Training labels: {training_labels.shape}")
    print(f"  - Test labels: {test_labels.shape}")
    
    # Initialize formatter and process data
    formatter = KansasBCGDatasetFormatter(segment_length=1000, sampling_rate=100)
    
    print("\n=== Processing Training Data ===")
    training_examples = formatter.process_processed_data(training_segments, training_labels, "train")
    
    print("\n=== Processing Test Data ===")
    test_examples = formatter.process_processed_data(test_segments, test_labels, "test")
    
    # Save formatted datasets
    output_dir = "/path/to/workspace/project-BCG-LLM/combined_data/format_data/kansas_bcg_peaks"
    os.makedirs(output_dir, exist_ok=True)
    
    all_examples = training_examples + test_examples
    
    # Save all datasets
    datasets = [
        ("kansas_bcg_peaks_train.json", training_examples),
        ("kansas_bcg_peaks_test.json", test_examples),
        ("kansas_bcg_peaks_combined.json", all_examples)
    ]
    
    for filename, data in datasets:
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(data)} examples to {filepath}")
    
    # Calculate and display average metrics
    metrics = {key: [ex[key] for ex in all_examples if key in ex] 
               for key in ['compression_ratio', 'mae', 'rmse', 'mape', 'corr_coef', 'recall']}
    
    print(f"\nFormatting complete! Total examples: {len(all_examples)}")
    print(f"  - Training: {len(training_examples)}")
    print(f"  - Test: {len(test_examples)}")
    print(f"\n=== Average Metrics ===")
    print(f"  - Compression Ratio: {np.mean(metrics['compression_ratio']):.4f} ± {np.std(metrics['compression_ratio']):.4f}")
    print(f"  - MAE: {np.mean(metrics['mae']):.4f} ± {np.std(metrics['mae']):.4f}")
    print(f"  - RMSE: {np.mean(metrics['rmse']):.4f} ± {np.std(metrics['rmse']):.4f}")
    print(f"  - MAPE: {np.mean(metrics['mape']):.2f}% ± {np.std(metrics['mape']):.2f}%")
    print(f"  - Correlation Coefficient: {np.mean(metrics['corr_coef']):.4f} ± {np.std(metrics['corr_coef']):.4f}")
    print(f"  - R-peak Recall: {np.mean(metrics['recall']):.4f} ± {np.std(metrics['recall']):.4f}")

if __name__ == "__main__":
    main()
