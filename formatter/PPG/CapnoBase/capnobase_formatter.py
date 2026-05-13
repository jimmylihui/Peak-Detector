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

# Note: We use processed ECG data with gaussian peak annotations

class FetalECGDatasetFormatter:
    """
    Formats ECG signal data for instruction finetuning.
    Converts processed ECG signals to R-peak detection tasks with length 10000.
    """
    
    def __init__(self, segment_length: int = 1000, sampling_rate: int = 360, use_special_tokens: bool = True):
        self.segment_length = segment_length  # Length of each segment (1000 samples)
        self.sampling_rate = sampling_rate    # PPG sampling rate is 300 Hz
        self.use_special_tokens = use_special_tokens
        
        # Special tokens for time series formatting
        self.start_token = "<TS_START>"
        self.end_token = "<TS_END>"
        self.sep_token = "<TS_SEP>"
        
    def load_mitbih_record(self, record_num: str, data_path: Path):
        """
        Load MIT-BIH record from CSV format
        
        Parameters:
        -----------
        record_num : str
            Record number (e.g., '100')
        data_path : Path
            Path to the CSV data directory
        
        Returns:
        --------
        ecg_data : pd.DataFrame
            ECG signal data with cleaned column names
        annotations : pd.DataFrame
            Beat annotations with timing and classification
        """
        # Load ECG signal data
        csv_file = data_path / f'{record_num}.csv'
        if not csv_file.exists():
            raise FileNotFoundError(f"ECG data file not found: {csv_file}")
        
        ecg_data = pd.read_csv(csv_file)
        
        # Clean column names (remove quotes)
        ecg_data.columns = [col.strip("'\"") for col in ecg_data.columns]
        
        # Load annotations with proper handling
        ann_file = data_path / f'{record_num}annotations.txt'
        if not ann_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {ann_file}")
        
        # Read annotations with proper column handling
        annotations = pd.read_csv(ann_file, delim_whitespace=True, skiprows=1,
                                 names=['Time', 'Sample', 'Type', 'Sub', 'Chan', 'Num', 'Aux'])
        
        # Clean annotations (remove the leading '+' from the first annotation if present)
        if len(annotations) > 0 and annotations.iloc[0]['Type'] == '+':
            annotations = annotations.iloc[1:]  # Skip the first rhythm annotation
        
        # Convert Sample column to numeric, handling any non-numeric values
        annotations['Sample'] = pd.to_numeric(annotations['Sample'], errors='coerce')
        
        # Remove any rows where Sample conversion failed (NaN values)
        annotations = annotations.dropna(subset=['Sample'])
        
        # Convert Sample to integer
        annotations['Sample'] = annotations['Sample'].astype(int)
        
        # Add time in seconds
        annotations['Time_seconds'] = annotations['Sample'] / self.sampling_rate
        
        print(f"✓ Loaded record {record_num}")
        print(f"  - ECG samples: {len(ecg_data):,}")
        print(f"  - Duration: {len(ecg_data)/self.sampling_rate/60:.1f} minutes")
        print(f"  - Beat annotations: {len(annotations)}")
        print(f"  - ECG columns: {list(ecg_data.columns)}")
        
        return ecg_data, annotations



    def detect_input_peaks(self, signal: np.ndarray) -> np.ndarray:
        """Detect both positive and negative peaks for input formatting."""
        signal = np.mean(signal, axis=1) if len(signal.shape) > 1 else signal
        try:
            pos_peaks, _ = find_peaks(signal, distance=30)
            neg_peaks, _ = find_peaks(-signal, distance=30)
            return np.sort(np.concatenate([pos_peaks, neg_peaks])) if len(pos_peaks) + len(neg_peaks) > 0 else np.array([])
        except:
            return np.array([])

    def normalize_signal(self, signal: np.ndarray) -> np.ndarray:
        """Normalize ECG signal to zero mean and unit variance."""
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

    def extract_segments(self, ecg_data: pd.DataFrame, annotations: pd.DataFrame, lead: str = 'MLII') -> List[Dict]:
        """
        Extract fixed-length segments (1000 samples) from ECG data
        
        Parameters:
        -----------
        ecg_data : pd.DataFrame
            Full ECG signal data
        annotations : pd.DataFrame
            Beat annotations (R-peaks are already marked in Sample column)
        lead : str
            ECG lead to use ('MLII' or 'V5')
        
        Returns:
        --------
        segments : List[Dict]
            List of segment dictionaries with ECG data and true R-peak annotations
        """
        # Check if the lead exists in the data
        available_leads = [col for col in ecg_data.columns if col not in ['sample #', 'sample']]
        if lead not in available_leads:
            print(f"Lead '{lead}' not found. Available leads: {available_leads}")
            # Use the first available lead
            lead = available_leads[0] if available_leads else 'MLII'
            print(f"Using lead '{lead}' instead.")
        
        # Get ECG signal
        ecg_signal = ecg_data[lead].values
        total_samples = len(ecg_signal)
        
        segments = []
        segment_id = 0
        skipped_segments = 0
        
        # Extract non-overlapping segments of length 1000
        for start_idx in range(0, total_samples - self.segment_length + 1, self.segment_length):
            end_idx = start_idx + self.segment_length
            
            # Extract segment
            segment_signal = ecg_signal[start_idx:end_idx]
            
            # Normalize segment
            normalized_segment = self.normalize_signal(segment_signal)
            
            # Find TRUE R-peak annotations within this segment
            # The annotations already contain R-peak locations in the Sample column
            segment_annotations = annotations[
                (annotations['Sample'] >= start_idx) & 
                (annotations['Sample'] < end_idx)
            ].copy()
            
            # Adjust annotation sample indices to be relative to segment start
            if len(segment_annotations) == 0:
                skipped_segments += 1
                continue
            
            segment_annotations['Relative_Sample'] = segment_annotations['Sample'] - start_idx
            r_peaks = segment_annotations['Relative_Sample'].values
            beat_types = segment_annotations['Type'].values
            
            segment_data = {
                'segment_id': segment_id,
                'start_sample': start_idx,
                'end_sample': end_idx,
                'start_time': start_idx / self.sampling_rate,
                'end_time': end_idx / self.sampling_rate,
                'ppg_signal': normalized_segment.tolist(),
                'r_peaks': r_peaks.tolist(),  # These are TRUE R-peaks from annotations
                'beat_types': beat_types.tolist(),
                'lead': 'MLII'
            }
            
            segments.append(segment_data)
            segment_id += 1
        
        print(f"  - Extracted {len(segments)} segments of length {self.segment_length}")
        print(f"  - Skipped {skipped_segments} segments with 0 R peaks")
        
        return segments

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

    def format_r_peaks_output(self, r_peaks: np.ndarray) -> str:
        """Format R peaks for structured output."""
        if len(r_peaks) == 0:
            return "PPG: []"
        
        from datetime import datetime, timedelta
        base_time = datetime(2020, 1, 1, 0, 0, 0)
        
        r_timestamps = [(base_time + timedelta(seconds=int(idx))).strftime("%Y-%m-%d %H:%M:%S") for idx in r_peaks]
        return f"R: [{','.join(r_timestamps)}]"

    def create_instruction_entry(self, segment_data: Dict[str, Any]) -> Dict[str, str]:
        """Create instruction finetuning entry for PPG data."""
        signal = np.array(segment_data['ppg_signal'])
        true_r_peaks = np.array(segment_data['r_peaks'])
        input_peaks = self.detect_input_peaks(signal)
        
        # Calculate compression and error metrics
        compression_ratio = len(input_peaks) / len(signal) if len(signal) > 0 else 0.0
        reconstructed_signal = self.reconstruct_signal_from_peaks(input_peaks, signal)
        error_metrics = self.calculate_error_metrics(signal, reconstructed_signal, peaks=input_peaks)
        
        # Calculate R-peak recall
        recall = self.calculate_r_peak_recall(true_r_peaks, input_peaks, tolerance=30)
        
        # Update segment data with metrics
        segment_data.update({'compression_ratio': compression_ratio, 'recall': recall, **error_metrics})
        
        instruction = ("You are a specialized assistant for biomedical signal analysis, specifically trained in PPG peak detection. "
                      "Analyze the following PPG signal peaks using PPG peak detection methodology. "
                      "Peak detection guidance: PPG peaks are the prominent positive deflections in PPG signals corresponding to the heart beat. Only detect positive peaks. "
                      "Pay attention to the timestamp, the time interval is stable for each peaks. "
                      "Output format: Structured format with PPG peak positions in brackets.")
        
        duration_seconds = self.segment_length / self.sampling_rate
        input_text = f"PPG signal sampled at {self.sampling_rate} Hz with duration of {duration_seconds:.2f} seconds. Detected peaks in signal: {self.format_peaks_for_input(input_peaks, signal)}"
        
        return {
            "instruction": instruction,
            "input": input_text,
            "output": self.format_r_peaks_output(true_r_peaks),
            "compression_ratio": compression_ratio,
            "recall": recall,
            **error_metrics
        }

    def process_record_data(self, record_num: str, ecg_data: pd.DataFrame, annotations: pd.DataFrame, 
                           lead: str = 'MLII') -> List[Dict[str, Any]]:
        """Process a single MIT-BIH record's data."""
        print(f"Processing Record {record_num}: shape = {ecg_data.shape}")
        
        # Extract segments
        segments = self.extract_segments(ecg_data, annotations, lead)
        
        processed_segments = []
        for segment_data in segments:
            # Add record information
            segment_data['record_num'] = record_num
            processed_segments.append(segment_data)
        
        return processed_segments

    


    def process_processed_data(self, segments: np.ndarray, labels: np.ndarray, data_type: str = "train") -> List[Dict[str, str]]:
        """Process pre-processed numpy arrays into instruction finetuning examples."""
        print(f"Processing {data_type} data: {segments.shape[0]} segments")
        
        examples = []
        for i, (segment_signal, segment_labels) in enumerate(zip(segments, labels)):
            normalized_segment = self.normalize_signal(segment_signal)
            r_peaks = np.where(segment_labels == 1)[0]
            
            if len(r_peaks) == 0:
                continue
            
            segment_data = {
                'segment_id': i,
                'start_sample': 0,
                'end_sample': len(normalized_segment),
                'start_time': 0,
                'end_time': len(normalized_segment) / self.sampling_rate,
                'ppg_signal': normalized_segment.tolist(),
                'r_peaks': r_peaks.tolist(),
                'beat_types': ['N'] * len(r_peaks),
                'lead': 'MLII',
                'data_type': data_type
            }
            
            examples.append(self.create_instruction_entry(segment_data))
        
        print(f"  - Generated {len(examples)} examples from {segments.shape[0]} segments")
        return examples

def main():
    """Main function to run the PPG dataset formatting using processed data."""
    base_path = '/path/to/workspace/project-BCG-LLM/combined_data/combined_splitted_data/PPG_capnobase/processed'
    
    # Load processed data
    training_segments = np.load(f'{base_path}/X_train.npy')
    test_segments = np.load(f'{base_path}/X_test.npy')
    training_labels = np.load(f'{base_path}/Y_train_gauss.npy')
    test_labels = np.load(f'{base_path}/Y_test_gauss.npy')

    print(f"Loaded processed data:")
    print(f"  - Training segments: {training_segments.shape}")
    print(f"  - Test segments: {test_segments.shape}")
    print(f"  - Training labels: {training_labels.shape}")
    print(f"  - Test labels: {test_labels.shape}")
    
    # Initialize formatter and process data
    formatter = FetalECGDatasetFormatter(segment_length=1000, sampling_rate=300)
    
    print("\n=== Processing Training Data ===")
    training_examples = formatter.process_processed_data(training_segments, training_labels, "train")
    
    print("\n=== Processing Test Data ===")
    test_examples = formatter.process_processed_data(test_segments, test_labels, "test")
    
    # Save formatted datasets
    output_dir = "/path/to/workspace/project-BCG-LLM/combined_data/format_data/capnobase_ppg_peaks"
    os.makedirs(output_dir, exist_ok=True)
    
    all_examples = training_examples + test_examples
    
    # Save all datasets
    datasets = [
        ("capnobase_ppg_peaks_train.json", training_examples),
        ("capnobase_ppg_peaks_test.json", test_examples),
        ("capnobase_ppg_peaks_combined.json", all_examples)
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
