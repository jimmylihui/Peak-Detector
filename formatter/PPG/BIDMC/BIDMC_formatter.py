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
    
    def __init__(self, segment_length: int = 1000, sampling_rate: int = 125, use_special_tokens: bool = True):
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
        if len(signal.shape) > 1:
            signal = np.mean(signal, axis=1)
            
        try:
            # Adjust min_distance for ECG sampling rate (360 Hz vs 100 Hz for BCG)
            min_distance = 10  # 300ms minimum between peaks
            pos_peaks, _ = find_peaks(signal, distance=min_distance)
            neg_peaks, _ = find_peaks(-signal, distance=min_distance)
            
            all_peaks = np.concatenate([pos_peaks, neg_peaks]) if len(pos_peaks) > 0 or len(neg_peaks) > 0 else np.array([])
            return np.sort(all_peaks) if len(all_peaks) > 0 else np.array([])
        except:
            return np.array([])

    def normalize_signal(self, signal: np.ndarray) -> np.ndarray:
        """Normalize ECG signal to zero mean and unit variance."""
        signal_mean = np.mean(signal)
        signal_std = np.std(signal)
        
        if signal_std > 0:
            return (signal - signal_mean) / signal_std
        else:
            return np.zeros_like(signal)

    def reconstruct_signal_from_peaks(self, peaks: np.ndarray, signal: np.ndarray) -> np.ndarray:
        """Reconstruct signal by interpolating from peak values."""
        if len(peaks) < 2:
            return np.zeros_like(signal)
        
        # Get peak positions and values
        peak_indices = peaks[peaks < len(signal)]
        if len(peak_indices) < 2:
            return np.zeros_like(signal)
        
        peak_values = signal[peak_indices]
        
        # Interpolate using linear interpolation
        f = interp1d(peak_indices, peak_values, kind='linear', fill_value='extrapolate')
        reconstructed = f(np.arange(len(signal)))
        
        return reconstructed
    
    def calculate_error_metrics(self, original: np.ndarray, reconstructed: np.ndarray, peaks: np.ndarray = None) -> Dict[str, float]:
        """Calculate MAE, RMSE, and MAPE between original and reconstructed signals.
        
        Parameters:
        -----------
        original : np.ndarray
            Original signal
        reconstructed : np.ndarray
            Reconstructed signal
        peaks : np.ndarray, optional
            Peak indices. If provided, only calculate metrics from first peak to last peak.
        
        Returns:
        --------
        Dict[str, float]
            Dictionary with mae, rmse, mape, and corr_coef values
        """
        # If peaks are provided, only consider range from first peak to last peak
        if peaks is not None and len(peaks) >= 2:
            first_peak = int(peaks[0])
            last_peak = int(peaks[-1])
            
            original = original[first_peak:last_peak]
            reconstructed = reconstructed[first_peak:last_peak]
        
        mae = np.mean(np.abs(original - reconstructed))
        rmse = np.sqrt(np.mean((original - reconstructed) ** 2))
        
        # MAPE with handling for zero values
        non_zero_mask = np.abs(original) > 1e-10
        if np.sum(non_zero_mask) > 0:
            mape = np.mean(np.abs((original[non_zero_mask] - reconstructed[non_zero_mask]) / original[non_zero_mask])) * 100
        else:
            mape = 0.0
        
        # Calculate correlation coefficient
        if len(original) > 1 and np.std(original) > 0 and np.std(reconstructed) > 0:
            corr_coef = np.corrcoef(original, reconstructed)[0, 1]
        else:
            corr_coef = 0.0
        
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
            if len(segment_annotations) > 0:
                segment_annotations['Relative_Sample'] = segment_annotations['Sample'] - start_idx
                r_peaks = segment_annotations['Relative_Sample'].values
                beat_types = segment_annotations['Type'].values
            else:
                r_peaks = np.array([])
                beat_types = np.array([])
            
            # Skip segments with 0 R peaks
            if len(r_peaks) == 0:
                skipped_segments += 1
                continue
            
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
        
        r_timestamps = []
        for sample_idx in r_peaks:
            # Use sample index directly as seconds (synthetic timestamp like BCG formatter)
            timestamp = base_time + timedelta(seconds=int(sample_idx))
            r_timestamps.append(timestamp.strftime("%Y-%m-%d %H:%M:%S"))
        
        return f"R: [{','.join(r_timestamps)}]"

    def create_instruction_entry(self, segment_data: Dict[str, Any]) -> Dict[str, str]:
        """Create instruction finetuning entry for MIT-BIH ECG data."""
        duration_seconds = self.segment_length / self.sampling_rate
        
        # Use true R-peaks from annotations for output
        true_r_peaks = np.array(segment_data['r_peaks'])
        signal = np.array(segment_data['ppg_signal'])
        
        # For input, we need to detect some peaks to show in the input format
        # (similar to how BCG formatter shows detected peaks in input)
        input_peaks = self.detect_input_peaks(signal)
        
        # Calculate compression ratio
        compression_ratio = len(input_peaks) / len(signal) if len(signal) > 0 else 0.0
        
        # Calculate error metrics by reconstructing signal from input peaks
        # Only consider range from first peak to last peak
        reconstructed_signal = self.reconstruct_signal_from_peaks(input_peaks, signal)
        error_metrics = self.calculate_error_metrics(signal, reconstructed_signal, peaks=input_peaks)
        
        # Calculate R-peak recall
        recall = self.calculate_r_peak_recall(true_r_peaks, input_peaks, tolerance=30)
        
        # Add metrics to segment data
        segment_data['compression_ratio'] = compression_ratio
        segment_data['mae'] = error_metrics['mae']
        segment_data['rmse'] = error_metrics['rmse']
        segment_data['mape'] = error_metrics['mape']
        segment_data['corr_coef'] = error_metrics['corr_coef']
        segment_data['recall'] = recall
        
        # Format input peaks using the same method as BCG formatter
        input_peaks_text = self.format_peaks_for_input(input_peaks, signal)
        
        # Format true R peaks output
        r_peaks_output = self.format_r_peaks_output(true_r_peaks)
        
        instruction = ("You are a specialized assistant for biomedical signal analysis, specifically trained in PPG peak detection. "
                      "Analyze the following PPG signal peaks using PPG peak detection methodology. "
                      "Peak detection guidance: PPG peaks are the prominent positive deflections in PPG signals corresponding to the heart beat. "
                      "Pay attention to the timestamp, the time interval is stable for each peaks. "
                      "Output format: Structured format with PPG peak positions in brackets.")
        
        input_text = f"PPG signal sampled at {self.sampling_rate} Hz with duration of {duration_seconds:.2f} seconds. Detected peaks in signal: {input_peaks_text}"
        
        return {
            "instruction": instruction,
            "input": input_text,
            "output": r_peaks_output,
            "compression_ratio": compression_ratio,
            "mae": error_metrics['mae'],
            "rmse": error_metrics['rmse'],
            "mape": error_metrics['mape'],
            "corr_coef": error_metrics['corr_coef'],
            "recall": recall
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
        """
        Process pre-processed numpy arrays instead of raw CSV files.
        
        Parameters:
        -----------
        segments : np.ndarray
            Pre-processed ECG segments (n_segments, segment_length)
        labels : np.ndarray
            Pre-processed labels/annotations (n_segments, segment_length) - gaussian peaks
        data_type : str
            Type of data being processed ("train" or "test")
        
        Returns:
        --------
        examples : List[Dict[str, str]]
            List of instruction finetuning examples
        """
        print(f"Processing {data_type} data: {segments.shape[0]} segments")
        
        examples = []
        
        for i, (segment_signal, segment_labels) in enumerate(zip(segments, labels)):
            # Normalize segment
            normalized_segment = self.normalize_signal(segment_signal)
            
            # Find R-peaks from the gaussian labels (where values are > 0.5)
            r_peaks = np.where(segment_labels ==1)[0]
            
            # Skip segments with 0 R peaks
            if len(r_peaks) == 0:
                continue
            
            # Create segment data structure
            segment_data = {
                'segment_id': i,
                'start_sample': 0,
                'end_sample': len(normalized_segment),
                'start_time': 0,
                'end_time': len(normalized_segment) / self.sampling_rate,
                'ppg_signal': normalized_segment.tolist(),
                'r_peaks': r_peaks.tolist(),
                'beat_types': ['N'] * len(r_peaks),  # Default beat type
                'lead': 'MLII',
                'data_type': data_type
            }
            
            # Create instruction entry
            instruction_entry = self.create_instruction_entry(segment_data)
            examples.append(instruction_entry)
        
        print(f"  - Generated {len(examples)} examples from {segments.shape[0]} segments")
        return examples

def main():
    """Main function to run the MIT-BIH dataset formatting using processed data."""
    # Load processed data
    training_segments = np.load('/path/to/workspace/project-BCG-LLM/combined_data/combined_splitted_data/PPG/processed/X_train.npy')
    test_segments = np.load('/path/to/workspace/project-BCG-LLM/combined_data/combined_splitted_data/PPG/processed/X_test.npy')
   
    
    training_labels = np.load('/path/to/workspace/project-BCG-LLM/combined_data/combined_splitted_data/PPG/processed/Y_train.npy')
    test_labels = np.load('/path/to/workspace/project-BCG-LLM/combined_data/combined_splitted_data/PPG/processed/Y_test.npy')

    print(f"Loaded processed data:")
    print(f"  - Training segments: {training_segments.shape}")
    print(f"  - Test segments: {test_segments.shape}")
    print(f"  - Training labels: {training_labels.shape}")
    print(f"  - Test labels: {test_labels.shape}")
    
   
    
    # Initialize formatter
    formatter = FetalECGDatasetFormatter(segment_length=1000, sampling_rate=125)
    
    # Process training data
    print("\n=== Processing Training Data ===")
    training_examples = formatter.process_processed_data(
        segments=training_segments,
        labels=training_labels,
        data_type="train"
    )
    
    # Process test data
    print("\n=== Processing Test Data ===")
    test_examples = formatter.process_processed_data(
        segments=test_segments,
        labels=test_labels,
        data_type="test"
    )
    
    # Save formatted datasets
    output_dir = "/path/to/workspace/project-BCG-LLM/combined_data/format_data/BIDMC_ppg_peaks"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save training data
    training_filename = os.path.join(output_dir, "BIDMC_ppg_peaks_train.json")
    with open(training_filename, 'w') as f:
        json.dump(training_examples, f, indent=2)
    print(f"Saved {len(training_examples)} training examples to {training_filename}")
    
    # Save test data
    test_filename = os.path.join(output_dir, "BIDMC_ppg_peaks_test.json")
    with open(test_filename, 'w') as f:
        json.dump(test_examples, f, indent=2)
    print(f"Saved {len(test_examples)} test examples to {test_filename}")
    
    # Save combined dataset
    all_examples = training_examples + test_examples
    combined_filename = os.path.join(output_dir, "BIDMC_ppg_peaks_combined.json")
    with open(combined_filename, 'w') as f:
        json.dump(all_examples, f, indent=2)
    print(f"Saved {len(all_examples)} total examples to {combined_filename}")
    
    # Calculate and display average metrics
    compression_ratios = [ex['compression_ratio'] for ex in all_examples if 'compression_ratio' in ex]
    maes = [ex['mae'] for ex in all_examples if 'mae' in ex]
    rmses = [ex['rmse'] for ex in all_examples if 'rmse' in ex]
    mapes = [ex['mape'] for ex in all_examples if 'mape' in ex]
    corr_coefs = [ex['corr_coef'] for ex in all_examples if 'corr_coef' in ex]
    recalls = [ex['recall'] for ex in all_examples if 'recall' in ex]
    
    print(f"\nFormatting complete! Total examples: {len(all_examples)}")
    print(f"  - Training: {len(training_examples)}")
    print(f"  - Test: {len(test_examples)}")
    print(f"\n=== Average Metrics ===")
    print(f"  - Compression Ratio: {np.mean(compression_ratios):.4f} ± {np.std(compression_ratios):.4f}")
    print(f"  - MAE: {np.mean(maes):.4f} ± {np.std(maes):.4f}")
    print(f"  - RMSE: {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")
    print(f"  - MAPE: {np.mean(mapes):.2f}% ± {np.std(mapes):.2f}%")
    print(f"  - Correlation Coefficient: {np.mean(corr_coefs):.4f} ± {np.std(corr_coefs):.4f}")
    print(f"  - R-peak Recall: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")

if __name__ == "__main__":
    main()
