#!/usr/bin/env python3
"""
BIDMC Dataset Processing Script

This script processes the BIDMC dataset by:
1. Loading ECG, PPG, and respiratory signals for each subject
2. Splitting signals into non-overlapping 1000-length segments
3. Storing as numpy arrays with shape (n, 1000) where n is the number of segments
4. Saving processed data for each subject

Author: AI Assistant
Date: 2024
"""

import numpy as np
import pandas as pd
from pathlib import Path
import os
import sys
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class BIDMCProcessor:
    def __init__(self, 
                 dataset_path="/path/to/workspace/PPG_peaks/dataset/BIDMC/bidmc-ppg-and-respiration-dataset-1.0.0",
                 output_path="/path/to/workspace/PPG_peaks/processed_dataset/BIDMC",
                 segment_length=1000):
        """
        Initialize BIDMC dataset processor
        
        Parameters:
            dataset_path: Path to the BIDMC dataset
            output_path: Path to save processed data
            segment_length: Length of each signal segment (default: 1000)
        """
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.segment_length = segment_length
        self.csv_path = self.dataset_path / "bidmc_csv"
        
        # Create output directory if it doesn't exist
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Verify dataset exists
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        
        print(f"BIDMC Processor initialized:")
        print(f"  Dataset path: {self.dataset_path}")
        print(f"  Output path: {self.output_path}")
        print(f"  Segment length: {self.segment_length}")
    
    def get_available_subjects(self):
        """Get list of available subject IDs"""
        csv_files = list(self.csv_path.glob("bidmc_*_Signals.csv"))
        subjects = [f.stem.split('_')[1] for f in csv_files]
        return sorted(subjects, key=lambda x: int(x))
    
    def read_signals(self, subject_id):
        """
        Read physiological signals for a subject
        
        Returns:
            DataFrame with columns: Time, RESP, PLETH, V, AVR, II
        """
        signals_file = self.csv_path / f"bidmc_{subject_id}_Signals.csv"
        
        if not signals_file.exists():
            raise FileNotFoundError(f"Signals file not found for subject {subject_id}")
        
        df = pd.read_csv(signals_file)
        # Fix column names by removing extra spaces
        df.columns = df.columns.str.strip()
        
        return df
    
    def split_signal(self, signal, segment_length):
        """
        Split a signal into non-overlapping segments
        
        Parameters:
            signal: 1D numpy array
            segment_length: Length of each segment
            
        Returns:
            segments: 2D numpy array of shape (n_segments, segment_length)
        """
        # Calculate number of complete segments
        n_segments = len(signal) // segment_length
        
        if n_segments == 0:
            return np.array([]).reshape(0, segment_length)
        
        # Reshape signal into segments
        segments = signal[:n_segments * segment_length].reshape(n_segments, segment_length)
        
        return segments
    
    def process_subject(self, subject_id):
        """
        Process signals for a single subject
        
        Parameters:
            subject_id: Subject ID (e.g., "01", "02", etc.)
            
        Returns:
            dict: Dictionary containing processed data
        """
        print(f"Processing subject {subject_id}...")
        
        try:
            # Read signals
            signals_df = self.read_signals(subject_id)
            
            # Extract ECG, PPG, and respiratory signals
            # Using Lead II ECG (most commonly used for analysis)
            ecg_signal = signals_df['II'].values
            ppg_signal = signals_df['PLETH'].values
            resp_signal = signals_df['RESP'].values
            
            print(f"  Original signal length: {len(ecg_signal)} samples")
            print(f"  Duration: {len(ecg_signal) / 125:.1f} seconds (at 125 Hz)")
            
            # Split signals into segments
            ecg_segments = self.split_signal(ecg_signal, self.segment_length)
            ppg_segments = self.split_signal(ppg_signal, self.segment_length)
            resp_segments = self.split_signal(resp_signal, self.segment_length)
            
            print(f"  ECG segments: {ecg_segments.shape}")
            print(f"  PPG segments: {ppg_segments.shape}")
            print(f"  RESP segments: {resp_segments.shape}")
            
            # Ensure all signals have the same number of segments
            min_segments = min(len(ecg_segments), len(ppg_segments), len(resp_segments))
            if min_segments == 0:
                print(f"  Warning: No complete segments for subject {subject_id}")
                return None
            
            ecg_segments = ecg_segments[:min_segments]
            ppg_segments = ppg_segments[:min_segments]
            resp_segments = resp_segments[:min_segments]
            
            # Create result dictionary
            result = {
                'subject_id': subject_id,
                'ecg_segments': ecg_segments,
                'ppg_segments': ppg_segments,
                'resp_segments': resp_segments,
                'n_segments': min_segments,
                'segment_length': self.segment_length,
                'sampling_rate': 125,  # BIDMC dataset sampling rate
                'total_duration': len(ecg_signal) / 125,
                'processed_duration': min_segments * self.segment_length / 125
            }
            
            print(f"  ✅ Successfully processed {min_segments} segments")
            return result
            
        except Exception as e:
            print(f"  ❌ Error processing subject {subject_id}: {e}")
            return None
    
    def save_processed_data(self, processed_data, subject_id):
        """
        Save processed data for a subject
        
        Parameters:
            processed_data: Dictionary containing processed data
            subject_id: Subject ID
        """
        if processed_data is None:
            return
        
        # Create subject-specific directory
        subject_dir = self.output_path / f"subject_{subject_id}"
        subject_dir.mkdir(exist_ok=True)
        
        # Save ECG segments
        ecg_file = subject_dir / "ecg_segments.npy"
        np.save(ecg_file, processed_data['ecg_segments'])
        
        # Save PPG segments
        ppg_file = subject_dir / "ppg_segments.npy"
        np.save(ppg_file, processed_data['ppg_segments'])
        
        # Save respiratory segments
        resp_file = subject_dir / "resp_segments.npy"
        np.save(resp_file, processed_data['resp_segments'])
        
        # Save metadata
        metadata = {
            'subject_id': processed_data['subject_id'],
            'n_segments': processed_data['n_segments'],
            'segment_length': processed_data['segment_length'],
            'sampling_rate': processed_data['sampling_rate'],
            'total_duration': processed_data['total_duration'],
            'processed_duration': processed_data['processed_duration'],
            'ecg_shape': processed_data['ecg_segments'].shape,
            'ppg_shape': processed_data['ppg_segments'].shape,
            'resp_shape': processed_data['resp_segments'].shape
        }
        
        metadata_file = subject_dir / "metadata.npy"
        np.save(metadata_file, metadata)
        
        print(f"  💾 Saved data to {subject_dir}")
    
    def process_all_subjects(self):
        """
        Process all subjects in the dataset
        """
        subjects = self.get_available_subjects()
        print(f"Found {len(subjects)} subjects to process")
        
        # Summary statistics
        total_segments = 0
        successful_subjects = 0
        failed_subjects = []
        
        # Process each subject
        for subject_id in tqdm(subjects, desc="Processing subjects"):
            processed_data = self.process_subject(subject_id)
            
            if processed_data is not None:
                self.save_processed_data(processed_data, subject_id)
                total_segments += processed_data['n_segments']
                successful_subjects += 1
            else:
                failed_subjects.append(subject_id)
        
        # Print summary
        print("\n" + "="*60)
        print("PROCESSING SUMMARY")
        print("="*60)
        print(f"Total subjects: {len(subjects)}")
        print(f"Successfully processed: {successful_subjects}")
        print(f"Failed: {len(failed_subjects)}")
        print(f"Total segments created: {total_segments}")
        print(f"Total duration processed: {total_segments * self.segment_length / 125 / 60:.1f} minutes")
        
        if failed_subjects:
            print(f"Failed subjects: {failed_subjects}")
        
        # Save overall summary
        summary = {
            'total_subjects': len(subjects),
            'successful_subjects': successful_subjects,
            'failed_subjects': failed_subjects,
            'total_segments': total_segments,
            'segment_length': self.segment_length,
            'sampling_rate': 125,
            'total_duration_minutes': total_segments * self.segment_length / 125 / 60
        }
        
        summary_file = self.output_path / "processing_summary.npy"
        np.save(summary_file, summary)
        print(f"\n💾 Processing summary saved to {summary_file}")
    
    def load_processed_subject(self, subject_id):
        """
        Load processed data for a specific subject
        
        Parameters:
            subject_id: Subject ID
            
        Returns:
            dict: Dictionary containing loaded data
        """
        subject_dir = self.output_path / f"subject_{subject_id}"
        
        if not subject_dir.exists():
            raise FileNotFoundError(f"Processed data not found for subject {subject_id}")
        
        # Load data
        ecg_segments = np.load(subject_dir / "ecg_segments.npy")
        ppg_segments = np.load(subject_dir / "ppg_segments.npy")
        resp_segments = np.load(subject_dir / "resp_segments.npy")
        metadata = np.load(subject_dir / "metadata.npy", allow_pickle=True).item()
        
        return {
            'ecg_segments': ecg_segments,
            'ppg_segments': ppg_segments,
            'resp_segments': resp_segments,
            'metadata': metadata
        }

def main():
    """Main function to run the processing"""
    print("BIDMC Dataset Processing")
    print("="*50)
    
    # Initialize processor
    processor = BIDMCProcessor(
        dataset_path="/path/to/workspace/project-BCG-LLM/PPG_peaks/dataset/BIDMC/bidmc-ppg-and-respiration-dataset-1.0.0",
        output_path="/path/to/workspace/project-BCG-LLM/PPG_peaks/processed_dataset/BIDMC",
        segment_length=1000
    )
    
    # Process all subjects
    processor.process_all_subjects()
    
    print("\n🎉 Processing completed successfully!")

if __name__ == "__main__":
    main()
