#!/usr/bin/env python3
"""
CapnoBase Dataset Processing Script

This script processes the CapnoBase dataset by:
1. Loading ECG and PPG signals for each subject
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

class CapnoBaseProcessor:
    def __init__(self, 
                 dataset_path="/path/to/workspace/project-BCG-LLM/PPG_peaks/dataset/capnobase/data",
                 output_path="/path/to/workspace/project-BCG-LLM/PPG_peaks/processed_dataset/CapnoBase",
                 segment_length=1000):
        """
        Initialize CapnoBase dataset processor
        
        Parameters:
            dataset_path: Path to the CapnoBase dataset
            output_path: Path to save processed data
            segment_length: Length of each signal segment (default: 1000)
        """
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.segment_length = segment_length
        self.csv_path = self.dataset_path / "csv"
        
        # Create output directory if it doesn't exist
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Verify dataset exists
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        
        print(f"CapnoBase Processor initialized:")
        print(f"  Dataset path: {self.dataset_path}")
        print(f"  Output path: {self.output_path}")
        print(f"  Segment length: {self.segment_length}")
    
    def get_available_subjects(self):
        """Get list of available subject IDs"""
        signal_files = list(self.csv_path.glob("*_signal.csv"))
        subjects = [f.stem.replace("_signal", "") for f in signal_files]
        return sorted(subjects)
    
    def read_signals(self, subject_id):
        """
        Read physiological signals for a subject
        
        Returns:
            DataFrame with columns: co2_y, pleth_y, ecg_y
        """
        signals_file = self.csv_path / f"{subject_id}_signal.csv"
        
        if not signals_file.exists():
            raise FileNotFoundError(f"Signals file not found for subject {subject_id}")
        
        df = pd.read_csv(signals_file)
        
        return df
    
    def read_parameters(self, subject_id):
        """
        Read recording parameters for a subject
        
        Returns:
            DataFrame with recording parameters
        """
        param_file = self.csv_path / f"{subject_id}_param.csv"
        
        if not param_file.exists():
            print(f"Warning: Parameter file not found for subject {subject_id}")
            return None
        
        df = pd.read_csv(param_file)
        return df
    
    def read_metadata(self, subject_id):
        """
        Read subject metadata
        
        Returns:
            DataFrame with subject metadata
        """
        meta_file = self.csv_path / f"{subject_id}_meta.csv"
        
        if not meta_file.exists():
            print(f"Warning: Metadata file not found for subject {subject_id}")
            return None
        
        df = pd.read_csv(meta_file)
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
            subject_id: Subject ID (e.g., "0009_8min", "0015_8min", etc.)
            
        Returns:
            dict: Dictionary containing processed data
        """
        print(f"Processing subject {subject_id}...")
        
        try:
            # Read signals
            signals_df = self.read_signals(subject_id)
            
            # Read parameters and metadata
            param_df = self.read_parameters(subject_id)
            meta_df = self.read_metadata(subject_id)
            
            # Extract ECG and PPG signals
            ecg_signal = signals_df['ecg_y'].values
            ppg_signal = signals_df['pleth_y'].values
            co2_signal = signals_df['co2_y'].values
            
            # Get sampling rate from parameters (default to 300 Hz if not available)
            sampling_rate = 300
            if param_df is not None and 'samplingrate_ecg' in param_df.columns:
                sampling_rate = param_df.iloc[0]['samplingrate_ecg']
            
            print(f"  Original signal length: {len(ecg_signal)} samples")
            print(f"  Duration: {len(ecg_signal) / sampling_rate / 60:.1f} minutes (at {sampling_rate} Hz)")
            
            # Split signals into segments
            ecg_segments = self.split_signal(ecg_signal, self.segment_length)
            ppg_segments = self.split_signal(ppg_signal, self.segment_length)
            co2_segments = self.split_signal(co2_signal, self.segment_length)
            
            print(f"  ECG segments: {ecg_segments.shape}")
            print(f"  PPG segments: {ppg_segments.shape}")
            print(f"  CO2 segments: {co2_segments.shape}")
            
            # Ensure all signals have the same number of segments
            min_segments = min(len(ecg_segments), len(ppg_segments), len(co2_segments))
            if min_segments == 0:
                print(f"  Warning: No complete segments for subject {subject_id}")
                return None
            
            ecg_segments = ecg_segments[:min_segments]
            ppg_segments = ppg_segments[:min_segments]
            co2_segments = co2_segments[:min_segments]
            
            # Extract metadata information
            metadata_info = {}
            if meta_df is not None:
                metadata_info = {
                    'subject_age': meta_df.iloc[0]['subject_age'] if 'subject_age' in meta_df.columns else None,
                    'subject_weight': meta_df.iloc[0]['subject_weight'] if 'subject_weight' in meta_df.columns else None,
                    'subject_gender': meta_df.iloc[0]['subject_gender'] if 'subject_gender' in meta_df.columns else None,
                    'treatment_ventilation': meta_df.iloc[0]['treatment_ventilation'] if 'treatment_ventilation' in meta_df.columns else None
                }
            
            if param_df is not None:
                metadata_info.update({
                    'case_ventilation': param_df.iloc[0]['case_ventilation'] if 'case_ventilation' in param_df.columns else None
                })
            
            # Create result dictionary
            result = {
                'subject_id': subject_id,
                'ecg_segments': ecg_segments,
                'ppg_segments': ppg_segments,
                'co2_segments': co2_segments,
                'n_segments': min_segments,
                'segment_length': self.segment_length,
                'sampling_rate': sampling_rate,
                'total_duration': len(ecg_signal) / sampling_rate,
                'processed_duration': min_segments * self.segment_length / sampling_rate,
                'metadata': metadata_info
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
        
        # Save CO2 segments
        co2_file = subject_dir / "co2_segments.npy"
        np.save(co2_file, processed_data['co2_segments'])
        
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
            'co2_shape': processed_data['co2_segments'].shape,
            'subject_metadata': processed_data['metadata']
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
        print(f"Total duration processed: {total_segments * self.segment_length / 300 / 60:.1f} minutes")
        
        if failed_subjects:
            print(f"Failed subjects: {failed_subjects}")
        
        # Save overall summary
        summary = {
            'total_subjects': len(subjects),
            'successful_subjects': successful_subjects,
            'failed_subjects': failed_subjects,
            'total_segments': total_segments,
            'segment_length': self.segment_length,
            'sampling_rate': 300,
            'total_duration_minutes': total_segments * self.segment_length / 300 / 60
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
        co2_segments = np.load(subject_dir / "co2_segments.npy")
        metadata = np.load(subject_dir / "metadata.npy", allow_pickle=True).item()
        
        return {
            'ecg_segments': ecg_segments,
            'ppg_segments': ppg_segments,
            'co2_segments': co2_segments,
            'metadata': metadata
        }
    
    def get_dataset_statistics(self):
        """
        Get overall statistics about the processed dataset
        
        Returns:
            dict: Dataset statistics
        """
        subjects = self.get_available_subjects()
        stats = {
            'total_subjects': len(subjects),
            'segment_length': self.segment_length,
            'sampling_rate': 300,
            'subjects': []
        }
        
        for subject_id in subjects:
            try:
                subject_dir = self.output_path / f"subject_{subject_id}"
                if subject_dir.exists():
                    metadata = np.load(subject_dir / "metadata.npy", allow_pickle=True).item()
                    stats['subjects'].append({
                        'subject_id': subject_id,
                        'n_segments': metadata['n_segments'],
                        'duration_minutes': metadata['total_duration'] / 60,
                        'metadata': metadata['subject_metadata']
                    })
            except Exception as e:
                print(f"Error loading stats for subject {subject_id}: {e}")
        
        return stats

def main():
    """Main function to run the processing"""
    print("CapnoBase Dataset Processing")
    print("="*50)
    
    # Initialize processor
    processor = CapnoBaseProcessor(
        dataset_path="/path/to/workspace/project-BCG-LLM/PPG_peaks/dataset/capnobase/data",
        output_path="/path/to/workspace/project-BCG-LLM/PPG_peaks/processed_dataset/CapnoBase",
        segment_length=1000
    )
    
    # Process all subjects
    processor.process_all_subjects()
    
    print("\n🎉 Processing completed successfully!")

if __name__ == "__main__":
    main()
