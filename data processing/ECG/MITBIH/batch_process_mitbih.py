#!/usr/bin/env python3
"""
Batch process all MIT-BIH records to create:
1. 1D series for MLII lead
2. Binary sequence for arrhythmia vs normal classification

This script processes all available IDs in the MIT-BIH database.
"""

import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path

def get_all_ids(database_dir):
    """Get all available IDs from the database directory"""
    csv_files = glob.glob(os.path.join(database_dir, "*.csv"))
    ids = []
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        id_num = filename.replace('.csv', '')
        ids.append(id_num)
    return sorted(ids, key=int)

def read_ecg_data(csv_file):
    """Read ECG data from CSV file"""
    print(f"Reading ECG data from {csv_file}")
    try:
        df = pd.read_csv(csv_file)
        print(f"ECG data shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        return None

def read_annotations(annotation_file):
    """Read annotation data from text file"""
    print(f"Reading annotations from {annotation_file}")
    
    try:
        # Read the annotation file, skipping the header
        with open(annotation_file, 'r') as f:
            lines = f.readlines()
        
        # Parse annotations
        annotations = []
        for line in lines[1:]:  # Skip header
            if line.strip():
                parts = line.split()
                if len(parts) >= 3:
                    time_str = parts[0]
                    sample_num = int(parts[1])
                    annotation_type = parts[2]
                    annotations.append({
                        'time': time_str,
                        'sample': sample_num,
                        'type': annotation_type
                    })
        
        df_annotations = pd.DataFrame(annotations)
        print(f"Annotations shape: {df_annotations.shape}")
        print(f"Annotation types: {df_annotations['type'].value_counts()}")
        
        return df_annotations
    except Exception as e:
        print(f"Error reading {annotation_file}: {e}")
        return None

def is_r_peak_explicit_annotated(symbol):
    """
    显式判断符号是否为R波峰值
    包含详细注释说明每个符号的含义
    """
    if symbol == 'N' or symbol == '/':
        return True
    # ========== 传导异常 ==========
    elif symbol == 'L':
        # 左束支传导阻滞心搏
        return True
    elif symbol == 'R':
        # 右束支传导阻滞心搏
        return True
    elif symbol == 'B':
        # 束支传导阻滞心搏（未指定类型）
        return True
    
    # ========== 室上性异位搏动 ==========
    elif symbol == 'a':
        # 畸变的房性期前收缩
        return True
    elif symbol == 'J':
        # 交界性（结性）期前收缩
        return True
    elif symbol == 'S':
        # 室上性期前收缩或异位搏动
        return True
    
    # ========== 室性异位搏动 ==========
    elif symbol == 'r':
        # R-on-T型室性期前收缩
        return True
    
    # ========== 融合波 ==========
    elif symbol == 'F':
        # 心室搏动与正常搏动的融合波
        return True
    elif symbol == 'f':
        # 起搏心搏与正常心搏的融合波
        return True
    
    # ========== 逸搏 ==========
    elif symbol == 'e':
        # 房性逸搏
        return True
    elif symbol == 'j':
        # 交界性（结性）逸搏
        return True
    elif symbol == 'E':
        # 心室逸搏
        return True
    elif symbol == 'n':
        # 室上性逸搏
        return True
    
    
    # ========== 未分类/特殊 ==========
    elif symbol == 'Q':
        # 无法分类的心搏
        return True
    elif symbol == '?':
        # 学习过程中未分类的心搏
        return True
    elif symbol == 'x':
        # 未传导的P波（阻滞的房性期前收缩）
        return True
    
    # ========== 非R波峰值符号 ==========
    else:
        # 包括: p, t, u, (, ), `, ', ^, |, ~, +, s, T, *, D, =, ", @
        # 以及所有节律注释 ((AFIB, (AFL, (VT, 等)
        # 和其他事件标记 (M, MISSB, P, PSE, T, TS, U, 等)
        return False
def create_binary_labels(ecg_data, annotations):
    """Create binary labels for arrhythmia vs normal"""
    print("Creating binary labels...")
    
    # Initialize all samples as other/unknown (0)
    binary_labels = np.zeros(len(ecg_data), dtype=int)
    
    # Map 'N' (Normal beat) -> 1, 'A' (Atrial premature beat) -> 2
    for _, ann in annotations.iterrows():
        if ann['sample'] < len(binary_labels):
            if ann['type'] == 'A' or ann['type'] == 'V':
                binary_labels[ann['sample']] = 2
            elif is_r_peak_explicit_annotated(ann['type']):
                binary_labels[ann['sample']] = 1
    
    print(f"Total samples: {len(binary_labels)}")
    print(f"Other/unknown (0): {np.sum(binary_labels == 0)}")
    print(f"Normal 'N' or '.' (1): {np.sum(binary_labels == 1)}")
    print(f"Atrial 'A' or R peak (2): {np.sum(binary_labels == 2)}")
    
    return binary_labels

def save_processed_data(ecg_series, binary_labels, output_dir, record_id, lead_name):
    """Save processed data to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save ECG series
    ecg_file = os.path.join(output_dir, f'{record_id}_{lead_name.lower()}_series.npy')
    np.save(ecg_file, ecg_series)
    print(f"Saved {lead_name} series to {ecg_file}")
    
    # Save binary labels
    labels_file = os.path.join(output_dir, f'{record_id}_binary_labels.npy')
    np.save(labels_file, binary_labels)
    print(f"Saved binary labels to {labels_file}")
    
    # Save as CSV for easy inspection
    df_output = pd.DataFrame({
        'sample': range(len(ecg_series)),
        'ecg_signal': ecg_series,
        'label': binary_labels
    })
    
    csv_file = os.path.join(output_dir, f'{record_id}_processed_data.csv')
    df_output.to_csv(csv_file, index=False)
    print(f"Saved combined data to {csv_file}")
    
    return ecg_file, labels_file, csv_file

def process_single_record(record_id, database_dir, output_base_dir):
    """Process a single MIT-BIH record"""
    print(f"\n{'='*60}")
    print(f"Processing record {record_id}")
    print(f"{'='*60}")
    
    # File paths
    csv_file = os.path.join(database_dir, f"{record_id}.csv")
    annotation_file = os.path.join(database_dir, f"{record_id}annotations.txt")
    output_dir = os.path.join(output_base_dir, f"processed_{record_id}")
    
    # Check if files exist
    if not os.path.exists(csv_file):
        print(f"CSV file not found: {csv_file}")
        return False
    
    if not os.path.exists(annotation_file):
        print(f"Annotation file not found: {annotation_file}")
        return False
    
    try:
        # Read data
        ecg_data = read_ecg_data(csv_file)
        if ecg_data is None:
            return False
            
        annotations = read_annotations(annotation_file)
        if annotations is None:
            return False
        
        # Extract ECG series - prefer MLII, but use first available lead if MLII not present
        if "'MLII'" in ecg_data.columns:
            ecg_series = ecg_data["'MLII'"].values
            lead_name = "MLII"
        elif "MLII" in ecg_data.columns:
            ecg_series = ecg_data["MLII"].values
            lead_name = "MLII"
        else:
            # Use the first available lead (excluding sample number column)
            available_columns = [col for col in ecg_data.columns if "'sample #'" not in col and "sample" not in col.lower()]
            if available_columns:
                lead_name = available_columns[0].strip("'")
                ecg_series = ecg_data[available_columns[0]].values
                print(f"MLII not available, using {lead_name} lead instead")
            else:
                print(f"No ECG lead columns found in {csv_file}")
                print(f"Available columns: {ecg_data.columns.tolist()}")
                return False
            
        print(f"ECG series shape: {ecg_series.shape}")
        print(f"ECG range: {ecg_series.min()} to {ecg_series.max()}")
        print(f"Using lead: {lead_name}")
        
        # Create binary labels
        binary_labels = create_binary_labels(ecg_data, annotations)
        
        # Save processed data
        ecg_file, labels_file, csv_file = save_processed_data(ecg_series, binary_labels, output_dir, record_id, lead_name)
        
        print(f"Successfully processed record {record_id}")
        return True
        
    except Exception as e:
        print(f"Error processing record {record_id}: {e}")
        return False

def create_summary_report(processed_records, output_base_dir):
    """Create a summary report of all processed records"""
    summary_file = os.path.join(output_base_dir, "processing_summary.txt")
    
    with open(summary_file, 'w') as f:
        f.write("MIT-BIH Database Processing Summary\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total records processed: {len(processed_records)}\n")
        f.write(f"Successfully processed: {sum(processed_records.values())}\n")
        f.write(f"Failed: {len(processed_records) - sum(processed_records.values())}\n\n")
        
        f.write("Processing Results:\n")
        f.write("-" * 20 + "\n")
        for record_id, success in processed_records.items():
            status = "SUCCESS" if success else "FAILED"
            f.write(f"Record {record_id}: {status}\n")
    
    print(f"\nSummary report saved to {summary_file}")

def main():
    """Main batch processing function"""
    # Directory paths
    database_dir = '/path/to/workspace/project-BCG-LLM/ECG_peak/dataset/mitbih_database'
    output_base_dir = '/path/to/workspace/project-BCG-LLM/ECG_peak/data_process/processed_records'
    
    # Get all available IDs
    all_ids = get_all_ids(database_dir)
    print(f"Found {len(all_ids)} records to process: {all_ids}")
    
    # Process each record
    processed_records = {}
    
    for record_id in all_ids:
        success = process_single_record(record_id, database_dir, output_base_dir)
        processed_records[record_id] = success
    
    # Create summary report
    create_summary_report(processed_records, output_base_dir)
    
    print(f"\n{'='*60}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total records: {len(all_ids)}")
    print(f"Successfully processed: {sum(processed_records.values())}")
    print(f"Failed: {len(all_ids) - sum(processed_records.values())}")
    print(f"Output directory: {output_base_dir}")

if __name__ == "__main__":
    main()
