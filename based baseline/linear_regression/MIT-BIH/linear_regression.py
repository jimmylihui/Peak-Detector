"""
R-peak classification from detected peaks using various ML algorithms.

This script:
1. Extracts all peaks using scipy.signal.find_peaks
2. Extracts features (position, height) from each peak
3. Uses various ML algorithms to classify peaks as R-peaks
4. Evaluates the classification performance

Available algorithms: Random Forest, SVM, XGBoost, Gradient Boosting, KNN, Logistic Regression
"""

import numpy as np
from scipy.signal import find_peaks
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm
import os

# Try to import XGBoost, use None if not available
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    XGBClassifier = None

# Data paths
x_train_path = '/path/to/workspace/project-BCG-LLM/combined_data/combined_splitted_data/ECG/total_X_train.npy'
y_train_path = '/path/to/workspace/project-BCG-LLM/combined_data/combined_splitted_data/ECG/total_y_train.npy'
x_test_path = '/path/to/workspace/project-BCG-LLM/combined_data/combined_splitted_data/ECG/total_X_test.npy'
y_test_path = '/path/to/workspace/project-BCG-LLM/combined_data/combined_splitted_data/ECG/total_y_test.npy'


def extract_all_peaks(signal, distance=10, prominence=None):
    """
    Extract all peaks from a signal using find_peaks.
    
    Args:
        signal: 1D numpy array of signal values
        distance: Minimum distance between peaks (samples)
        prominence: Minimum prominence of peaks. If None, uses adaptive value.
    
    Returns:
        peaks: Array of peak indices
        peak_heights: Array of peak heights (signal values at peak positions)
    """
    # Use adaptive prominence if not specified (5% of signal range)
    if prominence is None:
        signal_range = np.max(signal) - np.min(signal)
        prominence = signal_range * 0.05
    
    # Find peaks
    peaks, _ = find_peaks(signal, distance=10)
    
    # Get peak heights
    peak_heights = signal[peaks] if len(peaks) > 0 else np.array([])
    
    return peaks, peak_heights


def extract_peak_features(signal, peaks, peak_heights):
    """
    Extract features from detected peaks: position and height.
    
    Args:
        signal: 1D numpy array of signal values
        peaks: Array of peak indices
        peak_heights: Array of peak heights
    
    Returns:
        features: Array of shape (n_peaks, 2) containing [position, height] for each peak
    """
    if len(peaks) == 0:
        return np.array([]).reshape(0, 2)
    
    # Normalize position to [0, 1] range
    normalized_positions = peaks / len(signal)
    
    # Normalize heights relative to signal range
    signal_range = np.max(signal) - np.min(signal)
    normalized_heights = peak_heights / signal_range if signal_range > 0 else peak_heights
    
    # Combine features: [normalized_position, normalized_height]
    features = np.column_stack([normalized_positions, peak_heights])
    
    return features


def create_training_data_from_segments(signals, labels, distance=10):
    """
    Extract peaks and create training data from multiple signal segments.
    
    Args:
        signals: Array of shape (n_segments, segment_length)
        labels: Array of shape (n_segments, segment_length) with R-peak annotations (1 for R-peak, 0 otherwise)
        distance: Minimum distance between peaks
    
    Returns:
        X: Feature matrix of shape (n_total_peaks, 2) with [position, height] features
        y: Binary labels of shape (n_total_peaks,) with 1 for R-peaks, 0 for non-R-peaks
    """
    all_features = []
    all_labels = []
    
    print(f"Processing {len(signals)} segments to extract peaks...")
    for signal, label in tqdm(zip(signals, labels), total=len(signals)):
        # Extract all peaks
        peaks, peak_heights = extract_all_peaks(signal, distance=distance)
        
        if len(peaks) == 0:
            continue
        
        # Extract features
        features = extract_peak_features(signal, peaks, peak_heights)
        
        # Get ground truth labels for each peak (check if peak is within tolerance of an R-peak)
        tolerance = 10  # samples
        peak_labels = np.zeros(len(peaks), dtype=int)
        r_peak_indices = np.where(label == 1)[0]
        
        for i, peak_idx in enumerate(peaks):
            # Check if this peak is close to any R-peak
            if len(r_peak_indices) > 0:
                distances = np.abs(r_peak_indices - peak_idx)
                min_distance = np.min(distances)
                if min_distance <= tolerance:
                    peak_labels[i] = 1
        
        all_features.append(features)
        all_labels.append(peak_labels)
    
    # Concatenate all features and labels
    X = np.vstack(all_features) if len(all_features) > 0 else np.array([]).reshape(0, 2)
    y = np.concatenate(all_labels) if len(all_labels) > 0 else np.array([])
    
    return X, y


def get_classifier(algorithm='random_forest', y_train=None):
    """
    Get a classifier instance based on algorithm name.
    
    Args:
        algorithm: Name of algorithm ('random_forest', 'svm', 'xgboost', 
                   'gradient_boosting', 'knn', 'logistic_regression')
        y_train: Training labels (needed for XGBoost class weight calculation)
    
    Returns:
        model: Classifier instance
        scaler: StandardScaler instance (or None if not needed)
    """
    scaler = StandardScaler()
    
    if algorithm == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    elif algorithm == 'svm':
        model = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=42
        )
    elif algorithm == 'xgboost':
        if not HAS_XGBOOST:
            raise ValueError("XGBoost not installed. Install with: pip install xgboost")
        scale_pos_weight = 1.0
        if y_train is not None and len(np.unique(y_train)) > 1:
            n_neg = np.sum(y_train == 0)
            n_pos = np.sum(y_train == 1)
            if n_pos > 0:
                scale_pos_weight = n_neg / n_pos
        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1
        )
        scaler = None  # XGBoost doesn't need scaling
    elif algorithm == 'gradient_boosting':
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        scaler = None  # Gradient Boosting doesn't need scaling
    elif algorithm == 'knn':
        model = KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            metric='euclidean'
        )
    elif algorithm == 'logistic_regression':
        model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Choose from: 'random_forest', 'svm', 'xgboost', 'gradient_boosting', 'knn', 'logistic_regression'")
    
    return model, scaler


def train_r_peak_classifier(X_train, y_train, algorithm='random_forest'):
    """
    Train a classifier to classify R-peaks using the specified algorithm.
    
    Args:
        X_train: Feature matrix of shape (n_peaks, 2)
        y_train: Binary labels
        algorithm: ML algorithm to use
    
    Returns:
        model: Trained classifier model
        scaler: Fitted standard scaler (or None if not used)
    """
    model, scaler = get_classifier(algorithm, y_train=y_train)
    
    # Standardize features if scaler is provided
    if scaler is not None:
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = X_train
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    print(f"Trained {algorithm.upper()} model with {len(X_train)} samples")
    print(f"  - R-peaks (positive): {np.sum(y_train == 1)}")
    print(f"  - Non-R-peaks (negative): {np.sum(y_train == 0)}")
    
    return model, scaler


def evaluate_classifier(model, scaler, X_test, y_test):
    """
    Evaluate the R-peak classifier.
    
    Args:
        model: Trained model
        scaler: Fitted scaler (can be None if scaling not used)
        X_test: Test features
        y_test: Test labels
    
    Returns:
        Dictionary with evaluation metrics
    """
    if scaler is not None:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = X_test
    
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average='binary', zero_division=0
    )
    
    accuracy = np.mean(y_pred == y_test)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'n_samples': len(y_test),
        'n_r_peaks': np.sum(y_test == 1),
        'n_predicted_r_peaks': np.sum(y_pred == 1)
    }
    
    print("\n" + "="*50)
    print("Classification Performance:")
    print("="*50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"\nSamples: {len(y_test)}")
    print(f"True R-peaks: {np.sum(y_test == 1)}")
    print(f"Predicted R-peaks: {np.sum(y_pred == 1)}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-R-peak', 'R-peak']))
    
    return metrics


def predict_r_peaks_for_segments(model, scaler, signals, distance=10):
    """
    Predict R-peaks for signal segments.
    
    Args:
        model: Trained model
        scaler: Fitted scaler (can be None if scaling not used)
        signals: Array of signals
        distance: Minimum distance between peaks
    
    Returns:
        List of arrays, each containing predicted R-peak indices for each signal
    """
    predictions = []
    
    for signal in tqdm(signals, desc="Predicting R-peaks"):
        # Extract peaks
        peaks, peak_heights = extract_all_peaks(signal, distance=distance)
        
        if len(peaks) == 0:
            predictions.append(np.array([]))
            continue
        
        # Extract features
        features = extract_peak_features(signal, peaks, peak_heights)
        
        # Predict
        if scaler is not None:
            features_scaled = scaler.transform(features)
        else:
            features_scaled = features
        
        predictions_binary = model.predict(features_scaled)
        
        # Get indices of predicted R-peaks
        r_peak_indices = peaks[predictions_binary == 1]
        predictions.append(r_peak_indices)
    
    return predictions


def print_feature_importance(model, algorithm):
    """Print feature importance based on the model type."""
    feature_names = ['Normalized Position', 'Normalized Height']
    
    if algorithm == 'random_forest':
        importances = model.feature_importances_
        print("\n" + "="*60)
        print("Feature Importances (Random Forest):")
        print("="*60)
        for name, importance in zip(feature_names, importances):
            print(f"{name}: {importance:.4f}")
    elif algorithm == 'gradient_boosting':
        importances = model.feature_importances_
        print("\n" + "="*60)
        print("Feature Importances (Gradient Boosting):")
        print("="*60)
        for name, importance in zip(feature_names, importances):
            print(f"{name}: {importance:.4f}")
    elif algorithm == 'xgboost':
        importances = model.feature_importances_
        print("\n" + "="*60)
        print("Feature Importances (XGBoost):")
        print("="*60)
        for name, importance in zip(feature_names, importances):
            print(f"{name}: {importance:.4f}")
    elif algorithm == 'logistic_regression':
        print("\n" + "="*60)
        print("Model Coefficients (Logistic Regression):")
        print("="*60)
        for name, coef in zip(feature_names, model.coef_[0]):
            print(f"{name}: {coef:.4f}")
        print(f"Intercept: {model.intercept_[0]:.4f}")
    else:
        print("\n" + "="*60)
        print("Feature Importance:")
        print("="*60)
        print(f"Algorithm '{algorithm}' does not provide feature importance in this format.")


def main(algorithm='random_forest'):
    """
    Main function to run the R-peak classification pipeline.
    
    Args:
        algorithm: ML algorithm to use. Options:
                  - 'random_forest' (default)
                  - 'svm'
                  - 'xgboost'
                  - 'gradient_boosting'
                  - 'knn'
                  - 'logistic_regression'
    """
    print("="*60)
    print(f"R-peak Classification using {algorithm.upper().replace('_', ' ')}")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    X_train = np.load(x_train_path)
    y_train = np.load(y_train_path)
    X_test = np.load(x_test_path)
    y_test = np.load(y_test_path)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    # Extract peaks and create training data
    print("\n" + "="*60)
    print("Extracting peaks from training data...")
    print("="*60)
    X_train_features, y_train_labels = create_training_data_from_segments(X_train, y_train, distance=10)
    
    print(f"\nExtracted {len(X_train_features)} peaks from training data")
    print(f"  - R-peaks: {np.sum(y_train_labels == 1)}")
    print(f"  - Non-R-peaks: {np.sum(y_train_labels == 0)}")
    
    # Extract peaks from test data
    print("\n" + "="*60)
    print("Extracting peaks from test data...")
    print("="*60)
    X_test_features, y_test_labels = create_training_data_from_segments(X_test, y_test, distance=10)
    
    print(f"\nExtracted {len(X_test_features)} peaks from test data")
    print(f"  - R-peaks: {np.sum(y_test_labels == 1)}")
    print(f"  - Non-R-peaks: {np.sum(y_test_labels == 0)}")
    
    if len(X_train_features) == 0:
        print("ERROR: No peaks extracted from training data!")
        return
    
    if len(X_test_features) == 0:
        print("ERROR: No peaks extracted from test data!")
        return
    
    # Train classifier
    print("\n" + "="*60)
    print(f"Training R-peak classifier using {algorithm}...")
    print("="*60)
    try:
        model, scaler = train_r_peak_classifier(X_train_features, y_train_labels, algorithm=algorithm)
    except ValueError as e:
        print(f"ERROR: {e}")
        return
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("Evaluating on test set...")
    print("="*60)
    metrics = evaluate_classifier(model, scaler, X_test_features, y_test_labels)
    
    # Show feature importance (if available)
    print_feature_importance(model, algorithm)
    
    print("\n" + "="*60)
    print("Pipeline completed successfully!")
    print("="*60)
    
    return model, scaler, metrics


if __name__ == "__main__":
    # Change this to use different algorithms:
    # Options: 'random_forest', 'svm', 'xgboost', 'gradient_boosting', 'knn', 'logistic_regression'
    algorithm = 'xgboost'
    
    model, scaler, metrics = main(algorithm=algorithm)