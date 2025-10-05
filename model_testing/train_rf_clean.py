#!/usr/bin/env python3
"""
Clean Random Forest trainer for exoplanet datasets.
Automatically filters columns based on NASA Exoplanet Archive API documentation.
"""

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Define columns to exclude for each dataset based on NASA API documentation
KEPLER_EXCLUDE_PATTERNS = [
    # Identification Columns
    'kepid', 'kepoi_name', 'kepler_name',
    # Exoplanet Archive Information  
    'koi_disposition', 'koi_vet_stat', 'koi_vet_date',
    # Project Disposition Columns
    # 'koi_pdisposition', 'koi_score', 'koi_fpflag_', 'koi_disp_prov', 'koi_comment',
    # Other metadata
    'rowid', 'koi_tce_delivname', 'koi_datalink_', 'koi_quarters'
]

K2_EXCLUDE_PATTERNS = [
    # Identification Columns
    'epic_candname', 'epic_hostname', 'k2_name', 'epic_name',
    # Archive Information
    'disposition', 'disp_prov', 'vet_stat', 'vet_date', 'soltype',
    # Disposition Columns
    'score', 'fpflag_', 'comment', 'sy_pnum', 'hostname', 'pl_name',
    # Other metadata
    'rowid', 'delivname', 'datalink_', 'quarters'
]

TESS_EXCLUDE_PATTERNS = [
    # Identification Columns
    'tid', 'toi', 'tic_', 'toipfx', 'ctoi',
    # Archive Information
    'tfopwg_disp', 'disposition', 'disp_prov',
    # Disposition Columns
    # 'score', 'fpflag_', 'comment', 'vet_stat', 'vet_date',
    # Other metadata  
    'rowid', 'delivname', 'datalink_'
]

def detect_dataset_type(df):
    """Auto-detect dataset type based on column names."""
    columns = set(df.columns)
    
    if 'koi_disposition' in columns:
        return 'kepler', 'koi_disposition'
    elif 'disposition' in columns and any('epic' in col for col in columns):
        return 'k2', 'disposition'
    elif 'tfopwg_disp' in columns:
        return 'tess', 'tfopwg_disp'
    else:
        raise ValueError(f"Cannot detect dataset type. Available columns: {list(columns)[:10]}...")

def filter_columns(df, dataset_type, target_col):
    """Remove columns that should not be used as features."""
    if dataset_type == 'kepler':
        exclude_patterns = KEPLER_EXCLUDE_PATTERNS
    elif dataset_type == 'k2':
        exclude_patterns = K2_EXCLUDE_PATTERNS  
    elif dataset_type == 'tess':
        exclude_patterns = TESS_EXCLUDE_PATTERNS
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Find columns to exclude
    columns_to_drop = []
    for col in df.columns:
        # Never drop the target column
        if col == target_col:
            continue
        for pattern in exclude_patterns:
            if pattern.lower() in col.lower():
                columns_to_drop.append(col)
                break
    
    # Remove duplicates
    columns_to_drop = list(set(columns_to_drop))
    
    print(f"\nDataset type: {dataset_type.upper()}")
    print(f"Target column: {target_col}")
    print(f"Columns to drop ({len(columns_to_drop)}): {sorted(columns_to_drop)}")
    
    # Drop the columns
    df_filtered = df.drop(columns=columns_to_drop, errors='ignore')
    
    remaining_cols = [col for col in df_filtered.columns]
    print(f"Remaining columns ({len(remaining_cols)}): {sorted(remaining_cols)}")
    
    return df_filtered, columns_to_drop

def preprocess_features(X):
    """Preprocess feature matrix: handle missing values and encode categorical variables."""
    print(f"\nPreprocessing {X.shape[1]} features...")
    
    # Identify categorical columns
    categorical_cols = []
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            categorical_cols.append(col)
    
    if categorical_cols:
        print(f"Encoding categorical columns: {categorical_cols}")
        for col in categorical_cols:
            # Fill missing values with mode for categorical
            X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'unknown')
            # Label encode
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    
    # Handle numeric columns
    numeric_cols = [col for col in X.columns if col not in categorical_cols]
    if numeric_cols:
        print(f"Processing {len(numeric_cols)} numeric columns...")
        for col in numeric_cols:
            # Fill missing values with median for numeric
            X[col] = pd.to_numeric(X[col], errors='coerce')
            X[col] = X[col].fillna(X[col].median())
    
    # Final check - ensure all columns are numeric
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            print(f"Warning: {col} is still not numeric, converting...")
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    print(f"Preprocessing complete. Final shape: {X.shape}")
    return X

def load_and_prepare_data(data_path):
    """Load CSV file, handling comment lines."""
    print(f"Loading data from: {data_path}")
    
    # Skip comment lines that start with #
    with open(data_path, 'r') as f:
        lines = f.readlines()
    
    # Find first non-comment line for header
    header_idx = 0
    for i, line in enumerate(lines):
        if not line.startswith('#') and line.strip():
            header_idx = i
            break
    
    # Load the CSV
    df = pd.read_csv(data_path, skiprows=header_idx)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Train Random Forest on exoplanet datasets")
    parser.add_argument('--data', required=True, help='Dataset filename (e.g., kepler_raw.csv)')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test split ratio (default: 0.2)')
    parser.add_argument('--n-estimators', type=int, default=100, help='Number of trees (default: 100)')
    parser.add_argument('--max-depth', type=int, default=None, help='Maximum tree depth (default: None)')
    parser.add_argument('--min-samples-split', type=int, default=2, help='Min samples to split (default: 2)')
    parser.add_argument('--min-samples-leaf', type=int, default=1, help='Min samples in leaf (default: 1)')
    parser.add_argument('--random-state', type=int, default=42, help='Random state (default: 42)')
    parser.add_argument('--scale', action='store_true', help='Apply StandardScaler to features')
    
    args = parser.parse_args()
    
    # Construct full path
    data_path = os.path.join('data', args.data) if not os.path.isabs(args.data) else args.data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Load data
    df = load_and_prepare_data(data_path)
    
    # Detect dataset type and target column
    dataset_type, target_col = detect_dataset_type(df)
    
    # Filter out inappropriate columns
    df_clean, dropped_cols = filter_columns(df, dataset_type, target_col)
    
    # Separate features and target
    if target_col not in df_clean.columns:
        raise ValueError(f"Target column '{target_col}' not found after filtering!")
    
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    
    # Clean target labels (remove nulls, standardize case)
    y = y.dropna()
    X = X.loc[y.index]  # Keep only rows with valid targets
    
    # Standardize target labels
    y = y.str.upper().str.strip()
    
    print(f"\nTarget distribution:")
    print(y.value_counts())
    
    # Map TESS labels to standard labels
    if dataset_type == 'tess':
        # TESS mapping: PC=Planet Candidate, CP=Confirmed Planet, FP=False Positive
        # KP=Known Planet, APC=Ambiguous Planet Candidate, FA=False Alarm
        tess_mapping = {
            'PC': 'CANDIDATE',
            'APC': 'CANDIDATE', 
            'CP': 'CONFIRMED',
            'KP': 'CONFIRMED',
            'FP': 'FALSE POSITIVE',
            'FA': 'FALSE POSITIVE'
        }
        y = y.map(tess_mapping).fillna('UNKNOWN')
    
    # Filter to keep only common classes
    valid_classes = ['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE']
    mask = y.isin(valid_classes)
    X = X[mask]
    y = y[mask]
    
    print(f"\nAfter filtering to standard classes:")
    print(y.value_counts())
    print(f"Final dataset shape: {X.shape}")
    
    # Preprocess features
    X = preprocess_features(X)
    
    # Apply scaling if requested
    if args.scale:
        print("Applying StandardScaler...")
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=args.test_size, 
        random_state=args.random_state,
        stratify=y
    )
    
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train Random Forest
    print(f"\nTraining Random Forest with parameters:")
    print(f"  n_estimators: {args.n_estimators}")
    print(f"  max_depth: {args.max_depth}")
    print(f"  min_samples_split: {args.min_samples_split}")
    print(f"  min_samples_leaf: {args.min_samples_leaf}")
    print(f"  random_state: {args.random_state}")
    
    rf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
        n_jobs=-1
    )
    
    rf.fit(X_train, y_train)
    
    # Predictions
    y_pred = rf.predict(X_test)
    y_pred_proba = rf.predict_proba(X_test)
    
    # Results
    print(f"\n{'='*60}")
    print(f"RANDOM FOREST RESULTS - {dataset_type.upper()} Dataset")
    print(f"{'='*60}")
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred, labels=['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE'])
    print(f"                 Predicted")
    print(f"Actual      CAND  CONF  FP")
    for i, actual_class in enumerate(['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE']):
        row = f"{actual_class[:4]:8s} "
        row += " ".join(f"{cm[i][j]:5d}" for j in range(len(cm[i])))
        print(row)
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Most Important Features:")
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
        print(f"{i+1:2d}. {row['feature']:25s} {row['importance']:.4f}")
    
    # Prediction confidence statistics
    max_proba = np.max(y_pred_proba, axis=1)
    print(f"\nPrediction Confidence Statistics:")
    print(f"  Mean confidence: {np.mean(max_proba):.3f}")
    print(f"  Median confidence: {np.median(max_proba):.3f}")
    print(f"  Min confidence: {np.min(max_proba):.3f}")
    print(f"  Max confidence: {np.max(max_proba):.3f}")
    
    # Low confidence predictions
    low_conf_threshold = 0.5
    low_conf_mask = max_proba < low_conf_threshold
    if np.sum(low_conf_mask) > 0:
        print(f"\nLow confidence predictions (< {low_conf_threshold}):")
        print(f"  Count: {np.sum(low_conf_mask)} out of {len(y_test)} ({100*np.sum(low_conf_mask)/len(y_test):.1f}%)")
    
    print(f"\nSummary:")
    print(f"  Dataset: {dataset_type.upper()}")
    print(f"  Features used: {X.shape[1]}")
    print(f"  Features dropped: {len(dropped_cols)}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Test accuracy: {rf.score(X_test, y_test):.3f}")

if __name__ == "__main__":
    main()