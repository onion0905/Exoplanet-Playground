"""
Column filtering utilities for exoplanet datasets.
Implements NASA Exoplanet Archive API documentation-based column exclusion.
"""

from typing import Dict, List, Tuple, Optional
import pandas as pd
import logging

# Define columns to exclude for each dataset based on NASA API documentation
KEPLER_EXCLUDE_PATTERNS = [
    # Identification Columns
    # Exoplanet Archive Information  
    # Project Disposition Columns (commented out in train_rf_clean.py for better performance)
    # 'koi_pdisposition', 'koi_score', 'koi_fpflag_', 'koi_disp_prov', 'koi_comment',
    # Other metadata
]

K2_EXCLUDE_PATTERNS = [
    # Identification Columns
    # Archive Information (excluding target - disposition is never excluded as it's the target)
    # Disposition Columns (commented out in train_rf_clean.py for better performance)
    # 'score', 'fpflag_', 'comment', 
    # Other metadata
]

TESS_EXCLUDE_PATTERNS = [
    # Identification Columns
    # Archive Information (excluding target - tfopwg_disp is never excluded as it's the target)
    # Disposition Columns (commented out in train_rf_clean.py for better performance)
    # 'score', 'fpflag_', 'comment', 'vet_stat', 'vet_date',
    # Other metadata  
]


class ColumnFilter:
    """Filters columns based on NASA Exoplanet Archive guidelines."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.dataset_type = None
        self.target_column = None
        self.excluded_columns = []
        
    def detect_dataset_type(self, df: pd.DataFrame) -> Tuple[str, str]:
        """Auto-detect dataset type and target column based on column names."""
        columns = set(df.columns)
        
        if 'koi_disposition' in columns:
            return 'kepler', 'koi_disposition'
        elif 'tfopwg_disp' in columns:
            return 'tess', 'tfopwg_disp'
        elif 'disposition' in columns:
            # Check if it's likely K2 data based on other characteristics
            # K2 data has certain column patterns or we can default to K2 for general disposition data
            return 'k2', 'disposition'
        else:
            available_cols = list(columns)[:10]
            raise ValueError(f"Cannot detect dataset type. Available columns: {available_cols}...")
    
    def get_exclude_patterns(self, dataset_type: str) -> List[str]:
        """Get exclusion patterns for a dataset type."""
        if dataset_type == 'kepler':
            return KEPLER_EXCLUDE_PATTERNS
        elif dataset_type == 'k2':
            return K2_EXCLUDE_PATTERNS  
        elif dataset_type == 'tess':
            return TESS_EXCLUDE_PATTERNS
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    def filter_columns(self, df: pd.DataFrame, 
                      dataset_type: Optional[str] = None,
                      target_col: Optional[str] = None,
                      additional_exclude: Optional[List[str]] = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Remove columns that should not be used as features.
        
        Args:
            df: Input DataFrame
            dataset_type: Dataset type ('kepler', 'k2', 'tess'). Auto-detected if None.
            target_col: Target column name. Auto-detected if None.
            additional_exclude: Additional columns to exclude beyond standard patterns.
            
        Returns:
            Tuple of (filtered_df, list_of_excluded_columns)
        """
        # Auto-detect dataset type if not provided
        if dataset_type is None or target_col is None:
            detected_type, detected_target = self.detect_dataset_type(df)
            dataset_type = dataset_type or detected_type
            target_col = target_col or detected_target
        
        self.dataset_type = dataset_type
        self.target_column = target_col
        
        # Get exclusion patterns
        exclude_patterns = self.get_exclude_patterns(dataset_type)
        
        # Add additional exclusions if provided
        if additional_exclude:
            exclude_patterns.extend(additional_exclude)
        
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
        self.excluded_columns = columns_to_drop
        
        self.logger.info(f"Dataset type: {dataset_type.upper()}")
        self.logger.info(f"Target column: {target_col}")
        self.logger.info(f"Columns to drop ({len(columns_to_drop)}): {sorted(columns_to_drop)}")
        
        # Drop the columns
        df_filtered = df.drop(columns=columns_to_drop, errors='ignore')
        
        remaining_cols = [col for col in df_filtered.columns]
        self.logger.info(f"Remaining columns ({len(remaining_cols)}): {len(remaining_cols)} features + target")
        
        return df_filtered, columns_to_drop
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns (excluding target)."""
        if self.target_column and self.target_column in df.columns:
            return [col for col in df.columns if col != self.target_column]
        return list(df.columns)
    
    def map_target_labels(self, target_series: pd.Series, dataset_type: str) -> pd.Series:
        """Map target labels to standard format for the dataset type."""
        target_mapped = target_series.copy()
        
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
            target_mapped = target_mapped.map(tess_mapping).fillna(target_mapped)
            self.logger.info("Applied TESS label mapping")
        
        # Standardize case and strip whitespace
        target_mapped = target_mapped.str.upper().str.strip()
        
        return target_mapped