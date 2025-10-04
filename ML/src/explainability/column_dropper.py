from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import logging


class ColumnDropper:
    """Utility for dropping/selecting columns and analyzing the impact."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.dropped_columns = []
        self.original_columns = []
        
    def select_columns_by_importance(self, data: pd.DataFrame, 
                                   importance_scores: Dict[str, float],
                                   threshold: float = 0.01,
                                   top_n: Optional[int] = None) -> pd.DataFrame:
        """Select columns based on importance scores."""
        self.original_columns = list(data.columns)
        
        if top_n is not None:
            # Select top N most important features
            sorted_features = sorted(importance_scores.items(), 
                                   key=lambda x: x[1], reverse=True)
            selected_features = [feat for feat, _ in sorted_features[:top_n]]
        else:
            # Select features above threshold
            selected_features = [feat for feat, score in importance_scores.items() 
                               if score >= threshold]
        
        # Filter to only include columns that exist in data
        valid_features = [feat for feat in selected_features if feat in data.columns]
        self.dropped_columns = [col for col in data.columns if col not in valid_features]
        
        self.logger.info(f"Selected {len(valid_features)} features, dropped {len(self.dropped_columns)}")
        
        return data[valid_features]
    
    def drop_columns_by_list(self, data: pd.DataFrame, 
                           columns_to_drop: List[str]) -> pd.DataFrame:
        """Drop specific columns from the dataset."""
        self.original_columns = list(data.columns)
        
        # Filter to only drop columns that exist in data
        valid_drops = [col for col in columns_to_drop if col in data.columns]
        invalid_drops = [col for col in columns_to_drop if col not in data.columns]
        
        if invalid_drops:
            self.logger.warning(f"Columns not found in data: {invalid_drops}")
        
        self.dropped_columns = valid_drops
        remaining_columns = [col for col in data.columns if col not in valid_drops]
        
        self.logger.info(f"Dropped {len(valid_drops)} columns, {len(remaining_columns)} remaining")
        
        return data[remaining_columns]
    
    def drop_columns_by_missing_threshold(self, data: pd.DataFrame, 
                                        max_missing_ratio: float = 0.5) -> pd.DataFrame:
        """Drop columns with too many missing values."""
        self.original_columns = list(data.columns)
        
        columns_to_drop = []
        for column in data.columns:
            missing_ratio = data[column].isnull().sum() / len(data)
            if missing_ratio > max_missing_ratio:
                columns_to_drop.append(column)
        
        self.dropped_columns = columns_to_drop
        remaining_columns = [col for col in data.columns if col not in columns_to_drop]
        
        self.logger.info(f"Dropped {len(columns_to_drop)} columns with >{max_missing_ratio:.0%} missing values")
        
        return data[remaining_columns]
    
    def drop_low_variance_features(self, data: pd.DataFrame, 
                                 variance_threshold: float = 0.01) -> pd.DataFrame:
        """Drop features with low variance."""
        self.original_columns = list(data.columns)
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        columns_to_drop = []
        
        for column in numeric_columns:
            if data[column].var() < variance_threshold:
                columns_to_drop.append(column)
        
        self.dropped_columns = columns_to_drop
        remaining_columns = [col for col in data.columns if col not in columns_to_drop]
        
        self.logger.info(f"Dropped {len(columns_to_drop)} low-variance features")
        
        return data[remaining_columns]
    
    def drop_highly_correlated_features(self, data: pd.DataFrame, 
                                      correlation_threshold: float = 0.95) -> pd.DataFrame:
        """Drop highly correlated features."""
        self.original_columns = list(data.columns)
        
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            self.logger.warning("No numeric columns found for correlation analysis")
            return data
        
        # Calculate correlation matrix
        corr_matrix = numeric_data.corr().abs()
        
        # Find highly correlated pairs
        columns_to_drop = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > correlation_threshold:
                    # Drop the column with lower variance
                    col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    if numeric_data[col1].var() < numeric_data[col2].var():
                        columns_to_drop.add(col1)
                    else:
                        columns_to_drop.add(col2)
        
        self.dropped_columns = list(columns_to_drop)
        remaining_columns = [col for col in data.columns if col not in columns_to_drop]
        
        self.logger.info(f"Dropped {len(columns_to_drop)} highly correlated features")
        
        return data[remaining_columns]
    
    def interactive_column_selection(self, data: pd.DataFrame, 
                                   importance_scores: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Prepare data for interactive column selection in frontend.
        Returns column information for UI selection.
        """
        columns_info = []
        
        for column in data.columns:
            col_info = {
                'name': column,
                'dtype': str(data[column].dtype),
                'missing_count': int(data[column].isnull().sum()),
                'missing_ratio': float(data[column].isnull().sum() / len(data)),
                'unique_values': int(data[column].nunique()),
                'is_numeric': pd.api.types.is_numeric_dtype(data[column])
            }
            
            # Add importance score if available
            if importance_scores and column in importance_scores:
                col_info['importance_score'] = importance_scores[column]
            
            # Add basic statistics for numeric columns
            if col_info['is_numeric']:
                col_info.update({
                    'mean': float(data[column].mean()) if not data[column].isnull().all() else None,
                    'std': float(data[column].std()) if not data[column].isnull().all() else None,
                    'min': float(data[column].min()) if not data[column].isnull().all() else None,
                    'max': float(data[column].max()) if not data[column].isnull().all() else None
                })
            
            columns_info.append(col_info)
        
        # Sort by importance score if available
        if importance_scores:
            columns_info.sort(key=lambda x: x.get('importance_score', 0), reverse=True)
        
        return {
            'columns': columns_info,
            'total_columns': len(columns_info),
            'numeric_columns': len([c for c in columns_info if c['is_numeric']]),
            'categorical_columns': len([c for c in columns_info if not c['is_numeric']]),
            'data_shape': data.shape
        }
    
    def apply_column_selection(self, data: pd.DataFrame, 
                             selected_columns: List[str]) -> pd.DataFrame:
        """Apply user-selected column filtering."""
        self.original_columns = list(data.columns)
        
        # Filter to only include valid columns
        valid_columns = [col for col in selected_columns if col in data.columns]
        invalid_columns = [col for col in selected_columns if col not in data.columns]
        
        if invalid_columns:
            self.logger.warning(f"Selected columns not found in data: {invalid_columns}")
        
        self.dropped_columns = [col for col in data.columns if col not in valid_columns]
        
        self.logger.info(f"Selected {len(valid_columns)} columns out of {len(data.columns)}")
        
        return data[valid_columns]
    
    def get_dropped_columns(self) -> List[str]:
        """Get list of columns that were dropped in the last operation."""
        return self.dropped_columns.copy()
    
    def get_original_columns(self) -> List[str]:
        """Get list of original columns before any dropping."""
        return self.original_columns.copy()
    
    def get_column_drop_summary(self) -> Dict[str, Any]:
        """Get summary of column dropping operation."""
        return {
            'original_columns': self.original_columns,
            'dropped_columns': self.dropped_columns,
            'remaining_columns': [col for col in self.original_columns 
                                if col not in self.dropped_columns],
            'n_original': len(self.original_columns),
            'n_dropped': len(self.dropped_columns),
            'n_remaining': len(self.original_columns) - len(self.dropped_columns)
        }