from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import logging
from .column_filter import ColumnFilter


class DataProcessor:
    """Processes and prepares exoplanet data for ML training."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scalers = {}
        self.imputers = {}
        self.label_encoders = {}
        self.feature_names = None
        self.target_name = None
        self.processing_config = {}
        self.column_filter = ColumnFilter()
        
    def create_target_variable(self, data: pd.DataFrame, 
                             target_column: str,
                             target_mapping: Dict[str, str] = None) -> pd.Series:
        """Create a target variable for classification."""
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        target_series = data[target_column].copy()
        
        # Apply target mapping if provided
        if target_mapping:
            target_series = target_series.map(target_mapping)
            self.logger.info(f"Applied target mapping: {target_mapping}")
        
        # Handle common exoplanet dispositions
        if target_column in ['koi_disposition', 'disposition', 'tfopwg_disp']:
            # Standard classification for exoplanet validation
            disposition_mapping = {
                'CONFIRMED': 'planet',
                'CANDIDATE': 'candidate', 
                'FALSE POSITIVE': 'false_positive',
                'NOT DISPOSITIONED': 'unknown'
            }
            
            # Apply case-insensitive mapping
            target_series_upper = target_series.str.upper()
            for key, value in disposition_mapping.items():
                target_series.loc[target_series_upper == key] = value
        
        # Apply dataset-specific label mapping if needed
        if hasattr(self.column_filter, 'dataset_type') and self.column_filter.dataset_type:
            target_series = self.column_filter.map_target_labels(target_series, self.column_filter.dataset_type)
        
        self.target_name = target_column
        return target_series
    
    def select_features(self, data: pd.DataFrame, 
                       selected_columns: List[str] = None,
                       exclude_columns: List[str] = None,
                       max_missing_ratio: float = 0.5) -> pd.DataFrame:
        """Select features for training."""
        df = data.copy()
        
        # Start with all numeric columns if no selection provided
        if selected_columns is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            selected_columns = numeric_columns
        
        # Remove excluded columns
        if exclude_columns:
            selected_columns = [col for col in selected_columns if col not in exclude_columns]
        
        # Filter out columns with too many missing values
        valid_columns = []
        for col in selected_columns:
            if col in df.columns:
                missing_ratio = df[col].isnull().sum() / len(df)
                if missing_ratio <= max_missing_ratio:
                    valid_columns.append(col)
                else:
                    self.logger.warning(f"Excluding {col}: {missing_ratio:.2%} missing values")
        
        self.feature_names = valid_columns
        return df[valid_columns]
    
    def apply_nasa_api_filtering(self, data: pd.DataFrame, 
                               dataset_type: Optional[str] = None,
                               target_col: Optional[str] = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Apply NASA Exoplanet Archive API-based column filtering.
        
        Args:
            data: Input DataFrame
            dataset_type: Dataset type ('kepler', 'k2', 'tess'). Auto-detected if None.
            target_col: Target column name. Auto-detected if None.
            
        Returns:
            Tuple of (filtered_df, list_of_excluded_columns)
        """
        filtered_df, excluded_cols = self.column_filter.filter_columns(
            data, dataset_type=dataset_type, target_col=target_col
        )
        
        # Update target name if auto-detected
        if self.column_filter.target_column:
            self.target_name = self.column_filter.target_column
        
        self.logger.info(f"NASA API filtering complete. Excluded {len(excluded_cols)} columns.")
        
        return filtered_df, excluded_cols
    
    def handle_missing_values(self, X: pd.DataFrame, 
                            strategy: str = 'median',
                            fit: bool = True) -> pd.DataFrame:
        """Handle missing values in features."""
        if fit:
            self.imputers = {}
        
        X_imputed = X.copy()
        
        for column in X.columns:
            if X[column].isnull().any():
                if fit:
                    # Determine strategy for this column
                    if X[column].dtype in ['object', 'category']:
                        imputer_strategy = 'most_frequent'
                    else:
                        imputer_strategy = strategy
                    
                    imputer = SimpleImputer(strategy=imputer_strategy)
                    self.imputers[column] = imputer
                    X_imputed[column] = imputer.fit_transform(X[[column]]).flatten()
                else:
                    if column in self.imputers:
                        X_imputed[column] = self.imputers[column].transform(X[[column]]).flatten()
                    else:
                        self.logger.warning(f"No imputer found for column {column}")
        
        return X_imputed
    
    def scale_features(self, X: pd.DataFrame, 
                      scaler_type: str = 'standard',
                      fit: bool = True) -> pd.DataFrame:
        """Scale numerical features."""
        if fit:
            self.scalers = {}
        
        X_scaled = X.copy()
        
        # Select numeric columns for scaling
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) > 0:
            if fit:
                if scaler_type == 'standard':
                    scaler = StandardScaler()
                elif scaler_type == 'robust':
                    scaler = RobustScaler()
                else:
                    raise ValueError(f"Unknown scaler type: {scaler_type}")
                
                self.scalers['feature_scaler'] = scaler
                X_scaled[numeric_columns] = scaler.fit_transform(X[numeric_columns])
            else:
                if 'feature_scaler' in self.scalers:
                    X_scaled[numeric_columns] = self.scalers['feature_scaler'].transform(X[numeric_columns])
                else:
                    self.logger.warning("No fitted scaler found for features")
        
        return X_scaled
    
    def encode_categorical_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features."""
        if fit:
            self.label_encoders = {}
        
        X_encoded = X.copy()
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns
        
        for column in categorical_columns:
            if fit:
                encoder = LabelEncoder()
                self.label_encoders[column] = encoder
                X_encoded[column] = encoder.fit_transform(X[column].astype(str))
            else:
                if column in self.label_encoders:
                    # Handle unseen categories by encoding them as -1
                    try:
                        X_encoded[column] = self.label_encoders[column].transform(X[column].astype(str))
                    except ValueError:
                        # Handle unseen labels
                        known_labels = set(self.label_encoders[column].classes_)
                        X_temp = X[column].astype(str)
                        X_temp[~X_temp.isin(known_labels)] = 'UNKNOWN'
                        
                        # Add UNKNOWN to encoder if not present
                        if 'UNKNOWN' not in self.label_encoders[column].classes_:
                            encoder = self.label_encoders[column]
                            encoder.classes_ = np.append(encoder.classes_, 'UNKNOWN')
                        
                        X_encoded[column] = self.label_encoders[column].transform(X_temp)
                else:
                    self.logger.warning(f"No encoder found for column {column}")
        
        return X_encoded
    
    def remove_outliers(self, X: pd.DataFrame, y: pd.Series = None,
                       method: str = 'iqr', threshold: float = 3.0) -> Tuple[pd.DataFrame, pd.Series]:
        """Remove outliers from the dataset."""
        if method == 'iqr':
            # Interquartile Range method
            numeric_columns = X.select_dtypes(include=[np.number]).columns
            outlier_mask = pd.Series([False] * len(X), index=X.index)
            
            for column in numeric_columns:
                Q1 = X[column].quantile(0.25)
                Q3 = X[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                column_outliers = (X[column] < lower_bound) | (X[column] > upper_bound)
                outlier_mask = outlier_mask | column_outliers
            
        elif method == 'zscore':
            # Z-score method
            numeric_columns = X.select_dtypes(include=[np.number]).columns
            outlier_mask = pd.Series([False] * len(X), index=X.index)
            
            for column in numeric_columns:
                z_scores = np.abs((X[column] - X[column].mean()) / X[column].std())
                column_outliers = z_scores > threshold
                outlier_mask = outlier_mask | column_outliers
        
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        # Remove outliers
        X_cleaned = X[~outlier_mask]
        y_cleaned = y[~outlier_mask] if y is not None else None
        
        outliers_removed = outlier_mask.sum()
        self.logger.info(f"Removed {outliers_removed} outliers ({outliers_removed/len(X):.2%})")
        
        return X_cleaned, y_cleaned
    
    def prepare_data(self, data: pd.DataFrame, 
                    target_column: str,
                    feature_columns: List[str] = None,
                    test_size: float = 0.2,
                    val_size: float = 0.1,
                    random_state: int = 42,
                    preprocessing_config: Dict[str, Any] = None,
                    dataset_type: str = None) -> Dict[str, Any]:
        """Complete data preparation pipeline."""
        
        # Default preprocessing configuration
        default_config = {
            'handle_missing': True,
            'missing_strategy': 'median',
            'scale_features': False,  # Don't scale by default - matches train_rf_clean.py
            'scaler_type': 'standard',
            'encode_categorical': True,
            'remove_outliers': False,
            'outlier_method': 'iqr',
            'max_missing_ratio': 0.999,  # Conservative - only exclude 100% missing columns
            'impute_with_nasa_means': False
        }
        
        if preprocessing_config:
            default_config.update(preprocessing_config)
        
        self.processing_config = default_config
        
        # Create target variable
        y = self.create_target_variable(data, target_column)
        
        # Select features
        X = self.select_features(
            data, 
            selected_columns=feature_columns,
            max_missing_ratio=default_config['max_missing_ratio']
        )
        
        # Impute missing values with NASA means if requested
        if default_config.get('impute_with_nasa_means', False) and dataset_type is not None:
            from .nasa_impute import impute_with_nasa_means
            X = impute_with_nasa_means(X, dataset_type)
            self.logger.info(f"Imputed missing values with NASA {dataset_type} means.")

        # Remove rows where target is missing
        valid_mask = ~y.isnull()
        X = X[valid_mask]
        y = y[valid_mask]
        
        # Split data before preprocessing to prevent data leakage
        if val_size > 0:
            # Three-way split: train, validation, test
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Calculate validation size relative to remaining data
            val_size_adjusted = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size_adjusted, 
                random_state=random_state, stratify=y_temp
            )
        else:
            # Two-way split: train, test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            X_val, y_val = None, None
        
        # Apply preprocessing (fit on training data only)
        if default_config['handle_missing']:
            X_train = self.handle_missing_values(X_train, default_config['missing_strategy'], fit=True)
            X_test = self.handle_missing_values(X_test, fit=False)
            if X_val is not None:
                X_val = self.handle_missing_values(X_val, fit=False)
        
        if default_config['encode_categorical']:
            X_train = self.encode_categorical_features(X_train, fit=True)
            X_test = self.encode_categorical_features(X_test, fit=False)
            if X_val is not None:
                X_val = self.encode_categorical_features(X_val, fit=False)
        
        if default_config['remove_outliers']:
            X_train, y_train = self.remove_outliers(
                X_train, y_train, method=default_config['outlier_method']
            )
        
        if default_config['scale_features']:
            X_train = self.scale_features(X_train, default_config['scaler_type'], fit=True)
            X_test = self.scale_features(X_test, fit=False)
            if X_val is not None:
                X_val = self.scale_features(X_val, fit=False)
        
        # Prepare result dictionary
        result = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'preprocessing_config': self.processing_config
        }
        
        if X_val is not None:
            result['X_val'] = X_val
            result['y_val'] = y_val
        
        self.logger.info(f"Data preparation complete: Train={len(X_train)}, Test={len(X_test)}" + 
                        (f", Val={len(X_val)}" if X_val is not None else ""))
        
        return result