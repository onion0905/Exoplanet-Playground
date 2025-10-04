from typing import Dict, List, Optional, Tuple, Set
import pandas as pd
import numpy as np
import re
import logging


class DataValidator:
    """Validates and filters exoplanet datasets for ML training."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Define patterns for columns to exclude
        self.exclude_patterns = {
            'names': ['name', 'hostname', 'id', 'toi', 'kepid', 'tid'],
            'references': ['ref', 'delivname', 'facility', 'refname'],
            'dates': ['date', 'created', 'update', 'year', 'pubdate', 'release'],
            'flags_categorical': ['flag', 'lim', 'prov', 'str'],
            'target_leakage': ['disposition', 'disp', 'score'],  # These reveal the answer
            'metadata': ['default_flag', 'controv_flag', 'ttv_flag', 'snum', 'pnum', 'soltype']
        }
        
        # Define target columns for different datasets
        self.target_columns = {
            'kepler': 'koi_disposition',
            'tess': 'tfopwg_disp', 
            'k2': 'disposition'
        }
        
        # Define recommended feature columns for each dataset
        self.recommended_features = {
            'kepler': [
                # Planet properties
                'koi_period', 'koi_prad', 'koi_teq', 'koi_insol', 'koi_duration', 'koi_depth',
                'koi_impact', 'koi_model_snr',
                # Stellar properties  
                'koi_steff', 'koi_slogg', 'koi_srad', 'koi_kepmag',
                # Position (could be relevant for systematic effects)
                'ra', 'dec',
                # Error columns (uncertainty information)
                'koi_period_err1', 'koi_period_err2', 'koi_prad_err1', 'koi_prad_err2',
                'koi_teq_err1', 'koi_teq_err2', 'koi_insol_err1', 'koi_insol_err2',
                'koi_steff_err1', 'koi_steff_err2', 'koi_slogg_err1', 'koi_slogg_err2',
                'koi_srad_err1', 'koi_srad_err2'
            ],
            'tess': [
                # Planet properties
                'pl_orbper', 'pl_rade', 'pl_eqt', 'pl_insol', 'pl_trandurh', 'pl_trandep',
                'pl_tranmid',
                # Stellar properties
                'st_teff', 'st_logg', 'st_rad', 'st_tmag', 'st_dist',
                # Position
                'ra', 'dec',
                # Stellar motion
                'st_pmra', 'st_pmdec',
                # Error columns
                'pl_orbpererr1', 'pl_orbpererr2', 'pl_radeerr1', 'pl_radeerr2',
                'pl_eqterr1', 'pl_eqterr2', 'st_tefferr1', 'st_tefferr2'
            ],
            'k2': [
                # Planet properties
                'pl_orbper', 'pl_rade', 'pl_radj', 'pl_bmasse', 'pl_bmassj',
                'pl_orbsmax', 'pl_orbeccen', 'pl_eqt', 'pl_insol',
                # Stellar properties  
                'st_teff', 'st_rad', 'st_mass', 'st_met', 'st_logg',
                'sy_vmag', 'sy_kmag', 'sy_gaiamag', 'sy_dist',
                # Position
                'ra', 'dec',
                # Error columns
                'pl_orbpererr1', 'pl_orbpererr2', 'pl_radeerr1', 'pl_radeerr2',
                'st_tefferr1', 'st_tefferr2', 'st_raderr1', 'st_raderr2'
            ]
        }
    
    def identify_dataset_type(self, data: pd.DataFrame) -> str:
        """Identify which NASA dataset this is based on columns."""
        columns = set(data.columns)
        
        if 'koi_disposition' in columns:
            return 'kepler'
        elif 'tfopwg_disp' in columns:
            return 'tess'  
        elif 'disposition' in columns and 'pl_name' in columns:
            return 'k2'
        else:
            return 'unknown'
    
    def filter_irrelevant_columns(self, data: pd.DataFrame, 
                                dataset_type: str = None) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        """
        Filter out irrelevant columns based on patterns and dataset type.
        
        Returns:
            Tuple of (filtered_dataframe, exclusion_report)
        """
        if dataset_type is None:
            dataset_type = self.identify_dataset_type(data)
        
        original_columns = list(data.columns)
        excluded_columns = {'names': [], 'references': [], 'dates': [], 
                          'flags_categorical': [], 'target_leakage': [], 'metadata': []}
        
        # Get target column for this dataset
        target_col = self.target_columns.get(dataset_type)
        
        columns_to_keep = []
        
        for col in original_columns:
            col_lower = col.lower()
            should_exclude = False
            exclusion_reason = None
            
            # Keep target column
            if col == target_col:
                columns_to_keep.append(col)
                continue
            
            # Check each exclusion pattern
            for category, patterns in self.exclude_patterns.items():
                for pattern in patterns:
                    if pattern in col_lower:
                        # Special handling for disposition columns
                        if category == 'target_leakage':
                            # Only exclude if it's not the main target column
                            if col != target_col and ('disp' in col_lower or 'score' in col_lower):
                                should_exclude = True
                                exclusion_reason = category
                                break
                        else:
                            should_exclude = True
                            exclusion_reason = category
                            break
                if should_exclude:
                    break
            
            if should_exclude:
                excluded_columns[exclusion_reason].append(col)
            else:
                columns_to_keep.append(col)
        
        # Additional filtering: remove non-numeric columns that can't be easily processed
        final_columns = []
        for col in columns_to_keep:
            if col == target_col:
                final_columns.append(col)  # Always keep target
                continue
                
            # Check if column is numeric or can be made numeric
            if data[col].dtype in ['int64', 'float64']:
                final_columns.append(col)
            elif data[col].dtype == 'object':
                # Check if it's a limited categorical that can be encoded
                unique_values = data[col].dropna().unique()
                if len(unique_values) <= 10 and len(unique_values) >= 2:
                    # Keep small categorical columns
                    final_columns.append(col)
                else:
                    excluded_columns['flags_categorical'].append(col)
        
        filtered_data = data[final_columns]
        
        self.logger.info(f"Dataset type: {dataset_type}")
        self.logger.info(f"Original columns: {len(original_columns)}")
        self.logger.info(f"Filtered columns: {len(final_columns)}")
        
        return filtered_data, excluded_columns
    
    def get_recommended_features(self, data: pd.DataFrame, 
                               dataset_type: str = None) -> List[str]:
        """Get recommended feature columns for a dataset type."""
        if dataset_type is None:
            dataset_type = self.identify_dataset_type(data)
        
        recommended = self.recommended_features.get(dataset_type, [])
        
        # Filter to only include columns that exist in the data
        available_features = [col for col in recommended if col in data.columns]
        
        self.logger.info(f"Recommended features for {dataset_type}: {len(available_features)}/{len(recommended)}")
        
        return available_features
    
    def validate_target_column(self, data: pd.DataFrame, 
                             target_column: str) -> Dict[str, any]:
        """Validate target column and provide mapping suggestions."""
        if target_column not in data.columns:
            return {'valid': False, 'error': f'Target column {target_column} not found'}
        
        target_series = data[target_column]
        unique_values = target_series.dropna().unique()
        value_counts = target_series.value_counts()
        
        # Suggest mappings for common exoplanet dispositions
        mapping_suggestions = {}
        
        if target_column == 'koi_disposition':
            mapping_suggestions = {
                'CONFIRMED': 'confirmed_planet',
                'CANDIDATE': 'planet_candidate', 
                'FALSE POSITIVE': 'false_positive',
                'NOT DISPOSITIONED': 'unknown'
            }
        elif target_column == 'tfopwg_disp':
            mapping_suggestions = {
                'PC': 'planet_candidate',
                'FP': 'false_positive', 
                'KP': 'known_planet',
                'CP': 'confirmed_planet'
            }
        elif target_column == 'disposition':
            mapping_suggestions = {
                'Confirmed': 'confirmed_planet',
                'Candidate': 'planet_candidate',
                'False Positive': 'false_positive'
            }
        
        return {
            'valid': True,
            'unique_values': unique_values.tolist(),
            'value_counts': value_counts.to_dict(),
            'missing_count': target_series.isnull().sum(),
            'mapping_suggestions': mapping_suggestions
        }
    
    def validate_data_format(self, data: pd.DataFrame) -> Dict[str, any]:
        """Comprehensive data validation."""
        dataset_type = self.identify_dataset_type(data)
        
        validation_results = {
            'dataset_type': dataset_type,
            'shape': data.shape,
            'columns': list(data.columns),
            'missing_data': data.isnull().sum().to_dict(),
            'dtypes': data.dtypes.astype(str).to_dict()
        }
        
        # Get target column info
        target_col = self.target_columns.get(dataset_type)
        if target_col and target_col in data.columns:
            validation_results['target_info'] = self.validate_target_column(data, target_col)
        
        # Get recommended features
        validation_results['recommended_features'] = self.get_recommended_features(data, dataset_type)
        
        return validation_results
    
    def create_clean_dataset(self, data: pd.DataFrame,
                           dataset_type: str = None,
                           use_recommended_only: bool = False) -> Tuple[pd.DataFrame, Dict[str, any]]:
        """
        Create a clean dataset ready for ML training.
        
        Args:
            data: Input dataframe
            dataset_type: Override dataset type detection
            use_recommended_only: If True, only use recommended features
        
        Returns:
            Tuple of (clean_dataframe, cleaning_report)
        """
        if dataset_type is None:
            dataset_type = self.identify_dataset_type(data)
        
        # Step 1: Filter irrelevant columns
        filtered_data, exclusion_report = self.filter_irrelevant_columns(data, dataset_type)
        
        # Step 2: Use recommended features if requested
        if use_recommended_only:
            recommended_features = self.get_recommended_features(filtered_data, dataset_type)
            target_col = self.target_columns.get(dataset_type)
            
            if target_col:
                keep_columns = recommended_features + [target_col]
            else:
                keep_columns = recommended_features
            
            # Filter to columns that exist
            keep_columns = [col for col in keep_columns if col in filtered_data.columns]
            filtered_data = filtered_data[keep_columns]
        
        # Step 3: Clean the data
        clean_data = filtered_data.copy()
        
        # Remove rows where target is missing
        target_col = self.target_columns.get(dataset_type)
        if target_col and target_col in clean_data.columns:
            before_count = len(clean_data)
            clean_data = clean_data.dropna(subset=[target_col])
            after_count = len(clean_data)
            rows_removed = before_count - after_count
        else:
            rows_removed = 0
        
        cleaning_report = {
            'dataset_type': dataset_type,
            'original_shape': data.shape,
            'filtered_shape': clean_data.shape,
            'exclusion_report': exclusion_report,
            'target_column': target_col,
            'rows_with_missing_target_removed': rows_removed,
            'use_recommended_only': use_recommended_only,
            'final_features': [col for col in clean_data.columns if col != target_col]
        }
        
        return clean_data, cleaning_report