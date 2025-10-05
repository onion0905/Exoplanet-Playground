from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from pathlib import Path
import logging


class DataLoader:
    """Loads and manages exoplanet datasets from various sources."""
    
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            # Use centralized config
            from ..config import DATA_DIR
            data_dir = DATA_DIR
        
        self.data_dir = Path(data_dir)
        self.datasets = {}
        self.logger = logging.getLogger(__name__)
        
    def load_nasa_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Load a NASA dataset by name."""
        dataset_files = {
            'kepler': 'kepler_raw.csv',
            'tess': 'tess_raw.csv',
            'k2': 'k2_raw.csv'
        }
        
        if dataset_name not in dataset_files:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(dataset_files.keys())}")
        
        filepath = self.data_dir / dataset_files[dataset_name]
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
        try:
            # Read CSV, skipping comment lines that start with #
            df = pd.read_csv(filepath, comment='#')
            self.logger.info(f"Loaded {dataset_name} dataset: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Cache the dataset
            self.datasets[dataset_name] = df
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading {dataset_name} dataset: {str(e)}")
            raise
    
    def load_user_dataset(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """Load a user-uploaded dataset."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"User dataset file not found: {filepath}")
        
        try:
            # Determine file type and read accordingly
            if filepath.suffix.lower() == '.csv':
                df = pd.read_csv(filepath)
            elif filepath.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(filepath)
            elif filepath.suffix.lower() == '.json':
                df = pd.read_json(filepath)
            else:
                raise ValueError(f"Unsupported file format: {filepath.suffix}")
            
            self.logger.info(f"Loaded user dataset: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading user dataset: {str(e)}")
            raise
    
    def load_all_nasa_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load all available NASA datasets."""
        datasets = {}
        for dataset_name in ['kepler', 'tess', 'k2']:
            try:
                datasets[dataset_name] = self.load_nasa_dataset(dataset_name)
            except Exception as e:
                self.logger.warning(f"Could not load {dataset_name}: {str(e)}")
        
        return datasets
    
    def get_cached_dataset(self, dataset_name: str) -> Optional[pd.DataFrame]:
        """Get a cached dataset if available."""
        return self.datasets.get(dataset_name)
    
    def combine_datasets(self, dataset_names: List[str], 
                        common_columns_only: bool = True) -> pd.DataFrame:
        """Combine multiple datasets into one."""
        if not dataset_names:
            raise ValueError("At least one dataset name must be provided")
        
        # Load datasets if not cached
        dfs = []
        for name in dataset_names:
            if name in self.datasets:
                df = self.datasets[name].copy()
            else:
                df = self.load_nasa_dataset(name)
            
            # Add source column
            df['source_dataset'] = name
            dfs.append(df)
        
        if common_columns_only:
            # Find common columns across all datasets
            common_cols = set(dfs[0].columns)
            for df in dfs[1:]:
                common_cols = common_cols.intersection(set(df.columns))
            
            # Keep only common columns
            dfs = [df[list(common_cols)] for df in dfs]
            self.logger.info(f"Combined datasets using {len(common_cols)} common columns")
        
        # Concatenate datasets
        combined_df = pd.concat(dfs, ignore_index=True, sort=False)
        self.logger.info(f"Combined dataset shape: {combined_df.shape[0]} rows, {combined_df.shape[1]} columns")
        
        return combined_df
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, any]:
        """Get information about a dataset."""
        if dataset_name not in self.datasets:
            try:
                self.load_nasa_dataset(dataset_name)
            except Exception as e:
                return {'error': str(e)}
        
        df = self.datasets[dataset_name]
        
        info = {
            'name': dataset_name,
            'shape': df.shape,
            'columns': list(df.columns),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'missing_values': df.isnull().sum().to_dict(),
            'dtypes': df.dtypes.to_dict()
        }
        
        return info
    
    def sample_dataset(self, dataset_name: str, n_samples: int = 1000, 
                      random_state: int = 42) -> pd.DataFrame:
        """Get a sample from a dataset for quick testing."""
        if dataset_name not in self.datasets:
            self.load_nasa_dataset(dataset_name)
        
        df = self.datasets[dataset_name]
        if len(df) <= n_samples:
            return df.copy()
        
        return df.sample(n=n_samples, random_state=random_state)