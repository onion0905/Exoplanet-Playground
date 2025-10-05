"""
Data handling service for training workflows
"""
import os
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, List
import logging
from sklearn.model_selection import train_test_split


class DataService:
    """Service for handling different data sources and preparing them for training"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Get project root and data directory
        project_root = Path(__file__).parent.parent.parent
        self.data_dir = project_root / "data"
        self.upload_dir = project_root / "uploads"
        
    def prepare_training_data(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare training data based on configuration
        
        Args:
            config: Dictionary containing:
                - dataset_source: 'nasa' or 'upload'
                - dataset_name: for NASA datasets
                - uploaded_files: for custom uploads
                - target_column: target column name
                - test_size: size of test split (for NASA data)
        
        Returns:
            Dictionary with prepared data and metadata
        """
        try:
            dataset_source = config.get('dataset_source')
            
            if dataset_source == 'nasa':
                return self._prepare_nasa_data(config)
            elif dataset_source == 'upload':
                return self._prepare_uploaded_data(config)
            else:
                raise ValueError(f"Unknown dataset source: {dataset_source}")
                
        except Exception as e:
            self.logger.error(f"Error preparing training data: {e}")
            raise
    
    def _prepare_nasa_data(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare NASA dataset for training"""
        dataset_name = config.get('dataset_name')
        target_column = config.get('target_column', 'koi_disposition')
        test_size = config.get('test_size', 0.2)
        random_state = config.get('random_state', 42)
        
        # Dataset file mapping
        dataset_files = {
            'kepler': 'kepler_objects_of_interest.csv',
            'tess': 'tess_objects_of_interest.csv',
            'k2': 'k2_planets_and_candidates.csv'
        }
        
        if dataset_name not in dataset_files:
            raise ValueError(f"Unknown NASA dataset: {dataset_name}. Available: {list(dataset_files.keys())}")
        
        # Load dataset
        file_path = self.data_dir / dataset_files[dataset_name]
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        self.logger.info(f"Loading NASA dataset: {dataset_name} from {file_path}")
        df = pd.read_csv(file_path, comment='#')
        
        # Filter for three-class classification: confirmed, candidate, false positive
        if dataset_name == 'kepler' and target_column == 'koi_disposition':
            allowed_values = {'CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE'}
        elif dataset_name == 'k2' and target_column == 'disposition':
            allowed_values = {'Confirmed', 'Candidate', 'False Positive'}
        elif dataset_name == 'tess' and target_column == 'tfopwg_disp':
            allowed_values = {'PC', 'CP', 'FP'}  # Planet Candidate, Confirmed Planet, False Positive
        else:
            # Use all available values if we can't determine the expected ones
            allowed_values = set(df[target_column].dropna().unique())
        
        # Filter data
        before_filter = len(df)
        df = df[df[target_column].isin(allowed_values)].copy()
        after_filter = len(df)
        
        self.logger.info(f"Filtered {before_filter - after_filter} rows, kept {after_filter} rows with values: {allowed_values}")
        
        # Prepare features and target
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        # Get feature columns (exclude target and non-numeric columns)
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        feature_columns = [col for col in numeric_columns if col != target_column]
        
        if not feature_columns:
            raise ValueError("No numeric feature columns found in dataset")
        
        # Prepare data
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        return {
            'data_source': 'nasa',
            'dataset_name': dataset_name,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_columns': feature_columns,
            'target_column': target_column,
            'target_classes': list(allowed_values),
            'original_shape': df.shape,
            'train_shape': X_train.shape,
            'test_shape': X_test.shape,
            'metadata': {
                'test_size': test_size,
                'random_state': random_state,
                'filtered_classes': list(allowed_values),
                'total_features': len(feature_columns)
            }
        }
    
    def _prepare_uploaded_data(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare uploaded custom dataset for training"""
        uploaded_files = config.get('uploaded_files', {})
        target_column = config.get('target_column')
        
        if not target_column:
            raise ValueError("Target column must be specified for uploaded data")
        
        # Handle different upload scenarios
        if 'training_file' in uploaded_files and 'testing_file' in uploaded_files:
            # Separate train/test files provided
            return self._load_separate_train_test_files(uploaded_files, target_column)
        elif 'data_file' in uploaded_files:
            # Single file provided, need to split
            return self._load_single_file_and_split(uploaded_files, target_column, config)
        else:
            raise ValueError("Invalid upload configuration. Provide either 'data_file' or both 'training_file' and 'testing_file'")
    
    def _load_separate_train_test_files(self, files: Dict[str, str], target_column: str) -> Dict[str, Any]:
        """Load separate training and testing files"""
        train_file = files['training_file']
        test_file = files['testing_file']
        
        # Load training data
        train_path = self.upload_dir / train_file
        if not train_path.exists():
            raise FileNotFoundError(f"Training file not found: {train_path}")
        
        self.logger.info(f"Loading training data from: {train_path}")
        if train_path.suffix.lower() == '.csv':
            train_df = pd.read_csv(train_path)
        else:
            train_df = pd.read_excel(train_path)
        
        # Load testing data
        test_path = self.upload_dir / test_file
        if not test_path.exists():
            raise FileNotFoundError(f"Testing file not found: {test_path}")
        
        self.logger.info(f"Loading testing data from: {test_path}")
        if test_path.suffix.lower() == '.csv':
            test_df = pd.read_csv(test_path)
        else:
            test_df = pd.read_excel(test_path)
        
        # Validate columns
        if target_column not in train_df.columns:
            raise ValueError(f"Target column '{target_column}' not found in training data")
        if target_column not in test_df.columns:
            raise ValueError(f"Target column '{target_column}' not found in testing data")
        
        # Get common feature columns
        train_features = [col for col in train_df.columns if col != target_column]
        test_features = [col for col in test_df.columns if col != target_column]
        common_features = list(set(train_features) & set(test_features))
        
        if not common_features:
            raise ValueError("No common feature columns found between training and testing data")
        
        # Select only numeric columns for features
        numeric_features = []
        for col in common_features:
            if pd.api.types.is_numeric_dtype(train_df[col]) and pd.api.types.is_numeric_dtype(test_df[col]):
                numeric_features.append(col)
        
        if not numeric_features:
            raise ValueError("No common numeric feature columns found")
        
        # Prepare final datasets
        X_train = train_df[numeric_features].fillna(train_df[numeric_features].median())
        y_train = train_df[target_column]
        X_test = test_df[numeric_features].fillna(test_df[numeric_features].median())
        y_test = test_df[target_column]
        
        # Get target classes
        all_classes = set(y_train.unique()) | set(y_test.unique())
        
        return {
            'data_source': 'upload',
            'upload_type': 'separate_files',
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_columns': numeric_features,
            'target_column': target_column,
            'target_classes': list(all_classes),
            'original_shape': (len(train_df) + len(test_df), len(train_df.columns)),
            'train_shape': X_train.shape,
            'test_shape': X_test.shape,
            'metadata': {
                'training_file': train_file,
                'testing_file': test_file,
                'total_features': len(numeric_features)
            }
        }
    
    def _load_single_file_and_split(self, files: Dict[str, str], target_column: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Load single file and split into train/test"""
        data_file = files['data_file']
        test_size = config.get('test_size', 0.2)
        random_state = config.get('random_state', 42)
        
        # Load data file
        data_path = self.upload_dir / data_file
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        self.logger.info(f"Loading data from: {data_path}")
        if data_path.suffix.lower() == '.csv':
            df = pd.read_csv(data_path)
        else:
            df = pd.read_excel(data_path)
        
        # Validate target column
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Get numeric feature columns
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        feature_columns = [col for col in numeric_columns if col != target_column]
        
        if not feature_columns:
            raise ValueError("No numeric feature columns found in data")
        
        # Prepare data
        X = df[feature_columns].fillna(df[feature_columns].median())
        y = df[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        return {
            'data_source': 'upload',
            'upload_type': 'single_file',
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_columns': feature_columns,
            'target_column': target_column,
            'target_classes': list(y.unique()),
            'original_shape': df.shape,
            'train_shape': X_train.shape,
            'test_shape': X_test.shape,
            'metadata': {
                'data_file': data_file,
                'test_size': test_size,
                'random_state': random_state,
                'total_features': len(feature_columns)
            }
        }


# Global data service instance
data_service = DataService()