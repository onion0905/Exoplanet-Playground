from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class BaseExoplanetModel(ABC):
    """
    Abstract base class for all exoplanet classification models.
    Provides common interface and functionality for model training, prediction, and explainability.
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        self.feature_names = None
        self.target_classes = None
        self.training_history = {}
        
    @abstractmethod
    def build_model(self, **hyperparameters) -> None:
        """Initialize the model with given hyperparameters."""
        pass
    
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: Optional[pd.DataFrame] = None, 
              y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Train the model and return training metrics."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame, explain: bool = False):
        """Make predictions on input data. If explain=True, return per-sample explanations and confidence."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return prediction probabilities."""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        pass
    
    def validate_input(self, X: pd.DataFrame) -> bool:
        """Validate that input data has correct format and features."""
        if not self.is_trained:
            return True  # No validation needed before training
            
        if self.feature_names is None:
            return False
            
        # Check if all required features are present
        missing_features = set(self.feature_names) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
            
        # Check for extra features (should be subset)
        extra_features = set(X.columns) - set(self.feature_names)
        if extra_features:
            print(f"Warning: Extra features found (will be ignored): {extra_features}")
            
        return True
    
    def preprocess_input(self, X: pd.DataFrame) -> pd.DataFrame:
        """Preprocess input data to match training format."""
        if self.feature_names is not None:
            # Only keep features that were used during training
            X = X[self.feature_names]
        return X
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluate model performance on test data."""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
            
        self.validate_input(X_test)
        X_test = self.preprocess_input(X_test)
        
        predictions = self.predict(X_test)
        probabilities = self.predict_proba(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'classification_report': classification_report(y_test, predictions),
            'confusion_matrix': confusion_matrix(y_test, predictions).tolist(),
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist()
        }
        
        return metrics
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and status."""
        return {
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names,
            'target_classes': self.target_classes,
            'training_history': self.training_history
        }
    
    def save_model(self, filepath: str) -> None:
        """Save trained model to file."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        # Implementation depends on specific model type
        pass
    
    def load_model(self, filepath: str) -> None:
        """Load trained model from file."""
        # Implementation depends on specific model type
        pass