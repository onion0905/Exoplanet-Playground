from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
from .base_model import BaseExoplanetModel


class RandomForestModel(BaseExoplanetModel):
    """Random Forest model for exoplanet classification."""
    
    def __init__(self):
        super().__init__("Random Forest")
        
    def build_model(self, **hyperparameters) -> None:
        """Initialize Random Forest model with hyperparameters."""
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'criterion': 'gini',
            'random_state': 42,
            'n_jobs': -1  # Use all available cores
        }
        default_params.update(hyperparameters)
        
        self.model = RandomForestClassifier(**default_params)
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: Optional[pd.DataFrame] = None, 
              y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Train the Random Forest model."""
        if self.model is None:
            self.build_model()
            
        self.feature_names = list(X_train.columns)
        self.target_classes = list(y_train.unique())
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Calculate training metrics
        train_predictions = self.model.predict(X_train)
        train_accuracy = np.mean(train_predictions == y_train)
        
        metrics = {
            'train_accuracy': train_accuracy,
            'n_estimators': self.model.n_estimators,
            'feature_importances': self.model.feature_importances_.tolist(),
            'oob_score': getattr(self.model, 'oob_score_', None)
        }
        
        # Validation metrics if validation data provided
        if X_val is not None and y_val is not None:
            val_predictions = self.model.predict(X_val)
            val_accuracy = np.mean(val_predictions == y_val)
            metrics['val_accuracy'] = val_accuracy
            
        self.training_history = metrics
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
            
        X = self.preprocess_input(X)
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
            
        X = self.preprocess_input(X)
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from random forest."""
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
            
        importance_scores = self.model.feature_importances_
        return dict(zip(self.feature_names, importance_scores))
    
    def get_tree_count(self) -> int:
        """Get number of trees in the forest."""
        return self.model.n_estimators if self.is_trained else 0
    
    def save_model(self, filepath: str) -> None:
        """Save model."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
            
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'target_classes': self.target_classes,
            'training_history': self.training_history
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str) -> None:
        """Load model."""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.target_classes = model_data['target_classes']
        self.training_history = model_data['training_history']
        self.is_trained = True