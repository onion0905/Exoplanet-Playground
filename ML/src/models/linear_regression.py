from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
from .base_model import BaseExoplanetModel


class LinearRegressionModel(BaseExoplanetModel):
    """Linear/Logistic Regression model for exoplanet classification."""
    
    def __init__(self):
        super().__init__("Linear Regression")
        self.scaler = StandardScaler()
        
    def build_model(self, **hyperparameters) -> None:
        """Initialize logistic regression model with hyperparameters."""
        default_params = {
            'C': 1.0,
            'max_iter': 1000,
            'solver': 'lbfgs',
            'random_state': 42
        }
        default_params.update(hyperparameters)
        
        self.model = LogisticRegression(**default_params)
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: Optional[pd.DataFrame] = None, 
              y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Train the logistic regression model."""
        if self.model is None:
            self.build_model()
            
        self.feature_names = list(X_train.columns)
        self.target_classes = list(y_train.unique())
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Calculate training metrics
        train_predictions = self.model.predict(X_train_scaled)
        train_accuracy = np.mean(train_predictions == y_train)
        
        metrics = {
            'train_accuracy': train_accuracy,
            'model_coefficients': self.model.coef_.tolist(),
            'intercept': self.model.intercept_.tolist()
        }
        
        # Validation metrics if validation data provided
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            val_predictions = self.model.predict(X_val_scaled)
            val_accuracy = np.mean(val_predictions == y_val)
            metrics['val_accuracy'] = val_accuracy
            
        self.training_history = metrics
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
            
        X = self.preprocess_input(X)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
            
        X = self.preprocess_input(X)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance based on coefficient magnitudes."""
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
            
        # For multi-class, use mean of absolute coefficients across classes
        if len(self.model.coef_.shape) > 1:
            importance_scores = np.mean(np.abs(self.model.coef_), axis=0)
        else:
            importance_scores = np.abs(self.model.coef_[0])
            
        return dict(zip(self.feature_names, importance_scores))
    
    def save_model(self, filepath: str) -> None:
        """Save model and scaler."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
            
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'target_classes': self.target_classes,
            'training_history': self.training_history
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str) -> None:
        """Load model and scaler."""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.target_classes = model_data['target_classes']
        self.training_history = model_data['training_history']
        self.is_trained = True