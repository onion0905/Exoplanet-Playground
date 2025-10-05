from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
from .base_model import BaseExoplanetModel


class PCAModel(BaseExoplanetModel):
    """PCA + Logistic Regression model for exoplanet classification."""
    
    def __init__(self):
        super().__init__("PCA")
        self.pca = None
        self.classifier = None
        self.scaler = StandardScaler()
        self.explained_variance_ratio = None
        
    def build_model(self, **hyperparameters) -> None:
        """Initialize PCA + classifier model with hyperparameters."""
        pca_params = {
            'n_components': 0.95,  # Keep 95% of variance by default
            'random_state': 42
        }
        
        classifier_params = {
            'C': 1.0,
            'max_iter': 1000,
            'solver': 'lbfgs',
            'random_state': 42
        }
        
        # Update with any provided hyperparameters
        for key, value in hyperparameters.items():
            if key.startswith('pca_'):
                pca_params[key[4:]] = value  # Remove 'pca_' prefix
            elif key.startswith('classifier_'):
                classifier_params[key[11:]] = value  # Remove 'classifier_' prefix
            elif key in ['n_components']:
                pca_params[key] = value
            else:
                classifier_params[key] = value
        
        self.pca = PCA(**pca_params)
        self.classifier = LogisticRegression(**classifier_params)
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: Optional[pd.DataFrame] = None, 
              y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Train the PCA + classifier model."""
        if self.pca is None or self.classifier is None:
            self.build_model()
            
        self.feature_names = list(X_train.columns)
        self.target_classes = list(y_train.unique())
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Apply PCA
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        self.explained_variance_ratio = self.pca.explained_variance_ratio_
        
        # Train classifier on PCA components
        self.classifier.fit(X_train_pca, y_train)
        self.is_trained = True
        
        # Calculate training metrics
        train_predictions = self.classifier.predict(X_train_pca)
        train_accuracy = np.mean(train_predictions == y_train)
        
        metrics = {
            'train_accuracy': train_accuracy,
            'n_components': self.pca.n_components_,
            'explained_variance_ratio': self.explained_variance_ratio.tolist(),
            'cumulative_variance_explained': np.cumsum(self.explained_variance_ratio).tolist(),
            'total_variance_explained': np.sum(self.explained_variance_ratio)
        }
        
        # Validation metrics if validation data provided
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            X_val_pca = self.pca.transform(X_val_scaled)
            val_predictions = self.classifier.predict(X_val_pca)
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
        X_pca = self.pca.transform(X_scaled)
        return self.classifier.predict(X_pca)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
            
        X = self.preprocess_input(X)
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        return self.classifier.predict_proba(X_pca)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance by mapping PCA component importances back to original features.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        # Get classifier coefficients
        if len(self.classifier.coef_.shape) > 1:
            coef = np.mean(np.abs(self.classifier.coef_), axis=0)
        else:
            coef = np.abs(self.classifier.coef_[0])
        
        # Weight by explained variance ratio
        weighted_coef = coef * self.explained_variance_ratio
        
        # Map back to original features using PCA components
        # Each component's contribution to each original feature
        components = np.abs(self.pca.components_)
        feature_importance = np.dot(weighted_coef, components)
        
        return dict(zip(self.feature_names, feature_importance))
    
    def get_pca_components(self) -> Dict[str, Any]:
        """Get PCA component information."""
        if not self.is_trained:
            raise ValueError("Model must be trained to get PCA components")
            
        return {
            'components': self.pca.components_.tolist(),
            'explained_variance_ratio': self.explained_variance_ratio.tolist(),
            'n_components': self.pca.n_components_,
            'feature_names': self.feature_names
        }
    
    def transform_to_pca_space(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data to PCA space (for visualization)."""
        if not self.is_trained:
            raise ValueError("Model must be trained before transformation")
            
        X = self.preprocess_input(X)
        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled)
    
    def save_model(self, filepath: str) -> None:
        """Save model, PCA, and scaler."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
            
        model_data = {
            'pca': self.pca,
            'classifier': self.classifier,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'target_classes': self.target_classes,
            'training_history': self.training_history,
            'explained_variance_ratio': self.explained_variance_ratio
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str) -> None:
        """Load model, PCA, and scaler."""
        model_data = joblib.load(filepath)
        self.pca = model_data['pca']
        self.classifier = model_data['classifier']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.target_classes = model_data['target_classes']
        self.training_history = model_data['training_history']
        self.explained_variance_ratio = model_data['explained_variance_ratio']
        self.is_trained = True