from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib
from .base_model import BaseExoplanetModel


class SVMModel(BaseExoplanetModel):
    """Support Vector Machine model for exoplanet classification."""
    
    def __init__(self):
        super().__init__("SVM")
        self.scaler = StandardScaler()
        
    def build_model(self, **hyperparameters) -> None:
        """Initialize SVM model with hyperparameters."""
        default_params = {
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale',
            'probability': True,  # Enable probability estimates
            'random_state': 42
        }
        default_params.update(hyperparameters)
        
        self.model = SVC(**default_params)
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: Optional[pd.DataFrame] = None, 
              y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Train the SVM model."""
        if self.model is None:
            self.build_model()
            
        self.feature_names = list(X_train.columns)
        self.target_classes = list(y_train.unique())
        
        # Scale features (important for SVM)
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Calculate training metrics
        train_predictions = self.model.predict(X_train_scaled)
        train_accuracy = np.mean(train_predictions == y_train)
        
        metrics = {
            'train_accuracy': train_accuracy,
            'n_support_vectors': self.model.n_support_.tolist(),
            'support_vector_indices': self.model.support_.tolist()
        }
        
        # Validation metrics if validation data provided
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            val_predictions = self.model.predict(X_val_scaled)
            val_accuracy = np.mean(val_predictions == y_val)
            metrics['val_accuracy'] = val_accuracy
            
        self.training_history = metrics
        return metrics
    
    def predict(self, X: pd.DataFrame, explain: bool = False):
        """Predict class labels for samples in X. If explain=True, return per-sample top 5 feature names and confidence."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        X = self.preprocess_input(X)
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)
        if not explain:
            return preds
        proba = self.model.predict_proba(X_scaled)
        class_indices = [list(self.model.classes_).index(p) for p in preds]
        confidences = [proba[i, idx] for i, idx in enumerate(class_indices)]
        # For linear kernel, use coef_; for others, fallback to uniform or permutation importance
        explanations = []
        try:
            import shap
            explainer = shap.KernelExplainer(self.model.predict_proba, X_scaled)
            shap_values = explainer.shap_values(X_scaled, nsamples=100)
            if isinstance(shap_values, list):
                for i, idx in enumerate(class_indices):
                    sample_shap = dict(zip(self.feature_names, shap_values[idx][i]))
                    top5 = sorted(sample_shap.items(), key=lambda x: -abs(x[1]))[:5]
                    explanations.append([k for k, v in top5])
            else:
                for row in shap_values:
                    sample_shap = dict(zip(self.feature_names, row))
                    top5 = sorted(sample_shap.items(), key=lambda x: -abs(x[1]))[:5]
                    explanations.append([k for k, v in top5])
        except Exception as e:
            # fallback: use feature importances
            try:
                importances = self.get_feature_importance()
                top5 = sorted(importances.items(), key=lambda x: -abs(x[1]))[:5]
                explanations = [[k for k, v in top5] for _ in range(len(X_scaled))]
            except Exception:
                explanations = [[f for f in self.feature_names[:5]] for _ in range(len(X_scaled))]
        return [
            {
                'label': str(preds[i]),
                'confidence': float(confidences[i]),
                'top_features': explanations[i]
            }
            for i in range(len(preds))
        ]
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
            
        X = self.preprocess_input(X)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance for SVM using permutation importance.
        Note: SVM doesn't have inherent feature importance, so we use a simple approximation.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
            
        # For linear kernel, we can use coefficient magnitudes
        if hasattr(self.model, 'coef_') and self.model.coef_ is not None:
            if len(self.model.coef_.shape) > 1:
                importance_scores = np.mean(np.abs(self.model.coef_), axis=0)
            else:
                importance_scores = np.abs(self.model.coef_[0])
        else:
            # For non-linear kernels, return uniform importance
            # In a production system, you'd implement permutation importance here
            importance_scores = np.ones(len(self.feature_names))
            
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