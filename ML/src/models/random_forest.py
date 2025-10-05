from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
from .base_model import BaseExoplanetModel


class RandomForestModel(BaseExoplanetModel):
    def predict(self, X: pd.DataFrame, explain: bool = False):
        """Predict class labels for samples in X. If explain=True, return per-sample top 5 feature names and confidence."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        X = self.preprocess_input(X)
        preds = self.model.predict(X)
        if not explain:
            return preds
        # Compute confidence scores (max probability for predicted class)
        proba = self.model.predict_proba(X)
        class_indices = [list(self.model.classes_).index(p) for p in preds]
        confidences = [proba[i, idx] for i, idx in enumerate(class_indices)]
        # Compute per-sample feature importances using tree SHAP (mean absolute SHAP value per feature)
        try:
            import shap
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X)
            # For multiclass, shap_values is a list (one per class)
            if isinstance(shap_values, list):
                # Pick the SHAP values for the predicted class for each sample
                explanations = []
                for i, idx in enumerate(class_indices):
                    sample_shap = dict(zip(X.columns, shap_values[idx][i]))
                    # Get top 5 by absolute value
                    top5 = sorted(sample_shap.items(), key=lambda x: -abs(x[1]))[:5]
                    explanations.append([k for k, v in top5])
            else:
                explanations = []
                for row in shap_values:
                    sample_shap = dict(zip(X.columns, row))
                    top5 = sorted(sample_shap.items(), key=lambda x: -abs(x[1]))[:5]
                    explanations.append([k for k, v in top5])
        except Exception as e:
            # fallback: use feature importances
            try:
                importances = self.get_feature_importance()
                top5 = sorted(importances.items(), key=lambda x: -abs(x[1]))[:5]
                explanations = [[k for k, v in top5] for _ in range(len(X))]
            except Exception:
                explanations = [[f for f in self.feature_names[:5]] for _ in range(len(X))]
        # Determine dataset type by feature names
        is_kepler = any(f.startswith('koi_') for f in self.feature_names)
        is_k2 = any(f in ['disp_refname', 'disc_refname', 'default_flag', 'st_refname'] for f in self.feature_names)
        kepler_exclude = ['koi_pdisposition', 'koi_score', 'koi_fpflag']
        k2_exclude = ['disp_refname', 'disc_refname', 'default_flag', 'st_refname']
        def filter_features(feats):
            if is_kepler:
                allowed = [f for f in feats if not any(f.startswith(prefix) for prefix in kepler_exclude)]
            elif is_k2:
                allowed = [f for f in feats if f not in k2_exclude]
            else:
                allowed = feats
            return allowed[:5]
        result = []
        for i in range(len(preds)):
            feats = explanations[i]
            filtered = filter_features(feats)
            result.append({
                'label': str(preds[i]),
                'confidence': float(confidences[i]),
                'top_features': filtered
            })
        return result
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