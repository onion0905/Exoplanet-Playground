from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import joblib
from sklearn.preprocessing import LabelEncoder
from .base_model import BaseExoplanetModel


class XGBoostModel(BaseExoplanetModel):
    """XGBoost model for exoplanet classification."""
    
    def __init__(self):
        super().__init__("XGBoost")
        self.label_encoder = LabelEncoder()
        
    def build_model(self, **hyperparameters) -> None:
        """Initialize XGBoost model with hyperparameters."""
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 1.0,
            'colsample_bytree': 1.0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'random_state': 42,
            'eval_metric': 'mlogloss',
            'use_label_encoder': False
        }
        default_params.update(hyperparameters)
        
        self.model = XGBClassifier(**default_params)
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: Optional[pd.DataFrame] = None, 
              y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Train the XGBoost model."""
        if self.model is None:
            self.build_model()
            
        self.feature_names = list(X_train.columns)
        self.target_classes = list(y_train.unique())
        
        # Encode string labels to numeric values
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        # Prepare evaluation set for early stopping if validation data provided
        eval_set = None
        y_val_encoded = None
        if X_val is not None and y_val is not None:
            y_val_encoded = self.label_encoder.transform(y_val)
            eval_set = [(X_val, y_val_encoded)]
        
        # Train model
        self.model.fit(
            X_train, 
            y_train_encoded,
            eval_set=eval_set,
            verbose=False
        )
        self.is_trained = True
        
        # Calculate training metrics
        train_predictions_encoded = self.model.predict(X_train)
        train_accuracy = np.mean(train_predictions_encoded == y_train_encoded)
        
        metrics = {
            'train_accuracy': train_accuracy,
            'n_estimators': self.model.n_estimators,
            'feature_importances': self.model.feature_importances_.tolist(),
            'best_iteration': getattr(self.model, 'best_iteration', None)
        }
        
        # Validation metrics if validation data provided
        if X_val is not None and y_val is not None:
            val_predictions_encoded = self.model.predict(X_val)
            val_accuracy = np.mean(val_predictions_encoded == y_val_encoded)
            metrics['val_accuracy'] = val_accuracy
            
        self.training_history = metrics
        return metrics
    
    def predict(self, X: pd.DataFrame, explain: bool = False):
        """Predict class labels for samples in X. If explain=True, return per-sample top 5 feature names and confidence."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        X = self.preprocess_input(X)
        predictions_encoded = self.model.predict(X)
        labels = self.label_encoder.inverse_transform(predictions_encoded)
        if not explain:
            return labels
        proba = self.model.predict_proba(X)
        class_indices = [list(self.label_encoder.classes_).index(l) for l in labels]
        confidences = [proba[i, idx] for i, idx in enumerate(class_indices)]
        explanations = []
        try:
            import shap
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X)
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
                explanations = [[k for k, v in top5] for _ in range(len(X))]
            except Exception:
                explanations = [[f for f in self.feature_names[:5]] for _ in range(len(X))]
        return [
            {
                'label': str(labels[i]),
                'confidence': float(confidences[i]),
                'top_features': explanations[i]
            }
            for i in range(len(labels))
        ]
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
            
        X = self.preprocess_input(X)
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from XGBoost."""
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
            
        importance_scores = self.model.feature_importances_
        return dict(zip(self.feature_names, importance_scores))
    
    def get_feature_importance_detailed(self) -> Dict[str, Dict[str, float]]:
        """Get detailed feature importance with different importance types."""
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        importance_types = ['weight', 'gain', 'cover']
        detailed_importance = {}
        
        for imp_type in importance_types:
            try:
                importance_dict = self.model.get_booster().get_score(importance_type=imp_type)
                # Map feature indices back to feature names
                mapped_importance = {}
                for i, feature_name in enumerate(self.feature_names):
                    key = f'f{i}'  # XGBoost uses f0, f1, f2, ... as default feature names
                    mapped_importance[feature_name] = importance_dict.get(key, 0.0)
                detailed_importance[imp_type] = mapped_importance
            except:
                # Fallback to sklearn-style importance
                detailed_importance[imp_type] = dict(zip(self.feature_names, self.model.feature_importances_))
                
        return detailed_importance
    
    def save_model(self, filepath: str) -> None:
        """Save model."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
            
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'target_classes': self.target_classes,
            'training_history': self.training_history
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str) -> None:
        """Load model."""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.target_classes = model_data['target_classes']
        self.training_history = model_data['training_history']
        self.is_trained = True