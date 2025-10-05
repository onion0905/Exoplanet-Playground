from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import pickle
from .base_model import BaseExoplanetModel


class DeepLearningModel(BaseExoplanetModel):
    """Deep Learning (Neural Network) model for exoplanet classification."""
    
    def __init__(self):
        super().__init__("Deep Learning")
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model_history = None
        
    def build_model(self, **hyperparameters) -> None:
        """Initialize neural network model with hyperparameters."""
        default_params = {
            'hidden_layers': [128, 64, 32],
            'activation': 'relu',
            'dropout_rate': 0.3,
            'use_batch_norm': True,
            'learning_rate': 0.001,
            'optimizer': 'adam'
        }
        default_params.update(hyperparameters)
        
        # Store hyperparameters for model building
        self.hyperparameters = default_params
        
    def _create_model_architecture(self, input_dim: int, n_classes: int) -> tf.keras.Model:
        """Create the neural network architecture."""
        model = Sequential()
        
        # Input layer
        model.add(Dense(
            self.hyperparameters['hidden_layers'][0],
            input_dim=input_dim,
            activation=self.hyperparameters['activation']
        ))
        
        if self.hyperparameters['use_batch_norm']:
            model.add(BatchNormalization())
            
        if self.hyperparameters['dropout_rate'] > 0:
            model.add(Dropout(self.hyperparameters['dropout_rate']))
        
        # Hidden layers
        for layer_size in self.hyperparameters['hidden_layers'][1:]:
            model.add(Dense(layer_size, activation=self.hyperparameters['activation']))
            
            if self.hyperparameters['use_batch_norm']:
                model.add(BatchNormalization())
                
            if self.hyperparameters['dropout_rate'] > 0:
                model.add(Dropout(self.hyperparameters['dropout_rate']))
        
        # Output layer
        if n_classes == 2:
            model.add(Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:
            model.add(Dense(n_classes, activation='softmax'))
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        
        # Compile model
        if self.hyperparameters['optimizer'] == 'adam':
            optimizer = Adam(learning_rate=self.hyperparameters['learning_rate'])
        else:
            optimizer = self.hyperparameters['optimizer']
            
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        return model
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: Optional[pd.DataFrame] = None, 
              y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Train the neural network model."""
        if not hasattr(self, 'hyperparameters'):
            self.build_model()
            
        self.feature_names = list(X_train.columns)
        self.target_classes = list(y_train.unique())
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        # Create model
        n_classes = len(self.target_classes)
        self.model = self._create_model_architecture(X_train_scaled.shape[1], n_classes)
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            y_val_encoded = self.label_encoder.transform(y_val)
            validation_data = (X_val_scaled, y_val_encoded)
        
        # Prepare callbacks
        callbacks = []
        if validation_data is not None:
            callbacks.extend([
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
            ])
        
        # Train model
        history = self.model.fit(
            X_train_scaled,
            y_train_encoded,
            epochs=100,
            batch_size=32,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=0
        )
        
        self.model_history = history.history
        self.is_trained = True
        
        # Calculate training metrics
        train_predictions = self.model.predict(X_train_scaled, verbose=0)
        if n_classes == 2:
            train_pred_classes = (train_predictions > 0.5).astype(int).flatten()
        else:
            train_pred_classes = np.argmax(train_predictions, axis=1)
            
        train_accuracy = np.mean(train_pred_classes == y_train_encoded)
        
        metrics = {
            'train_accuracy': train_accuracy,
            'n_epochs_trained': len(history.history['loss']),
            'final_train_loss': history.history['loss'][-1],
            'model_parameters': self.model.count_params()
        }
        
        # Validation metrics if validation data provided
        if validation_data is not None:
            val_predictions = self.model.predict(X_val_scaled, verbose=0)
            if n_classes == 2:
                val_pred_classes = (val_predictions > 0.5).astype(int).flatten()
            else:
                val_pred_classes = np.argmax(val_predictions, axis=1)
                
            val_accuracy = np.mean(val_pred_classes == y_val_encoded)
            metrics['val_accuracy'] = val_accuracy
            metrics['final_val_loss'] = history.history['val_loss'][-1]
            
        self.training_history = metrics
        return metrics
    
    def predict(self, X: pd.DataFrame, explain: bool = False):
        """Predict class labels for samples in X. If explain=True, return per-sample top 5 feature names and confidence."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        X = self.preprocess_input(X)
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled, verbose=0)
        n_classes = len(self.target_classes)
        if n_classes == 2:
            pred_classes = (predictions > 0.5).astype(int).flatten()
            proba = np.column_stack([1 - predictions.flatten(), predictions.flatten()])
        else:
            pred_classes = np.argmax(predictions, axis=1)
            proba = predictions
        labels = self.label_encoder.inverse_transform(pred_classes)
        if not explain:
            return labels
        class_indices = [list(self.label_encoder.classes_).index(l) for l in labels]
        confidences = [proba[i, idx] for i, idx in enumerate(class_indices)]
        explanations = []
        try:
            import shap
            explainer = shap.DeepExplainer(self.model, X_scaled)
            shap_values = explainer.shap_values(X_scaled)
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
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled, verbose=0)
        
        n_classes = len(self.target_classes)
        if n_classes == 2:
            # Convert binary predictions to class probabilities
            proba = np.column_stack([1 - predictions.flatten(), predictions.flatten()])
        else:
            proba = predictions
            
        return proba
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance using gradient-based method.
        Note: This is a simplified approximation for neural networks.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        # For neural networks, feature importance is not straightforward
        # This is a simplified approach using input layer weights
        try:
            # Get weights from first layer
            first_layer_weights = self.model.layers[0].get_weights()[0]
            
            # Calculate importance as mean absolute weight per input feature
            importance_scores = np.mean(np.abs(first_layer_weights), axis=1)
            
            # Normalize to sum to 1
            importance_scores = importance_scores / np.sum(importance_scores)
            
            return dict(zip(self.feature_names, importance_scores))
        except Exception as e:
            # Fallback to uniform importance
            uniform_importance = 1.0 / len(self.feature_names)
            return {name: uniform_importance for name in self.feature_names}
    
    def get_training_history(self) -> Dict[str, List[float]]:
        """Get training history (loss, accuracy curves)."""
        if self.model_history is None:
            raise ValueError("Model must be trained to get training history")
        return self.model_history
    
    def save_model(self, filepath: str) -> None:
        """Save model, scaler, and label encoder."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        # Save Keras model with .keras extension
        self.model.save(f"{filepath}_keras_model.keras")
        
        # Save other components
        model_data = {
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'target_classes': self.target_classes,
            'training_history': self.training_history,
            'model_history': self.model_history,
            'hyperparameters': self.hyperparameters
        }
        
        with open(f"{filepath}_metadata.pkl", 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str) -> None:
        """Load model, scaler, and label encoder."""
        # Load Keras model with .keras extension
        self.model = tf.keras.models.load_model(f"{filepath}_keras_model.keras")
        
        # Load other components
        with open(f"{filepath}_metadata.pkl", 'rb') as f:
            model_data = pickle.load(f)
            
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.target_classes = model_data['target_classes']
        self.training_history = model_data['training_history']
        self.model_history = model_data['model_history']
        self.hyperparameters = model_data['hyperparameters']
        self.is_trained = True