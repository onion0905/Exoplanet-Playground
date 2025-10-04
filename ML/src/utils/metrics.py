from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, roc_curve
)
import logging


class ModelMetrics:
    """Utility class for calculating and managing model performance metrics."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_classification_metrics(self, y_true: np.ndarray, 
                                       y_pred: np.ndarray,
                                       y_proba: Optional[np.ndarray] = None,
                                       class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Calculate comprehensive classification metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
        
        # Handle multi-class vs binary
        n_classes = len(np.unique(y_true))
        
        if n_classes == 2:
            # Binary classification
            metrics['precision'] = float(precision_score(y_true, y_pred))
            metrics['recall'] = float(recall_score(y_true, y_pred))
            metrics['f1_score'] = float(f1_score(y_true, y_pred))
            
            if y_proba is not None:
                # ROC AUC for binary classification
                if y_proba.ndim == 2:
                    y_proba_positive = y_proba[:, 1]  # Probability of positive class
                else:
                    y_proba_positive = y_proba
                metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba_positive))
        else:
            # Multi-class classification
            metrics['precision_macro'] = float(precision_score(y_true, y_pred, average='macro'))
            metrics['precision_micro'] = float(precision_score(y_true, y_pred, average='micro'))
            metrics['precision_weighted'] = float(precision_score(y_true, y_pred, average='weighted'))
            
            metrics['recall_macro'] = float(recall_score(y_true, y_pred, average='macro'))
            metrics['recall_micro'] = float(recall_score(y_true, y_pred, average='micro'))
            metrics['recall_weighted'] = float(recall_score(y_true, y_pred, average='weighted'))
            
            metrics['f1_macro'] = float(f1_score(y_true, y_pred, average='macro'))
            metrics['f1_micro'] = float(f1_score(y_true, y_pred, average='micro'))
            metrics['f1_weighted'] = float(f1_score(y_true, y_pred, average='weighted'))
            
            if y_proba is not None:
                # ROC AUC for multi-class (one-vs-rest)
                try:
                    metrics['roc_auc_ovr'] = float(roc_auc_score(y_true, y_proba, 
                                                               multi_class='ovr', average='macro'))
                except Exception as e:
                    self.logger.warning(f"Could not calculate ROC AUC: {str(e)}")
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Classification report
        if class_names:
            target_names = class_names
        else:
            target_names = [f'Class_{i}' for i in range(n_classes)]
        
        class_report = classification_report(y_true, y_pred, 
                                           target_names=target_names, 
                                           output_dict=True)
        metrics['classification_report'] = class_report
        
        # Class distribution
        unique_labels, counts = np.unique(y_true, return_counts=True)
        metrics['class_distribution'] = dict(zip(unique_labels.tolist(), counts.tolist()))
        
        return metrics
    
    def calculate_model_comparison_metrics(self, models_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compare multiple models and provide ranking."""
        if not models_results:
            return {'error': 'No model results provided'}
        
        comparison_metrics = {}
        
        # Extract key metrics for comparison
        for model_name, results in models_results.items():
            if 'evaluation_metrics' in results:
                metrics = results['evaluation_metrics']
                comparison_metrics[model_name] = {
                    'accuracy': metrics.get('accuracy', 0),
                    'f1_score': metrics.get('f1_score', metrics.get('f1_macro', 0)),
                    'precision': metrics.get('precision', metrics.get('precision_macro', 0)),
                    'recall': metrics.get('recall', metrics.get('recall_macro', 0))
                }
        
        # Rank models by accuracy
        accuracy_ranking = sorted(comparison_metrics.items(), 
                                key=lambda x: x[1]['accuracy'], reverse=True)
        
        # Create comprehensive comparison
        comparison_summary = {
            'model_count': len(models_results),
            'accuracy_ranking': [(name, metrics['accuracy']) for name, metrics in accuracy_ranking],
            'best_model': accuracy_ranking[0][0] if accuracy_ranking else None,
            'metrics_comparison': comparison_metrics
        }
        
        return comparison_summary
    
    def calculate_cross_validation_metrics(self, cv_scores: Dict[str, List[float]]) -> Dict[str, Any]:
        """Calculate statistics for cross-validation results."""
        cv_summary = {}
        
        for metric_name, scores in cv_scores.items():
            scores_array = np.array(scores)
            cv_summary[metric_name] = {
                'mean': float(np.mean(scores_array)),
                'std': float(np.std(scores_array)),
                'min': float(np.min(scores_array)),
                'max': float(np.max(scores_array)),
                'scores': scores
            }
        
        return cv_summary
    
    def calculate_learning_curve_metrics(self, train_sizes: np.ndarray,
                                       train_scores: np.ndarray,
                                       validation_scores: np.ndarray) -> Dict[str, Any]:
        """Calculate learning curve statistics."""
        learning_curve_data = {
            'train_sizes': train_sizes.tolist(),
            'train_scores_mean': np.mean(train_scores, axis=1).tolist(),
            'train_scores_std': np.std(train_scores, axis=1).tolist(),
            'validation_scores_mean': np.mean(validation_scores, axis=1).tolist(),
            'validation_scores_std': np.std(validation_scores, axis=1).tolist(),
            'overfitting_analysis': self._analyze_overfitting(train_scores, validation_scores)
        }
        
        return learning_curve_data
    
    def _analyze_overfitting(self, train_scores: np.ndarray, 
                           validation_scores: np.ndarray) -> Dict[str, Any]:
        """Analyze overfitting from learning curves."""
        train_mean = np.mean(train_scores, axis=1)
        val_mean = np.mean(validation_scores, axis=1)
        
        # Calculate gap between training and validation scores
        gap = train_mean - val_mean
        
        analysis = {
            'max_gap': float(np.max(gap)),
            'final_gap': float(gap[-1]),
            'overfitting_severity': 'low' if gap[-1] < 0.05 else 'medium' if gap[-1] < 0.15 else 'high'
        }
        
        return analysis
    
    def generate_performance_summary(self, model_results: Dict[str, Any]) -> str:
        """Generate human-readable performance summary."""
        if 'evaluation_metrics' not in model_results:
            return "No evaluation metrics available"
        
        metrics = model_results['evaluation_metrics']
        accuracy = metrics.get('accuracy', 0)
        
        # Determine performance level
        if accuracy >= 0.9:
            performance_level = "Excellent"
        elif accuracy >= 0.8:
            performance_level = "Good"
        elif accuracy >= 0.7:
            performance_level = "Fair"
        else:
            performance_level = "Poor"
        
        summary = f"{performance_level} performance with {accuracy:.1%} accuracy"
        
        # Add F1 score if available
        f1 = metrics.get('f1_score', metrics.get('f1_macro'))
        if f1:
            summary += f" and {f1:.3f} F1-score"
        
        return summary


class ModelValidator:
    """Utility class for validating models and their outputs."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_predictions(self, predictions: np.ndarray,
                           expected_classes: Optional[List] = None) -> Dict[str, Any]:
        """Validate model predictions."""
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Check for NaN values
        if np.isnan(predictions).any():
            validation_results['is_valid'] = False
            validation_results['errors'].append("Predictions contain NaN values")
        
        # Check prediction distribution
        unique_preds, counts = np.unique(predictions, return_counts=True)
        pred_distribution = dict(zip(unique_preds, counts))
        
        # Check if all predictions are the same class
        if len(unique_preds) == 1:
            validation_results['warnings'].append("All predictions are the same class")
        
        # Validate against expected classes if provided
        if expected_classes:
            unexpected_classes = set(unique_preds) - set(expected_classes)
            if unexpected_classes:
                validation_results['is_valid'] = False
                validation_results['errors'].append(f"Unexpected prediction classes: {unexpected_classes}")
        
        validation_results['prediction_distribution'] = pred_distribution
        validation_results['unique_predictions'] = len(unique_preds)
        
        return validation_results
    
    def validate_probabilities(self, probabilities: np.ndarray) -> Dict[str, Any]:
        """Validate prediction probabilities."""
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Check shape
        if probabilities.ndim != 2:
            validation_results['is_valid'] = False
            validation_results['errors'].append("Probabilities must be 2-dimensional")
            return validation_results
        
        # Check for NaN values
        if np.isnan(probabilities).any():
            validation_results['is_valid'] = False
            validation_results['errors'].append("Probabilities contain NaN values")
        
        # Check probability constraints
        if (probabilities < 0).any() or (probabilities > 1).any():
            validation_results['is_valid'] = False
            validation_results['errors'].append("Probabilities must be between 0 and 1")
        
        # Check if rows sum to 1
        row_sums = np.sum(probabilities, axis=1)
        if not np.allclose(row_sums, 1.0, rtol=1e-5):
            validation_results['warnings'].append("Probability rows do not sum to 1")
        
        validation_results['shape'] = probabilities.shape
        validation_results['mean_confidence'] = float(np.mean(np.max(probabilities, axis=1)))
        
        return validation_results
    
    def validate_training_data(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Validate training data quality."""
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'data_quality_score': 1.0
        }
        
        quality_penalties = 0
        
        # Check for empty data
        if len(X) == 0 or len(y) == 0:
            validation_results['is_valid'] = False
            validation_results['errors'].append("Training data is empty")
            return validation_results
        
        # Check dimension mismatch
        if len(X) != len(y):
            validation_results['is_valid'] = False
            validation_results['errors'].append("Feature and target dimension mismatch")
            return validation_results
        
        # Check missing values
        missing_features = X.isnull().sum()
        missing_ratio = missing_features.sum() / (len(X) * len(X.columns))
        
        if missing_ratio > 0.5:
            validation_results['warnings'].append(f"High missing value ratio: {missing_ratio:.2%}")
            quality_penalties += 0.3
        elif missing_ratio > 0.2:
            validation_results['warnings'].append(f"Moderate missing value ratio: {missing_ratio:.2%}")
            quality_penalties += 0.1
        
        # Check class imbalance
        class_counts = y.value_counts()
        class_ratio = class_counts.max() / class_counts.min() if len(class_counts) > 1 else 1
        
        if class_ratio > 10:
            validation_results['warnings'].append(f"Severe class imbalance (ratio: {class_ratio:.1f})")
            quality_penalties += 0.2
        elif class_ratio > 5:
            validation_results['warnings'].append(f"Moderate class imbalance (ratio: {class_ratio:.1f})")
            quality_penalties += 0.1
        
        # Check feature variance
        numeric_features = X.select_dtypes(include=[np.number])
        low_variance_features = []
        
        for col in numeric_features.columns:
            if numeric_features[col].var() < 1e-6:
                low_variance_features.append(col)
        
        if low_variance_features:
            validation_results['warnings'].append(f"Low variance features: {low_variance_features}")
            quality_penalties += 0.1
        
        # Calculate final quality score
        validation_results['data_quality_score'] = max(0, 1.0 - quality_penalties)
        
        # Add statistics
        validation_results['statistics'] = {
            'n_samples': len(X),
            'n_features': len(X.columns),
            'n_classes': len(class_counts),
            'missing_ratio': missing_ratio,
            'class_distribution': class_counts.to_dict(),
            'class_imbalance_ratio': class_ratio
        }
        
        return validation_results