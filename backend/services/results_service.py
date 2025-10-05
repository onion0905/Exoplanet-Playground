"""
Results processing service for ML predictions and analysis
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import logging
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


class ResultsService:
    """Service for processing ML results, predictions, and generating analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_testing_results(self, model, X_test: pd.DataFrame, y_test: pd.Series, 
                               target_classes: List[str], feature_columns: List[str]) -> Dict[str, Any]:
        """
        Generate comprehensive testing results including predictions and confusion matrix
        
        Args:
            model: Trained ML model
            X_test: Test features
            y_test: Test labels
            target_classes: List of target class names
            feature_columns: List of feature column names
            
        Returns:
            Dictionary with comprehensive results
        """
        try:
            # Make predictions
            predictions = model.predict(X_test)
            
            # Get prediction probabilities if available
            probabilities = None
            if hasattr(model, 'predict_proba'):
                try:
                    probabilities = model.predict_proba(X_test)
                except:
                    self.logger.warning("Could not get prediction probabilities")
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, predictions)
            
            # Handle binary and multiclass scenarios (now supporting 3-class prediction)
            average_method = 'binary' if len(target_classes) == 2 else 'weighted'
            
            try:
                precision = precision_score(y_test, predictions, average=average_method, zero_division=0)
                recall = recall_score(y_test, predictions, average=average_method, zero_division=0)
                f1 = f1_score(y_test, predictions, average=average_method, zero_division=0)
            except:
                precision = recall = f1 = 0.0
            
            # ROC AUC for binary classification
            roc_auc = None
            if len(target_classes) == 2 and probabilities is not None:
                try:
                    roc_auc = roc_auc_score(y_test, probabilities[:, 1])
                except:
                    self.logger.warning("Could not calculate ROC AUC")
            
            # Generate confusion matrix
            cm = confusion_matrix(y_test, predictions, labels=target_classes)
            confusion_matrix_dict = self._format_confusion_matrix(cm, target_classes)
            
            # Generate classification report
            try:
                class_report = classification_report(y_test, predictions, 
                                                   target_names=target_classes, 
                                                   output_dict=True, 
                                                   zero_division=0)
            except:
                class_report = {}
            
            # Create predictions table
            predictions_table = self._create_predictions_table(
                X_test, y_test, predictions, probabilities, feature_columns, target_classes
            )
            
            # Calculate per-class metrics
            per_class_metrics = self._calculate_per_class_metrics(y_test, predictions, target_classes)
            
            # Feature importance if available
            feature_importance = self._get_feature_importance(model, feature_columns)
            
            results = {
                'summary_metrics': {
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1),
                    'roc_auc': float(roc_auc) if roc_auc is not None else None,
                    'total_samples': int(len(y_test)),
                    'correct_predictions': int(np.sum(predictions == y_test)),
                    'incorrect_predictions': int(np.sum(predictions != y_test))
                },
                'confusion_matrix': confusion_matrix_dict,
                'classification_report': class_report,
                'per_class_metrics': per_class_metrics,
                'predictions_table': predictions_table,
                'feature_importance': feature_importance,
                'target_classes': target_classes,
                'feature_columns': feature_columns
            }
            
            # Apply comprehensive numpy type conversion for JSON serialization
            results = convert_numpy_types(results)
            
            self.logger.info(f"Generated testing results with accuracy: {accuracy:.4f}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error generating testing results: {e}")
            raise
    
    def _format_confusion_matrix(self, cm: np.ndarray, target_classes: List[str]) -> Dict[str, Any]:
        """Format confusion matrix for frontend consumption"""
        cm_dict = {
            'matrix': [[int(val) for val in row] for row in cm.tolist()],
            'labels': target_classes,
            'matrix_with_labels': []
        }
        
        # Create labeled matrix for easier interpretation
        for i, true_label in enumerate(target_classes):
            row = {
                'true_label': str(true_label),
                'predictions': {}
            }
            for j, pred_label in enumerate(target_classes):
                row['predictions'][str(pred_label)] = int(cm[i, j])
            cm_dict['matrix_with_labels'].append(row)
        
        return cm_dict
    
    def _create_predictions_table(self, X_test: pd.DataFrame, y_test: pd.Series, 
                                predictions: np.ndarray, probabilities: np.ndarray,
                                feature_columns: List[str], target_classes: List[str]) -> List[Dict]:
        """Create detailed predictions table"""
        predictions_table = []
        
        # Limit to first 100 predictions for performance
        max_rows = min(100, len(X_test))
        
        for i in range(max_rows):
            row = {
                'index': int(i),
                'true_label': str(y_test.iloc[i]),
                'predicted_label': str(predictions[i]),
                'correct': bool(y_test.iloc[i] == predictions[i]),
                'features': {str(col): float(X_test.iloc[i][col]) for col in feature_columns[:10]},  # Limit features for performance
                'probabilities': {}
            }
            
            # Add probabilities if available
            if probabilities is not None:
                for j, class_name in enumerate(target_classes):
                    row['probabilities'][str(class_name)] = float(probabilities[i, j])
                row['confidence'] = float(np.max(probabilities[i]))
            else:
                row['confidence'] = 1.0 if row['correct'] else 0.0
            
            predictions_table.append(row)
        
        return predictions_table
    
    def _calculate_per_class_metrics(self, y_test: pd.Series, predictions: np.ndarray, 
                                   target_classes: List[str]) -> Dict[str, Dict]:
        """Calculate metrics for each class"""
        per_class = {}
        
        for class_name in target_classes:
            # Create binary classification for this class
            y_binary = (y_test == class_name).astype(int)
            pred_binary = (predictions == class_name).astype(int)
            
            try:
                precision = precision_score(y_binary, pred_binary, zero_division=0)
                recall = recall_score(y_binary, pred_binary, zero_division=0)
                f1 = f1_score(y_binary, pred_binary, zero_division=0)
                
                # Count statistics
                true_positives = int(np.sum((y_binary == 1) & (pred_binary == 1)))
                false_positives = int(np.sum((y_binary == 0) & (pred_binary == 1)))
                false_negatives = int(np.sum((y_binary == 1) & (pred_binary == 0)))
                true_negatives = int(np.sum((y_binary == 0) & (pred_binary == 0)))
                
                per_class[class_name] = {
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1),
                    'support': int(np.sum(y_test == class_name)),
                    'true_positives': true_positives,
                    'false_positives': false_positives,
                    'false_negatives': false_negatives,
                    'true_negatives': true_negatives
                }
            except Exception as e:
                self.logger.warning(f"Could not calculate metrics for class {class_name}: {e}")
                per_class[class_name] = {
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'support': int(np.sum(y_test == class_name)),
                    'true_positives': 0,
                    'false_positives': 0,
                    'false_negatives': 0,
                    'true_negatives': 0
                }
        
        return per_class
    
    def _get_feature_importance(self, model, feature_columns: List[str]) -> Dict[str, Any]:
        """Extract feature importance if available"""
        try:
            importance = None
            importance_type = None
            
            # Try different methods to get feature importance
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                importance_type = 'gini_importance'
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_).flatten()
                importance_type = 'coefficient_magnitude'
            elif hasattr(model, 'get_feature_importance'):
                importance = model.get_feature_importance()
                importance_type = 'model_specific'
            
            if importance is not None:
                # Create sorted feature importance
                feature_importance_pairs = list(zip(feature_columns, importance))
                feature_importance_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
                
                return {
                    'available': True,
                    'type': importance_type,
                    'features': [
                        {
                            'feature': feature,
                            'importance': float(imp),
                            'rank': rank + 1
                        }
                        for rank, (feature, imp) in enumerate(feature_importance_pairs)
                    ],
                    'top_5_features': [
                        {
                            'feature': feature,
                            'importance': float(imp)
                        }
                        for feature, imp in feature_importance_pairs[:5]
                    ]
                }
            
        except Exception as e:
            self.logger.warning(f"Could not extract feature importance: {e}")
        
        return {'available': False, 'message': 'Feature importance not available for this model'}
    
    def make_single_prediction(self, model, input_features: Dict[str, float], 
                             feature_columns: List[str], target_classes: List[str]) -> Dict[str, Any]:
        """Make a single prediction with confidence"""
        try:
            # Prepare input data
            input_array = np.array([[input_features.get(col, 0.0) for col in feature_columns]])
            
            # Make prediction
            prediction = model.predict(input_array)[0]
            
            # Get probabilities if available
            probabilities = {}
            confidence = 0.0
            
            if hasattr(model, 'predict_proba'):
                try:
                    proba = model.predict_proba(input_array)[0]
                    for i, class_name in enumerate(target_classes):
                        probabilities[class_name] = float(proba[i])
                    confidence = float(np.max(proba))
                except:
                    self.logger.warning("Could not get prediction probabilities")
            
            return {
                'prediction': str(prediction),
                'confidence': confidence,
                'probabilities': probabilities,
                'input_features': input_features,
                'feature_columns': feature_columns
            }
            
        except Exception as e:
            self.logger.error(f"Error making single prediction: {e}")
            raise


# Global results service instance
results_service = ResultsService()