from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import logging


class FeatureImportanceAnalyzer:
    """Analyzes feature importance using various methods including column dropping."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def get_model_feature_importance(self, model) -> Dict[str, float]:
        """Get feature importance directly from the model."""
        try:
            return model.get_feature_importance()
        except Exception as e:
            self.logger.warning(f"Could not get model feature importance: {str(e)}")
            return {}
    
    def permutation_importance(self, model, X: pd.DataFrame, y: pd.Series, 
                             metric_func=None, n_repeats: int = 5,
                             random_state: int = 42) -> Dict[str, Dict[str, float]]:
        """Calculate permutation importance for each feature."""
        if metric_func is None:
            metric_func = accuracy_score
        
        # Get baseline score
        baseline_predictions = model.predict(X)
        baseline_score = metric_func(y, baseline_predictions)
        
        importance_scores = {}
        np.random.seed(random_state)
        
        for feature in X.columns:
            feature_scores = []
            
            for _ in range(n_repeats):
                # Create a copy of X and shuffle the feature
                X_permuted = X.copy()
                X_permuted[feature] = np.random.permutation(X_permuted[feature].values)
                
                # Get predictions and score
                permuted_predictions = model.predict(X_permuted)
                permuted_score = metric_func(y, permuted_predictions)
                
                # Calculate importance as decrease in score
                importance = baseline_score - permuted_score
                feature_scores.append(importance)
            
            importance_scores[feature] = {
                'importance_mean': np.mean(feature_scores),
                'importance_std': np.std(feature_scores),
                'importance_scores': feature_scores
            }
        
        return importance_scores
    
    def column_drop_importance(self, model, X_train: pd.DataFrame, y_train: pd.Series,
                              X_test: pd.DataFrame, y_test: pd.Series,
                              metric_func=None) -> Dict[str, Dict[str, float]]:
        """
        Analyze feature importance by dropping each column and measuring accuracy change.
        This implements the method mentioned in the user's request.
        """
        if metric_func is None:
            metric_func = accuracy_score
        
        # Get baseline accuracy
        baseline_predictions = model.predict(X_test)
        baseline_accuracy = metric_func(y_test, baseline_predictions)
        
        importance_results = {}
        
        for feature in X_train.columns:
            self.logger.info(f"Analyzing importance of feature: {feature}")
            
            # Create datasets without this feature
            X_train_dropped = X_train.drop(columns=[feature])
            X_test_dropped = X_test.drop(columns=[feature])
            
            try:
                # Create and train a new model without this feature
                # We need to create a new instance of the same model type
                model_class = type(model)
                dropped_model = model_class()
                
                # Copy hyperparameters if they exist
                if hasattr(model, 'hyperparameters'):
                    dropped_model.build_model(**model.hyperparameters)
                else:
                    dropped_model.build_model()
                
                # Train the model without this feature
                dropped_model.train(X_train_dropped, y_train)
                
                # Test the model
                dropped_predictions = dropped_model.predict(X_test_dropped)
                dropped_accuracy = metric_func(y_test, dropped_predictions)
                
                # Calculate importance as accuracy drop
                accuracy_drop = baseline_accuracy - dropped_accuracy
                relative_drop = (accuracy_drop / baseline_accuracy) * 100 if baseline_accuracy > 0 else 0
                
                importance_results[feature] = {
                    'baseline_accuracy': baseline_accuracy,
                    'dropped_accuracy': dropped_accuracy,
                    'accuracy_drop': accuracy_drop,
                    'relative_drop_percent': relative_drop
                }
                
                self.logger.info(f"Feature {feature}: {accuracy_drop:.4f} accuracy drop ({relative_drop:.2f}%)")
                
            except Exception as e:
                self.logger.error(f"Error analyzing feature {feature}: {str(e)}")
                importance_results[feature] = {
                    'baseline_accuracy': baseline_accuracy,
                    'dropped_accuracy': None,
                    'accuracy_drop': None,
                    'relative_drop_percent': None,
                    'error': str(e)
                }
        
        return importance_results
    
    def average_column_replacement_importance(self, model, X: pd.DataFrame, y: pd.Series,
                                           metric_func=None) -> Dict[str, Dict[str, float]]:
        """
        Analyze feature importance by replacing each column with its average value.
        This is another interpretation of the user's request.
        """
        if metric_func is None:
            metric_func = accuracy_score
        
        # Get baseline accuracy
        baseline_predictions = model.predict(X)
        baseline_accuracy = metric_func(y, baseline_predictions)
        
        importance_results = {}
        
        for feature in X.columns:
            try:
                # Create dataset with this feature set to its mean/mode
                X_averaged = X.copy()
                
                if X[feature].dtype in ['float64', 'float32', 'int64', 'int32']:
                    # For numeric features, use mean
                    X_averaged[feature] = X[feature].mean()
                else:
                    # For categorical features, use mode
                    X_averaged[feature] = X[feature].mode().iloc[0] if len(X[feature].mode()) > 0 else X[feature].iloc[0]
                
                # Test the model with averaged feature
                averaged_predictions = model.predict(X_averaged)
                averaged_accuracy = metric_func(y, averaged_predictions)
                
                # Calculate importance as accuracy drop
                accuracy_drop = baseline_accuracy - averaged_accuracy
                relative_drop = (accuracy_drop / baseline_accuracy) * 100 if baseline_accuracy > 0 else 0
                
                importance_results[feature] = {
                    'baseline_accuracy': baseline_accuracy,
                    'averaged_accuracy': averaged_accuracy,
                    'accuracy_drop': accuracy_drop,
                    'relative_drop_percent': relative_drop,
                    'replacement_value': X_averaged[feature].iloc[0]
                }
                
            except Exception as e:
                self.logger.error(f"Error analyzing feature {feature}: {str(e)}")
                importance_results[feature] = {
                    'baseline_accuracy': baseline_accuracy,
                    'averaged_accuracy': None,
                    'accuracy_drop': None,
                    'relative_drop_percent': None,
                    'error': str(e)
                }
        
        return importance_results
    
    def comprehensive_feature_analysis(self, model, X_train: pd.DataFrame, y_train: pd.Series,
                                     X_test: pd.DataFrame, y_test: pd.Series,
                                     methods: List[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive feature importance analysis using multiple methods.
        """
        if methods is None:
            methods = ['model_importance', 'column_drop', 'average_replacement', 'permutation']
        
        results = {}
        
        if 'model_importance' in methods:
            self.logger.info("Calculating model-based feature importance...")
            results['model_importance'] = self.get_model_feature_importance(model)
        
        if 'column_drop' in methods:
            self.logger.info("Calculating column drop importance...")
            results['column_drop_importance'] = self.column_drop_importance(
                model, X_train, y_train, X_test, y_test
            )
        
        if 'average_replacement' in methods:
            self.logger.info("Calculating average replacement importance...")
            results['average_replacement_importance'] = self.average_column_replacement_importance(
                model, X_test, y_test
            )
        
        if 'permutation' in methods:
            self.logger.info("Calculating permutation importance...")
            results['permutation_importance'] = self.permutation_importance(
                model, X_test, y_test
            )
        
        # Create a summary ranking
        results['feature_ranking'] = self._create_feature_ranking(results)
        
        return results
    
    def _create_feature_ranking(self, importance_results: Dict[str, Any]) -> Dict[str, float]:
        """Create a unified feature ranking from multiple importance methods."""
        feature_scores = {}
        
        # Collect scores from different methods
        if 'model_importance' in importance_results:
            model_imp = importance_results['model_importance']
            for feature, score in model_imp.items():
                if feature not in feature_scores:
                    feature_scores[feature] = []
                feature_scores[feature].append(score)
        
        if 'column_drop_importance' in importance_results:
            drop_imp = importance_results['column_drop_importance']
            for feature, data in drop_imp.items():
                if data.get('relative_drop_percent') is not None:
                    if feature not in feature_scores:
                        feature_scores[feature] = []
                    feature_scores[feature].append(data['relative_drop_percent'] / 100)
        
        if 'average_replacement_importance' in importance_results:
            avg_imp = importance_results['average_replacement_importance']
            for feature, data in avg_imp.items():
                if data.get('relative_drop_percent') is not None:
                    if feature not in feature_scores:
                        feature_scores[feature] = []
                    feature_scores[feature].append(data['relative_drop_percent'] / 100)
        
        if 'permutation_importance' in importance_results:
            perm_imp = importance_results['permutation_importance']
            for feature, data in perm_imp.items():
                if data.get('importance_mean') is not None:
                    if feature not in feature_scores:
                        feature_scores[feature] = []
                    feature_scores[feature].append(data['importance_mean'])
        
        # Calculate average ranking for each feature
        unified_ranking = {}
        for feature, scores in feature_scores.items():
            if scores:
                unified_ranking[feature] = np.mean(scores)
            else:
                unified_ranking[feature] = 0.0
        
        # Sort by importance (descending)
        sorted_features = sorted(unified_ranking.items(), key=lambda x: x[1], reverse=True)
        
        return dict(sorted_features)
    
    def get_top_features(self, importance_results: Dict[str, Any], 
                        n_features: int = 10) -> List[Tuple[str, float]]:
        """Get the top N most important features."""
        ranking = importance_results.get('feature_ranking', {})
        sorted_features = sorted(ranking.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:n_features]