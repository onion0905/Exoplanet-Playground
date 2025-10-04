from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
import logging

from ..explainability.feature_importance import FeatureImportanceAnalyzer
from ..explainability.column_dropper import ColumnDropper
from ..api.prediction_api import PredictionAPI


class ExplanationAPI:
    """API interface for explaining model predictions and feature importance."""
    
    def __init__(self, prediction_api=None):
        self.feature_analyzer = FeatureImportanceAnalyzer()
        self.column_dropper = ColumnDropper()
        self.prediction_api = prediction_api or PredictionAPI()
        self.logger = logging.getLogger(__name__)
        
        # Cache for explanation results
        self.explanation_cache = {}
    
    def explain_model_global(self, model_id: str,
                           X_train: pd.DataFrame, y_train: pd.Series,
                           X_test: pd.DataFrame, y_test: pd.Series,
                           methods: List[str] = None) -> Dict[str, Any]:
        """
        Generate global explanations for a model.
        
        Args:
            model_id: ID of the loaded model
            X_train, y_train: Training data
            X_test, y_test: Test data
            methods: List of explanation methods to use
        """
        try:
            # Get model from prediction API
            if model_id not in self.prediction_api.loaded_models:
                raise ValueError(f"Model {model_id} not loaded")
            
            model = self.prediction_api.loaded_models[model_id]['model']
            
            if methods is None:
                methods = ['model_importance', 'column_drop', 'average_replacement']
            
            # Generate comprehensive feature analysis
            explanation_results = self.feature_analyzer.comprehensive_feature_analysis(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                methods=methods
            )
            
            # Get top features
            top_features = self.feature_analyzer.get_top_features(explanation_results, n_features=10)
            
            # Cache results
            cache_key = f"{model_id}_global"
            self.explanation_cache[cache_key] = explanation_results
            
            return {
                'model_id': model_id,
                'status': 'success',
                'explanation_type': 'global',
                'results': explanation_results,
                'top_features': top_features,
                'methods_used': methods
            }
            
        except Exception as e:
            self.logger.error(f"Error generating global explanation for model {model_id}: {str(e)}")
            return {
                'model_id': model_id,
                'status': 'error',
                'error': str(e)
            }
    
    def explain_prediction_local(self, model_id: str,
                               instance_data: Dict[str, Any],
                               X_train: pd.DataFrame = None,
                               y_train: pd.Series = None) -> Dict[str, Any]:
        """
        Generate local explanation for a single prediction.
        
        Args:
            model_id: ID of the loaded model
            instance_data: Single instance to explain
            X_train, y_train: Training data for reference (optional)
        """
        try:
            if model_id not in self.prediction_api.loaded_models:
                raise ValueError(f"Model {model_id} not loaded")
            
            model = self.prediction_api.loaded_models[model_id]['model']
            
            # Make prediction first
            prediction_result = self.prediction_api.predict_single(model_id, instance_data)
            
            if prediction_result['status'] != 'success':
                return prediction_result
            
            # Create DataFrame for the instance
            instance_df = pd.DataFrame([instance_data])
            
            # Feature importance for this instance (using average replacement method)
            dummy_y = pd.Series([prediction_result['prediction']])
            local_importance = self.feature_analyzer.average_column_replacement_importance(
                model, instance_df, dummy_y
            )
            
            # Get feature contributions
            feature_contributions = {}
            for feature, importance_data in local_importance.items():
                if importance_data.get('accuracy_drop') is not None:
                    feature_contributions[feature] = {
                        'value': instance_data.get(feature, 'N/A'),
                        'importance_score': importance_data['accuracy_drop'],
                        'relative_importance': importance_data['relative_drop_percent']
                    }
            
            # Sort features by importance
            sorted_contributions = sorted(
                feature_contributions.items(),
                key=lambda x: x[1]['importance_score'],
                reverse=True
            )
            
            return {
                'model_id': model_id,
                'status': 'success',
                'explanation_type': 'local',
                'instance_data': instance_data,
                'prediction': prediction_result['prediction'],
                'probabilities': prediction_result['probabilities'],
                'feature_contributions': dict(sorted_contributions),
                'top_contributing_features': sorted_contributions[:5]
            }
            
        except Exception as e:
            self.logger.error(f"Error generating local explanation: {str(e)}")
            return {
                'model_id': model_id,
                'status': 'error',
                'error': str(e)
            }
    
    def analyze_feature_importance_drop(self, model_id: str,
                                      X_train: pd.DataFrame, y_train: pd.Series,
                                      X_test: pd.DataFrame, y_test: pd.Series,
                                      features_to_analyze: List[str] = None) -> Dict[str, Any]:
        """
        Analyze feature importance using the column dropping method specifically.
        This implements the user's requested functionality.
        """
        try:
            if model_id not in self.prediction_api.loaded_models:
                raise ValueError(f"Model {model_id} not loaded")
            
            model = self.prediction_api.loaded_models[model_id]['model']
            
            # Limit analysis to specific features if provided
            if features_to_analyze:
                available_features = [f for f in features_to_analyze if f in X_train.columns]
                X_train_subset = X_train[available_features]
                X_test_subset = X_test[available_features]
            else:
                X_train_subset = X_train
                X_test_subset = X_test
            
            # Perform column drop analysis
            drop_results = self.feature_analyzer.column_drop_importance(
                model=model,
                X_train=X_train_subset,
                y_train=y_train,
                X_test=X_test_subset,
                y_test=y_test
            )
            
            # Create summary
            valid_results = {k: v for k, v in drop_results.items() 
                           if v.get('accuracy_drop') is not None}
            
            # Sort by importance
            sorted_features = sorted(
                valid_results.items(),
                key=lambda x: x[1]['accuracy_drop'],
                reverse=True
            )
            
            return {
                'model_id': model_id,
                'status': 'success',
                'method': 'column_drop_importance',
                'baseline_accuracy': list(valid_results.values())[0]['baseline_accuracy'] if valid_results else None,
                'feature_importance_ranking': sorted_features,
                'detailed_results': drop_results,
                'features_analyzed': len(drop_results)
            }
            
        except Exception as e:
            self.logger.error(f"Error in feature importance drop analysis: {str(e)}")
            return {
                'model_id': model_id,
                'status': 'error',
                'error': str(e)
            }
    
    def get_column_selection_info(self, data: pd.DataFrame,
                                 importance_scores: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Get information for interactive column selection.
        """
        try:
            column_info = self.column_dropper.interactive_column_selection(
                data=data,
                importance_scores=importance_scores
            )
            
            return {
                'status': 'success',
                'column_info': column_info
            }
            
        except Exception as e:
            self.logger.error(f"Error getting column selection info: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def apply_column_selection(self, data: pd.DataFrame,
                             selected_columns: List[str]) -> Dict[str, Any]:
        """
        Apply user-selected column filtering.
        """
        try:
            filtered_data = self.column_dropper.apply_column_selection(
                data=data,
                selected_columns=selected_columns
            )
            
            drop_summary = self.column_dropper.get_column_drop_summary()
            
            return {
                'status': 'success',
                'filtered_data_shape': filtered_data.shape,
                'drop_summary': drop_summary
            }
            
        except Exception as e:
            self.logger.error(f"Error applying column selection: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def explain_model_decision_path(self, model_id: str,
                                  instance_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Explain the decision path for tree-based models.
        """
        try:
            if model_id not in self.prediction_api.loaded_models:
                raise ValueError(f"Model {model_id} not loaded")
            
            model = self.prediction_api.loaded_models[model_id]['model']
            
            # Check if model supports decision path explanation
            if not hasattr(model, 'get_tree_rules'):
                return {
                    'model_id': model_id,
                    'status': 'not_supported',
                    'message': 'Decision path explanation only available for tree-based models'
                }
            
            # Get tree rules
            tree_rules = model.get_tree_rules()
            
            # Make prediction
            prediction_result = self.prediction_api.predict_single(model_id, instance_data)
            
            return {
                'model_id': model_id,
                'status': 'success',
                'explanation_type': 'decision_path',
                'instance_data': instance_data,
                'prediction': prediction_result['prediction'],
                'tree_rules': tree_rules,
                'model_type': model.model_name
            }
            
        except Exception as e:
            self.logger.error(f"Error explaining decision path: {str(e)}")
            return {
                'model_id': model_id,
                'status': 'error',
                'error': str(e)
            }
    
    def compare_feature_importance_methods(self, model_id: str,
                                         X_train: pd.DataFrame, y_train: pd.Series,
                                         X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Compare different feature importance methods for the same model.
        """
        try:
            if model_id not in self.prediction_api.loaded_models:
                raise ValueError(f"Model {model_id} not loaded")
            
            model = self.prediction_api.loaded_models[model_id]['model']
            
            # Get importance from different methods
            methods = {
                'model_importance': self.feature_analyzer.get_model_feature_importance(model),
                'column_drop': self.feature_analyzer.column_drop_importance(
                    model, X_train, y_train, X_test, y_test
                ),
                'average_replacement': self.feature_analyzer.average_column_replacement_importance(
                    model, X_test, y_test
                )
            }
            
            # Create comparison table
            features = list(X_train.columns)
            comparison_data = {}
            
            for feature in features:
                comparison_data[feature] = {}
                
                # Model importance
                if feature in methods['model_importance']:
                    comparison_data[feature]['model_importance'] = methods['model_importance'][feature]
                
                # Column drop importance
                if feature in methods['column_drop']:
                    comparison_data[feature]['column_drop'] = methods['column_drop'][feature].get('relative_drop_percent', 0)
                
                # Average replacement importance  
                if feature in methods['average_replacement']:
                    comparison_data[feature]['average_replacement'] = methods['average_replacement'][feature].get('relative_drop_percent', 0)
            
            return {
                'model_id': model_id,
                'status': 'success',
                'comparison_methods': list(methods.keys()),
                'feature_comparison': comparison_data,
                'detailed_results': methods
            }
            
        except Exception as e:
            self.logger.error(f"Error comparing feature importance methods: {str(e)}")
            return {
                'model_id': model_id,
                'status': 'error',
                'error': str(e)
            }