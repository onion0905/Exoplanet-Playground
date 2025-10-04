from typing import Dict, Any, Optional, List
import logging

from ..models.linear_regression import LinearRegressionModel
from ..models.svm import SVMModel
from ..models.decision_tree import DecisionTreeModel
from ..models.random_forest import RandomForestModel
from ..models.xgboost_model import XGBoostModel
from ..models.pca import PCAModel
from ..models.deep_learning import DeepLearningModel


class ModelFactory:
    """Factory class for creating ML models."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Registry of available models
        self.model_registry = {
            'linear_regression': LinearRegressionModel,
            'logistic_regression': LinearRegressionModel,  # Alias
            'svm': SVMModel,
            'support_vector_machine': SVMModel,  # Alias
            'decision_tree': DecisionTreeModel,
            'tree': DecisionTreeModel,  # Alias
            'random_forest': RandomForestModel,
            'rf': RandomForestModel,  # Alias
            'xgboost': XGBoostModel,
            'xgb': XGBoostModel,  # Alias
            'pca': PCAModel,
            'principal_component_analysis': PCAModel,  # Alias
            'deep_learning': DeepLearningModel,
            'neural_network': DeepLearningModel,  # Alias
            'nn': DeepLearningModel  # Alias
        }
    
    def create_model(self, model_type: str, **kwargs):
        """
        Create a model instance by type.
        
        Args:
            model_type: String identifier for the model type
            **kwargs: Additional arguments passed to model constructor
            
        Returns:
            Model instance
        """
        model_type_lower = model_type.lower().replace(' ', '_')
        
        if model_type_lower not in self.model_registry:
            available_types = list(self.model_registry.keys())
            raise ValueError(f"Unknown model type: {model_type}. Available types: {available_types}")
        
        model_class = self.model_registry[model_type_lower]
        
        try:
            model_instance = model_class(**kwargs)
            self.logger.info(f"Created model: {model_type} ({model_instance.model_name})")
            return model_instance
            
        except Exception as e:
            self.logger.error(f"Error creating model {model_type}: {str(e)}")
            raise
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all available models."""
        models_info = {}
        
        # Group aliases together
        primary_models = {
            'linear_regression': LinearRegressionModel,
            'svm': SVMModel,
            'decision_tree': DecisionTreeModel,
            'random_forest': RandomForestModel,
            'xgboost': XGBoostModel,
            'pca': PCAModel,
            'deep_learning': DeepLearningModel
        }
        
        for model_type, model_class in primary_models.items():
            # Get aliases for this model
            aliases = [k for k, v in self.model_registry.items() if v == model_class and k != model_type]
            
            models_info[model_type] = {
                'class_name': model_class.__name__,
                'aliases': aliases,
                'description': self._get_model_description(model_type),
                'hyperparameters': self._get_model_hyperparameters(model_type)
            }
        
        return models_info
    
    def _get_model_description(self, model_type: str) -> str:
        """Get description for a model type."""
        descriptions = {
            'linear_regression': 'Logistic Regression for classification with linear decision boundaries',
            'svm': 'Support Vector Machine with kernel-based classification',
            'decision_tree': 'Decision Tree classifier with interpretable rules',
            'random_forest': 'Random Forest ensemble of decision trees',
            'xgboost': 'Gradient boosting with XGBoost library',
            'pca': 'Principal Component Analysis with Logistic Regression',
            'deep_learning': 'Deep Neural Network with configurable architecture'
        }
        return descriptions.get(model_type, 'No description available')
    
    def _get_model_hyperparameters(self, model_type: str) -> Dict[str, Dict[str, Any]]:
        """Get hyperparameter information for a model type."""
        hyperparameters = {
            'linear_regression': {
                'C': {'type': 'float', 'default': 1.0, 'description': 'Regularization strength'},
                'max_iter': {'type': 'int', 'default': 1000, 'description': 'Maximum iterations'},
                'solver': {'type': 'choice', 'choices': ['lbfgs', 'liblinear', 'newton-cg'], 'default': 'lbfgs'}
            },
            'svm': {
                'C': {'type': 'float', 'default': 1.0, 'description': 'Regularization parameter'},
                'kernel': {'type': 'choice', 'choices': ['rbf', 'linear', 'poly', 'sigmoid'], 'default': 'rbf'},
                'gamma': {'type': 'choice', 'choices': ['scale', 'auto'], 'default': 'scale'}
            },
            'decision_tree': {
                'max_depth': {'type': 'int', 'default': None, 'description': 'Maximum tree depth'},
                'min_samples_split': {'type': 'int', 'default': 2, 'description': 'Min samples to split'},
                'min_samples_leaf': {'type': 'int', 'default': 1, 'description': 'Min samples in leaf'},
                'criterion': {'type': 'choice', 'choices': ['gini', 'entropy'], 'default': 'gini'}
            },
            'random_forest': {
                'n_estimators': {'type': 'int', 'default': 100, 'description': 'Number of trees'},
                'max_depth': {'type': 'int', 'default': None, 'description': 'Maximum tree depth'},
                'min_samples_split': {'type': 'int', 'default': 2, 'description': 'Min samples to split'},
                'criterion': {'type': 'choice', 'choices': ['gini', 'entropy'], 'default': 'gini'}
            },
            'xgboost': {
                'n_estimators': {'type': 'int', 'default': 100, 'description': 'Number of boosting rounds'},
                'max_depth': {'type': 'int', 'default': 6, 'description': 'Maximum tree depth'},
                'learning_rate': {'type': 'float', 'default': 0.1, 'description': 'Boosting learning rate'},
                'subsample': {'type': 'float', 'default': 1.0, 'description': 'Subsample ratio'}
            },
            'pca': {
                'n_components': {'type': 'float', 'default': 0.95, 'description': 'Variance to retain'},
                'C': {'type': 'float', 'default': 1.0, 'description': 'Classifier regularization'},
                'solver': {'type': 'choice', 'choices': ['lbfgs', 'liblinear'], 'default': 'lbfgs'}
            },
            'deep_learning': {
                'hidden_layers': {'type': 'list', 'default': [128, 64, 32], 'description': 'Hidden layer sizes'},
                'dropout_rate': {'type': 'float', 'default': 0.3, 'description': 'Dropout rate'},
                'learning_rate': {'type': 'float', 'default': 0.001, 'description': 'Learning rate'},
                'activation': {'type': 'choice', 'choices': ['relu', 'tanh', 'sigmoid'], 'default': 'relu'}
            }
        }
        return hyperparameters.get(model_type, {})
    
    def validate_hyperparameters(self, model_type: str, 
                               hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate hyperparameters for a model type."""
        model_type_lower = model_type.lower().replace(' ', '_')
        
        if model_type_lower not in self.model_registry:
            raise ValueError(f"Unknown model type: {model_type}")
        
        valid_params = self._get_model_hyperparameters(model_type_lower)
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'cleaned_params': {}
        }
        
        for param_name, param_value in hyperparameters.items():
            if param_name not in valid_params:
                validation_results['warnings'].append(f"Unknown parameter: {param_name}")
                continue
            
            param_spec = valid_params[param_name]
            param_type = param_spec['type']
            
            try:
                # Type validation and conversion
                if param_type == 'int':
                    cleaned_value = int(param_value)
                elif param_type == 'float':
                    cleaned_value = float(param_value)
                elif param_type == 'choice':
                    if param_value not in param_spec['choices']:
                        raise ValueError(f"Invalid choice. Must be one of: {param_spec['choices']}")
                    cleaned_value = param_value
                elif param_type == 'list':
                    if not isinstance(param_value, list):
                        raise ValueError("Parameter must be a list")
                    cleaned_value = param_value
                else:
                    cleaned_value = param_value
                
                validation_results['cleaned_params'][param_name] = cleaned_value
                
            except (ValueError, TypeError) as e:
                validation_results['valid'] = False
                validation_results['errors'].append(f"Invalid {param_name}: {str(e)}")
        
        return validation_results
    
    def get_model_recommendations(self, dataset_size: int, 
                                n_features: int,
                                problem_type: str = 'classification') -> List[str]:
        """Get model recommendations based on dataset characteristics."""
        recommendations = []
        
        if dataset_size < 1000:
            # Small dataset
            if n_features < 20:
                recommendations = ['decision_tree', 'linear_regression', 'svm']
            else:
                recommendations = ['linear_regression', 'pca', 'decision_tree']
                
        elif dataset_size < 10000:
            # Medium dataset
            if n_features < 50:
                recommendations = ['random_forest', 'xgboost', 'svm', 'decision_tree']
            else:
                recommendations = ['random_forest', 'pca', 'xgboost']
                
        else:
            # Large dataset
            if n_features < 100:
                recommendations = ['xgboost', 'random_forest', 'deep_learning']
            else:
                recommendations = ['pca', 'xgboost', 'deep_learning']
        
        return recommendations