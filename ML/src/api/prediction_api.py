from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging

from ..utils.model_factory import ModelFactory
from ..data.data_processor import DataProcessor


class PredictionAPI:
    """API interface for making predictions with trained exoplanet models."""
    
    def __init__(self):
        self.model_factory = ModelFactory()
        self.logger = logging.getLogger(__name__)
        
        # Loaded models cache
        self.loaded_models = {}
        
    def load_model(self, model_path: str, model_id: str = None) -> Dict[str, Any]:
        """Load a trained model for predictions."""
        try:
            if model_id is None:
                model_id = Path(model_path).stem
            
            # Load metadata to determine model type
            metadata_path = f"{model_path}_metadata.json"
            if Path(metadata_path).exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                model_type = metadata['model_type']
            else:
                # Try to infer from path or default
                model_type = 'random_forest'  # Default fallback
                metadata = {}
            
            # Create model instance
            model = self.model_factory.create_model(model_type)
            
            # Load the trained model
            model.load_model(model_path)
            
            # Cache the model
            self.loaded_models[model_id] = {
                'model': model,
                'metadata': metadata,
                'model_path': model_path
            }
            
            return {
                'model_id': model_id,
                'status': 'success',
                'model_info': model.get_model_info(),
                'metadata': metadata
            }
            
        except Exception as e:
            self.logger.error(f"Error loading model from {model_path}: {str(e)}")
            return {
                'model_id': model_id,
                'status': 'error',
                'error': str(e)
            }
    
    def predict_single(self, model_id: str, 
                      input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction for a single instance."""
        try:
            if model_id not in self.loaded_models:
                raise ValueError(f"Model {model_id} not loaded")
            
            model_info = self.loaded_models[model_id]
            model = model_info['model']
            
            # Convert input to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Validate input
            model.validate_input(input_df)
            
            # Make prediction
            prediction = model.predict(input_df)
            probabilities = model.predict_proba(input_df)
            
            # Format results
            result = {
                'model_id': model_id,
                'status': 'success',
                'prediction': prediction[0],
                'probabilities': probabilities[0].tolist(),
                'input_data': input_data
            }
            
            # Add class labels if available
            if model.target_classes:
                prob_dict = dict(zip(model.target_classes, probabilities[0]))
                result['class_probabilities'] = prob_dict
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error making prediction with model {model_id}: {str(e)}")
            return {
                'model_id': model_id,
                'status': 'error',
                'error': str(e)
            }
    
    def predict_batch(self, model_id: str, 
                     input_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Make predictions for multiple instances."""
        try:
            if model_id not in self.loaded_models:
                raise ValueError(f"Model {model_id} not loaded")
            
            model_info = self.loaded_models[model_id]
            model = model_info['model']
            
            # Convert input to DataFrame
            input_df = pd.DataFrame(input_data)
            
            # Validate input
            model.validate_input(input_df)
            
            # Make predictions
            predictions = model.predict(input_df)
            probabilities = model.predict_proba(input_df)
            
            # Format results
            results = []
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                result_item = {
                    'index': i,
                    'prediction': pred,
                    'probabilities': prob.tolist(),
                    'input_data': input_data[i]
                }
                
                # Add class labels if available
                if model.target_classes:
                    prob_dict = dict(zip(model.target_classes, prob))
                    result_item['class_probabilities'] = prob_dict
                
                results.append(result_item)
            
            return {
                'model_id': model_id,
                'status': 'success',
                'predictions': results,
                'batch_size': len(input_data)
            }
            
        except Exception as e:
            self.logger.error(f"Error making batch predictions with model {model_id}: {str(e)}")
            return {
                'model_id': model_id,
                'status': 'error',
                'error': str(e)
            }
    
    def predict_from_csv(self, model_id: str, 
                        csv_path: str) -> Dict[str, Any]:
        """Make predictions for data from CSV file."""
        try:
            if model_id not in self.loaded_models:
                raise ValueError(f"Model {model_id} not loaded")
            
            model_info = self.loaded_models[model_id]
            model = model_info['model']
            
            # Load CSV data
            input_df = pd.read_csv(csv_path)
            
            # Validate input
            model.validate_input(input_df)
            
            # Make predictions
            predictions = model.predict(input_df)
            probabilities = model.predict_proba(input_df)
            
            # Create results DataFrame
            results_df = input_df.copy()
            results_df['prediction'] = predictions
            
            # Add probability columns
            if model.target_classes:
                for i, class_name in enumerate(model.target_classes):
                    results_df[f'prob_{class_name}'] = probabilities[:, i]
            
            # Save results
            output_path = csv_path.replace('.csv', '_predictions.csv')
            results_df.to_csv(output_path, index=False)
            
            return {
                'model_id': model_id,
                'status': 'success',
                'input_file': csv_path,
                'output_file': output_path,
                'predictions_count': len(predictions),
                'predictions_summary': pd.Series(predictions).value_counts().to_dict()
            }
            
        except Exception as e:
            self.logger.error(f"Error making predictions from CSV with model {model_id}: {str(e)}")
            return {
                'model_id': model_id,
                'status': 'error',
                'error': str(e)
            }
    
    def get_prediction_confidence(self, model_id: str, 
                                input_data: Dict[str, Any],
                                confidence_threshold: float = 0.8) -> Dict[str, Any]:
        """Get prediction with confidence analysis."""
        try:
            prediction_result = self.predict_single(model_id, input_data)
            
            if prediction_result['status'] != 'success':
                return prediction_result
            
            probabilities = np.array(prediction_result['probabilities'])
            max_prob = np.max(probabilities)
            
            # Determine confidence level
            if max_prob >= confidence_threshold:
                confidence_level = 'high'
            elif max_prob >= 0.6:
                confidence_level = 'medium'
            else:
                confidence_level = 'low'
            
            # Add confidence analysis
            prediction_result['confidence_analysis'] = {
                'max_probability': float(max_prob),
                'confidence_level': confidence_level,
                'confidence_threshold': confidence_threshold,
                'entropy': float(-np.sum(probabilities * np.log(probabilities + 1e-10)))
            }
            
            return prediction_result
            
        except Exception as e:
            self.logger.error(f"Error analyzing prediction confidence: {str(e)}")
            return {
                'model_id': model_id,
                'status': 'error',
                'error': str(e)
            }
    
    def get_loaded_models(self) -> Dict[str, Any]:
        """Get information about all loaded models."""
        models_info = {}
        
        for model_id, model_data in self.loaded_models.items():
            model = model_data['model']
            models_info[model_id] = {
                'model_info': model.get_model_info(),
                'metadata': model_data['metadata'],
                'model_path': model_data['model_path']
            }
        
        return {
            'status': 'success',
            'loaded_models': models_info,
            'total_models': len(self.loaded_models)
        }
    
    def unload_model(self, model_id: str) -> Dict[str, Any]:
        """Unload a model from memory."""
        if model_id in self.loaded_models:
            del self.loaded_models[model_id]
            return {
                'model_id': model_id,
                'status': 'success',
                'message': 'Model unloaded'
            }
        else:
            return {
                'model_id': model_id,
                'status': 'error',
                'error': 'Model not found'
            }
    
    def validate_input_format(self, model_id: str, 
                            input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input data format for a model."""
        try:
            if model_id not in self.loaded_models:
                raise ValueError(f"Model {model_id} not loaded")
            
            model = self.loaded_models[model_id]['model']
            
            # Convert to DataFrame and validate
            input_df = pd.DataFrame([input_data])
            is_valid = model.validate_input(input_df)
            
            validation_info = {
                'is_valid': is_valid,
                'required_features': model.feature_names,
                'provided_features': list(input_data.keys()),
                'missing_features': [f for f in model.feature_names if f not in input_data.keys()],
                'extra_features': [f for f in input_data.keys() if f not in model.feature_names]
            }
            
            return {
                'model_id': model_id,
                'status': 'success',
                'validation': validation_info
            }
            
        except Exception as e:
            return {
                'model_id': model_id,
                'status': 'error',
                'error': str(e)
            }