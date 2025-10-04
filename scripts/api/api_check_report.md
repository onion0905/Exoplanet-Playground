# API Function Check Report
Generated: 2025-10-04 12:42:14

## Summary
- Total APIs checked: 4
- Total issues found: 35

## TrainingAPI
- Total methods: 8
- Public methods: 7
- Issues: 7

### Issues
- TrainingAPI.configure_training: API method should return Dict[str, Any] for consistency
- TrainingAPI.get_session_info: API method should return Dict[str, Any] for consistency
- TrainingAPI.get_training_progress: API method should return Dict[str, Any] for consistency
- TrainingAPI.load_data_for_training: API method should return Dict[str, Any] for consistency
- TrainingAPI.save_trained_model: API method should return Dict[str, Any] for consistency
- TrainingAPI.start_training: API method should return Dict[str, Any] for consistency
- TrainingAPI.start_training_session: API method should return Dict[str, Any] for consistency

## PredictionAPI
- Total methods: 8
- Public methods: 8
- Issues: 8

### Issues
- PredictionAPI.get_loaded_models: API method should return Dict[str, Any] for consistency
- PredictionAPI.get_prediction_confidence: API method should return Dict[str, Any] for consistency
- PredictionAPI.load_model: API method should return Dict[str, Any] for consistency
- PredictionAPI.predict_batch: API method should return Dict[str, Any] for consistency
- PredictionAPI.predict_from_csv: API method should return Dict[str, Any] for consistency
- PredictionAPI.predict_single: API method should return Dict[str, Any] for consistency
- PredictionAPI.unload_model: API method should return Dict[str, Any] for consistency
- PredictionAPI.validate_input_format: API method should return Dict[str, Any] for consistency

## ExplanationAPI
- Total methods: 7
- Public methods: 7
- Issues: 9

### Issues
- ExplanationAPI.analyze_feature_importance_drop: API method should return Dict[str, Any] for consistency
- ExplanationAPI.apply_column_selection: ExplanationAPI method should typically have model_id parameter
- ExplanationAPI.apply_column_selection: API method should return Dict[str, Any] for consistency
- ExplanationAPI.compare_feature_importance_methods: API method should return Dict[str, Any] for consistency
- ExplanationAPI.explain_model_decision_path: API method should return Dict[str, Any] for consistency
- ExplanationAPI.explain_model_global: API method should return Dict[str, Any] for consistency
- ExplanationAPI.explain_prediction_local: API method should return Dict[str, Any] for consistency
- ExplanationAPI.get_column_selection_info: ExplanationAPI method should typically have model_id parameter
- ExplanationAPI.get_column_selection_info: API method should return Dict[str, Any] for consistency

## ExoplanetMLAPI
- Total methods: 11
- Public methods: 11
- Issues: 11

### Issues
- ExoplanetMLAPI.analyze_prediction_factors: API method should return Dict[str, Any] for consistency
- ExoplanetMLAPI.generate_pretrained_models: API method should return Dict[str, Any] for consistency
- ExoplanetMLAPI.get_dataset_info: API method should return Dict[str, Any] for consistency
- ExoplanetMLAPI.get_feature_importance: API method should return Dict[str, Any] for consistency
- ExoplanetMLAPI.get_sample_data: API method should return Dict[str, Any] for consistency
- ExoplanetMLAPI.list_available_datasets: API method should return Dict[str, Any] for consistency
- ExoplanetMLAPI.list_available_models: API method should return Dict[str, Any] for consistency
- ExoplanetMLAPI.list_trained_models: API method should return Dict[str, Any] for consistency
- ExoplanetMLAPI.predict_batch: API method should return Dict[str, Any] for consistency
- ExoplanetMLAPI.predict_single: API method should return Dict[str, Any] for consistency
- ExoplanetMLAPI.train_model: API method should return Dict[str, Any] for consistency

