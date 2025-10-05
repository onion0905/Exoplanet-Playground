import pandas as pd
import os
from src.api.training_api import TrainingAPI
from src.api.prediction_api import PredictionAPI
from src.data.include_patterns import KEPLER_INCLUDE_PATTERNS, K2_INCLUDE_PATTERNS, TESS_INCLUDE_PATTERNS
from src.utils.model_factory import ModelFactory

datasets = [
    {
        'name': 'tess',
        'csv': 'data/tess_trainval.csv',
        'test_csv': 'data/tess_test.csv',
        'target': 'tfopwg_disp',
        'include': TESS_INCLUDE_PATTERNS,
    },
    {
        'name': 'kepler',
        'csv': 'data/kepler_trainval.csv',
        'test_csv': 'data/kepler_test.csv',
        'target': 'koi_disposition',
        'include': KEPLER_INCLUDE_PATTERNS,
    },
    {
        'name': 'k2',
        'csv': 'data/k2_trainval.csv',
        'test_csv': 'data/k2_test.csv',
        'target': 'disposition',
        'include': K2_INCLUDE_PATTERNS,
    },
]

model_types = [
    'random_forest',
    'decision_tree',
    'svm',
    'linear_regression',
    'xgboost',
    'pca',
    'deep_learning'
]

for ds in datasets:
    print(f"\n{'='*40}\nDATASET: {ds['name'].upper()}\n{'='*40}")
    df = pd.read_csv(ds['csv'], comment="#")
    train_df = df[df[ds['target']].notnull()].copy()
    for col in ds['include']:
        if col not in train_df.columns:
            train_df[col] = float('nan')
    # Prepare test set
    try:
        test_df = pd.read_csv(ds['test_csv'], comment="#")
        # Only keep columns in the include whitelist
        test_df = test_df[[col for col in ds['include'] if col in test_df.columns]]
        for col in ds['include']:
            if col not in test_df.columns:
                test_df[col] = float('nan')
        # Drop the target column if present
        if ds['target'] in test_df.columns:
            test_df = test_df.drop(columns=[ds['target']])
        # Drop any non-numeric columns (e.g., string columns like 'CANDIDATE')
        test_df = test_df.select_dtypes(include=['number'])
        print(f"Loaded {len(test_df)} samples from {ds['test_csv']} for prediction.")
    except Exception as e:
        print(f"Could not load {ds['test_csv']}: {e}")
        test_df = None

    for model_type in model_types:
        print(f"\n{'-'*30}\nModel: {model_type}\n{'-'*30}")
        api = TrainingAPI()
        pred_api = PredictionAPI()
        config = {
            'dataset_name': ds['name'],
            'target_column': ds['target'],
            'model_type': model_type,
            'feature_columns': [col for col in ds['include'] if col != ds['target']],
            'hyperparameters': {},
            'preprocessing_config': {
                'impute_with_nasa_means': True,
                'scale_features': False
            }
        }
        if model_type == 'random_forest':
            config['hyperparameters'] = {'n_estimators': 400, 'max_depth': 20}
        elif model_type == 'decision_tree':
            config['hyperparameters'] = {'max_depth': 20}
        elif model_type == 'xgboost':
            config['hyperparameters'] = {'n_estimators': 100, 'max_depth': 6}
        elif model_type == 'deep_learning':
            config['hyperparameters'] = {'hidden_layers': [64, 32], 'dropout_rate': 0.2, 'learning_rate': 0.001}
        session_id = api.quick_configure_training(train_df, config)
        api.configure_training(session_id, api.current_session[session_id]['training_config'])
        try:
            train_result = api.start_training(session_id)
        except Exception as e:
            print(f"Training failed for {model_type}: {e}")
            continue
        print("\n=== Confusion Matrix (Validation) ===")
        cm_result = api.get_validation_confusion_matrix(session_id)
        if cm_result:
            print("Labels:", cm_result['labels'])
            print(cm_result['confusion_matrix'])
        else:
            print("No validation set available.")
        # Predict on external test set if available
        if test_df is not None:
            print(f"\n=== Predict on {ds['test_csv']} (external test set) ===")
            try:
                processor = api.data_processor
                test_df_proc = test_df.copy()
                if processor.processing_config.get('handle_missing', True):
                    test_df_proc = processor.handle_missing_values(test_df_proc, processor.processing_config.get('missing_strategy', 'median'), fit=False)
                if processor.processing_config.get('encode_categorical', True):
                    test_df_proc = processor.encode_categorical_features(test_df_proc, fit=False)
                if processor.processing_config.get('scale_features', False):
                    test_df_proc = processor.scale_features(test_df_proc, fit=False)
                # Ensure only numeric columns are passed to the model
                test_df_proc = test_df_proc.select_dtypes(include=['number'])
                model = api.current_session[session_id]['model']
                pred_api.loaded_models['current'] = {'model': model, 'metadata': {}, 'model_path': None}
                # Only print summary: accuracy if possible, and 3 sample explanations
                try:
                    # If y_test is available, compute accuracy (optional, may not match test set exactly)
                    if 'y_test' in api.current_session[session_id]['prepared_data']:
                        y_test = api.current_session[session_id]['prepared_data']['y_test']
                        y_pred = model.predict(test_df_proc)
                        acc = (y_pred == y_test[:len(y_pred)]).mean()
                        print(f"External test set accuracy (first {len(y_pred)} samples): {acc:.3f}")
                    print("Sample explanations (first 3):")
                    for i, row in test_df_proc.iloc[:3].iterrows():
                        expl = pred_api.predict_with_explanation('current', row.to_dict())
                        print(f"  Sample {i}: label={expl['label']}, confidence={expl['confidence']}")
                        print("    Top 5 features:", expl.get('top_features', expl.get('explanation', None)))
                except Exception as e:
                    print(f"Explanation or accuracy on external test set failed for {model_type}: {e}")
            except Exception as e:
                print(f"Prediction on {ds['test_csv']} failed for {model_type}: {e}")
        print("\n=== Predict on First 3 Test Samples (top 5 features) ===")
        try:
            # Use PredictionAPI for explanations
            model = api.current_session[session_id]['model']
            pred_api.loaded_models['current'] = {'model': model, 'metadata': {}, 'model_path': None}
            X_test = api.current_session[session_id]['prepared_data']['X_test']
            for i, row in X_test.iloc[:3].iterrows():
                expl = pred_api.predict_with_explanation('current', row.to_dict())
                print(f"Sample {i}: label={expl['label']}, confidence={expl['confidence']:.3f}")
                print("  Top 5 features:", expl.get('top_features', expl.get('explanation', None)))
        except Exception as e:
            print(f"Explanation failed for {model_type}: {e}")
        print("\n=== Validation Accuracy and Metrics ===")
        val_acc = api.current_session[session_id].get('validation_accuracy', None)
        print(f"Validation accuracy: {val_acc}")
        val_metrics = api.get_validation_confusion_matrix(session_id)
        if val_metrics:
            print("Confusion matrix:")
            print(val_metrics['confusion_matrix'])
            print("Labels:", val_metrics['labels'])
        else:
            print("No validation set available.")
