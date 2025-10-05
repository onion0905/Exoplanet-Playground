import pandas as pd
from src.api.training_api import TrainingAPI
from src.data.include_patterns import KEPLER_INCLUDE_PATTERNS
from src.utils.model_factory import ModelFactory


# Load Kepler trainval data for training
kepler_trainval_path = "data/kepler_trainval.csv"
df = pd.read_csv(kepler_trainval_path, comment="#")

# Use only rows with non-null koi_disposition for training
train_df = df[df["koi_disposition"].notnull()].copy()

# Ensure all include pattern columns are present (add NaN if missing)
for col in KEPLER_INCLUDE_PATTERNS:
    if col not in train_df.columns:
        train_df[col] = float('nan')

# Load Kepler test set for prediction
kepler_test_path = "data/kepler_test.csv"
try:
    test_df = pd.read_csv(kepler_test_path, comment="#")
    # Only keep columns in KEPLER_INCLUDE_PATTERNS
    test_df = test_df[[col for col in KEPLER_INCLUDE_PATTERNS if col in test_df.columns]]
    # Add missing columns as NaN
    for col in KEPLER_INCLUDE_PATTERNS:
        if col not in test_df.columns:
            test_df[col] = float('nan')
    # Remove label column if present (simulate true test set)
    if 'koi_disposition' in test_df.columns:
        test_df = test_df.drop(columns=['koi_disposition'])
    print(f"Loaded {len(test_df)} samples from kepler_test.csv for prediction.")
except Exception as e:
    print(f"Could not load kepler_test.csv: {e}")
    test_df = None


# Loop over all model types in ModelFactory
model_types = [
    'random_forest',
    'decision_tree',
    'svm',
    'linear_regression',
    'xgboost',
    'pca',
    'deep_learning'
]


for model_type in model_types:
    print(f"\n{'='*30}\nTesting model_type: {model_type}\n{'='*30}")
    api = TrainingAPI()
    config = {
        'dataset_name': 'kepler',
        'target_column': 'koi_disposition',
        'model_type': model_type,
        'feature_columns': [col for col in KEPLER_INCLUDE_PATTERNS if col != 'koi_disposition'],
        'hyperparameters': {},
        'preprocessing_config': {
            'impute_with_nasa_means': True,
            'scale_features': False
        }
    }
    # Add model-specific hyperparameters if needed
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
        print("\n=== Predict on kepler_test.csv (external test set) ===")
        try:
            y_pred_external = api.current_session[session_id]['model'].predict(test_df)
            print(y_pred_external)
            # Optionally, print explanations for first 3 samples
            try:
                pred_expl = api.current_session[session_id]['model'].predict(test_df.iloc[:3], explain=True)
                for i, entry in enumerate(pred_expl):
                    print(f"Sample {i}: label={entry.get('label', None)}, confidence={entry.get('confidence', None)}")
                    print("  Top 5 features:", entry.get('top_features', entry.get('explanation', None)))
            except Exception as e:
                print(f"Explanation on external test set failed for {model_type}: {e}")
        except Exception as e:
            print(f"Prediction on kepler_test.csv failed for {model_type}: {e}")

    print("\n=== Feature Importances ===")
    importances = api.get_feature_importances(session_id)
    if importances:
        for feat, imp in importances:
            print(f"{feat}: {imp:.4f}")
    else:
        print("Model does not support feature importances.")

    print("\n=== Predict on First 3 Test Samples (top 5 features) ===")
    try:
        pred_expl = api.predict_with_explanation(session_id, X=api.current_session[session_id]['prepared_data']['X_test'].iloc[:3])
        for i, entry in enumerate(pred_expl):
            print(f"Sample {i}: label={entry['label']}, confidence={entry['confidence']:.3f}")
            print("  Top 5 features:", entry.get('top_features', entry.get('explanation', None)))
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
