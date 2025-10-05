import os
import pandas as pd
from typing import Optional, Dict, Any, List
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ML"))
from src.api.training_api import TrainingAPI
from src.api.prediction_api import PredictionAPI

def train_model(
    train_csv: str,
    target_col: str,
    feature_columns: List[str],
    model_type: str = 'random_forest',
    hyperparameters: Optional[Dict[str, Any]] = None,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Train a model with the given config and data file.
    Returns validation accuracy and confusion matrix. Optionally saves the model.
    """
    df = pd.read_csv(train_csv, comment="#")
    df = df[df[target_col].notnull()].copy()
    for col in feature_columns:
        if col not in df.columns:
            df[col] = float('nan')
    api = TrainingAPI()
    config = {
        'dataset_name': os.path.splitext(os.path.basename(train_csv))[0],
        'target_column': target_col,
        'model_type': model_type,
        'feature_columns': feature_columns,
        'hyperparameters': hyperparameters or {},
        'preprocessing_config': {
            'impute_with_nasa_means': True,
            'scale_features': False
        }
    }
    session_id = api.quick_configure_training(df, config)
    api.configure_training(session_id, api.current_session[session_id]['training_config'])
    train_result = api.start_training(session_id)
    val_acc = train_result.get('validation_accuracy')
    cm_result = api.get_validation_confusion_matrix(session_id)
    if save_path:
        save_path = os.path.abspath(save_path)
        api.save_trained_model(session_id, f"{model_type}_{os.path.basename(save_path)}", save_path)
    return {
        'validation_accuracy': val_acc,
        'confusion_matrix': cm_result['confusion_matrix'] if cm_result else None,
        'labels': cm_result['labels'] if cm_result else None,
        'model_path': save_path
    }

def predict_with_model(
    model_path: str,
    test_csv: str,
    feature_columns: List[str],
    target_col: Optional[str] = None,
    return_explanation: bool = False
) -> List[Dict[str, Any]]:
    """
    Load a model and predict on test data. Returns verdict, confidence, and optionally top 5 features.
    """
    df = pd.read_csv(test_csv, comment="#")
    if target_col and target_col in df.columns:
        df = df[df[target_col].notnull()].copy()
    for col in feature_columns:
        if col not in df.columns:
            df[col] = float('nan')
    X = df[feature_columns].copy()
    api = PredictionAPI()
    model_id = f"user_{os.path.basename(model_path)}"
    api.load_model(os.path.abspath(model_path), model_id)
    required_features = api.loaded_models[model_id]['model'].feature_names
    for col in required_features:
        if col not in X.columns or not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = float('nan')
    X = X[required_features]
    X = X.apply(pd.to_numeric, errors='coerce')
    input_data = X.to_dict(orient='records')
    results = []
    for i, row in enumerate(input_data):
        if i >= 10:
            break
        res = api.predict_with_explanation(model_id, row)
        result = {
            'verdict': res['label'],
            'confidence': res['confidence']
        }
        if return_explanation:
            result['top_features'] = res['top_features']
        results.append(result)
    return results

# Demo usage
if __name__ == "__main__":
    # Example: Train a model on Kepler data and save it
    from src.data.include_patterns import KEPLER_INCLUDE_PATTERNS
    train_result = train_model(
        train_csv='data/kepler_trainval.csv',
        target_col='koi_disposition',
        feature_columns=[col for col in KEPLER_INCLUDE_PATTERNS if col != 'koi_disposition'],
        model_type='random_forest',
        hyperparameters={'n_estimators': 100, 'max_depth': 10},
        save_path='ML/models/user/random_forest_kepler_demo'
    )
    print("Train result:", train_result)

    # Example: Predict with the saved model
    predictions = predict_with_model(
        model_path='ML/models/user/random_forest_kepler_demo',
        test_csv='data/kepler_test.csv',
        feature_columns=[col for col in KEPLER_INCLUDE_PATTERNS if col != 'koi_disposition'],
        target_col='koi_disposition',
        return_explanation=True
    )
    print("Predictions (first 3):", predictions[:3])
