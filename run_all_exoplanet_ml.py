
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ML"))

import pandas as pd
from src.api.training_api import TrainingAPI
from src.api.prediction_api import PredictionAPI
from src.data.include_patterns import KEPLER_INCLUDE_PATTERNS, K2_INCLUDE_PATTERNS, TESS_INCLUDE_PATTERNS

def tess_label_map(label):
    label = str(label).strip().upper()
    if label in ["PC", "CANDIDATE"]:
        return "candidate"
    if label in ["PL", "PLANET", "CONFIRMED"]:
        return "planet"
    if label in ["FP", "FALSE POSITIVE"]:
        return "false_positive"
    return label.lower()

DATASETS = [
    {
        'name': 'kepler',
        'trainval_csv': 'data/kepler_trainval.csv',
        'test_csv': 'data/kepler_test.csv',
        'target': 'koi_disposition',
        'include': KEPLER_INCLUDE_PATTERNS,
        'model_path': 'ML/models/pretrained/random_forest_kepler',
    },
    {
        'name': 'k2',
        'trainval_csv': 'data/k2_trainval.csv',
        'test_csv': 'data/k2_test.csv',
        'target': 'disposition',
        'include': K2_INCLUDE_PATTERNS,
        'model_path': 'ML/models/pretrained/random_forest_k2',
    },
    {
        'name': 'tess',
        'trainval_csv': 'data/tess_trainval.csv',
        'test_csv': 'data/tess_test.csv',
        'target': 'tfopwg_disp',
        'include': TESS_INCLUDE_PATTERNS,
        'model_path': 'ML/models/pretrained/random_forest_tess',
    },
]

def train_and_save(ds):
    print(f"\n=== Training Random Forest for {ds['name'].upper()} ===")
    df = pd.read_csv(ds['trainval_csv'], comment="#")
    df = df[df[ds['target']].notnull()].copy()
    for col in ds['include']:
        if col not in df.columns:
            df[col] = float('nan')
    api = TrainingAPI()
    config = {
        'dataset_name': ds['name'],
        'target_column': ds['target'],
        'model_type': 'random_forest',
        'feature_columns': [col for col in ds['include'] if col != ds['target']],
        'hyperparameters': {'n_estimators': 400, 'max_depth': 20},
        'preprocessing_config': {
            'impute_with_nasa_means': True,
            'scale_features': False
        }
    }
    session_id = api.quick_configure_training(df, config)
    api.configure_training(session_id, api.current_session[session_id]['training_config'])
    train_result = api.start_training(session_id)
    print("Training metrics:", train_result.get('training_metrics'))
    print("Evaluation metrics:", train_result.get('evaluation_metrics'))
    print(f"Validation accuracy: {train_result.get('validation_accuracy')}")
    cm_result = api.get_validation_confusion_matrix(session_id)
    if cm_result:
        print("Confusion Matrix (Validation):")
        print("Labels:", cm_result['labels'])
        print(cm_result['confusion_matrix'])
    else:
        print("No validation set available.")
    save_path = os.path.abspath(ds['model_path'])
    api.save_trained_model(session_id, f"random_forest_{ds['name']}", save_path)
    print(f"Model saved to {save_path}")

def predict_and_explain(ds):
    print(f"\n=== {ds['name'].upper()} TEST SET PREDICTION ===")
    df = pd.read_csv(ds['test_csv'], comment="#")
    df = df[df[ds['target']].notnull()].copy()
    for col in ds['include']:
        if col not in df.columns:
            df[col] = float('nan')
    X = df[[col for col in ds['include'] if col != ds['target'] and col in df.columns]].copy()
    api = PredictionAPI()
    model_id = f"pretrained_{ds['name']}"
    api.load_model(os.path.abspath(ds['model_path']), model_id)
    required_features = api.loaded_models[model_id]['model'].feature_names
    for col in required_features:
        if col not in X.columns or not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = float('nan')
    X = X[required_features]
    X = X.apply(pd.to_numeric, errors='coerce')
    input_data = X.to_dict(orient='records')
    for i, row in enumerate(input_data):
        res = api.predict_with_explanation(model_id, row)
        if ds['name'] == 'tess':
            res['label'] = tess_label_map(res['label'])
        if i < 10:
            print(f"Sample {i+1}: label={res['label']}, confidence={res['confidence']:.3f}, top_features={res['top_features']}")
        else:
            break

if __name__ == "__main__":
    for ds in DATASETS:
        train_and_save(ds)
        predict_and_explain(ds)
