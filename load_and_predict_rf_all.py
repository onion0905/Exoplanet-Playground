import pandas as pd
import os
from ML.src.api.prediction_api import PredictionAPI
from ML.src.data.include_patterns import KEPLER_INCLUDE_PATTERNS, K2_INCLUDE_PATTERNS, TESS_INCLUDE_PATTERNS

def load_and_predict(dataset_name, model_path, test_csv, target_col, include_patterns):
    print(f"\n=== {dataset_name.upper()} ===")
    df = pd.read_csv(test_csv, comment="#")
    df = df[df[target_col].notnull()].copy()
    for col in include_patterns:
        if col not in df.columns:
            df[col] = float('nan')
    X = df[[col for col in include_patterns if col != target_col and col in df.columns]].copy()
    y = df[target_col]
    api = PredictionAPI()
    model_id = f"pretrained_{dataset_name}"
    api.load_model(model_path, model_id)
    required_features = api.loaded_models[model_id]['model'].feature_names
    for col in required_features:
        if col not in X.columns or not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = float('nan')
    X = X[required_features]
    X = X.apply(pd.to_numeric, errors='coerce')
    input_data = X.to_dict(orient='records')
    def tess_label_map(label):
        label = str(label).strip().upper()
        if label in ["PC", "CANDIDATE"]:
            return "candidate"
        if label in ["PL", "PLANET", "CONFIRMED"]:
            return "planet"
        if label in ["FP", "FALSE POSITIVE"]:
            return "false_positive"
        return label.lower()
    for i, row in enumerate(input_data):
        res = api.predict_with_explanation(model_id, row)
        if dataset_name == 'tess':
            res['label'] = tess_label_map(res['label'])
        if i < 10:
            print(f"Sample {i+1}: label={res['label']}, confidence={res['confidence']:.3f}, top_features={res['top_features']}")
        else:
            return

if __name__ == "__main__":
    configs = [
        {
            'dataset_name': 'kepler',
            'model_path': 'ML/models/pretrained/random_forest_kepler',
            'test_csv': 'data/kepler_test.csv',
            'target_col': 'koi_disposition',
            'include_patterns': KEPLER_INCLUDE_PATTERNS,
        },
        {
            'dataset_name': 'k2',
            'model_path': 'ML/models/pretrained/random_forest_k2',
            'test_csv': 'data/k2_test.csv',
            'target_col': 'disposition',
            'include_patterns': K2_INCLUDE_PATTERNS,
        },
        {
            'dataset_name': 'tess',
            'model_path': 'ML/models/pretrained/random_forest_tess',
            'test_csv': 'data/tess_test.csv',
            'target_col': 'tfopwg_disp',
            'include_patterns': TESS_INCLUDE_PATTERNS,
        },
    ]
    for cfg in configs:
        load_and_predict(cfg['dataset_name'], os.path.abspath(cfg['model_path']), cfg['test_csv'], cfg['target_col'], cfg['include_patterns'])
