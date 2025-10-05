import pandas as pd
import os
from src.api.training_api import TrainingAPI
from src.data.include_patterns import KEPLER_INCLUDE_PATTERNS, K2_INCLUDE_PATTERNS, TESS_INCLUDE_PATTERNS

# Example: Load a pretrained random forest model and predict on test set for Kepler, K2, and TESS
def load_and_predict(dataset_name, model_path, test_csv, target_col, include_patterns):
    print(f"\n=== {dataset_name.upper()} ===")
    df = pd.read_csv(test_csv, comment="#")
    df = df[df[target_col].notnull()].copy()
    # Ensure all include pattern columns are present
    for col in include_patterns:
        if col not in df.columns:
            df[col] = float('nan')
    X = df[[col for col in include_patterns if col != target_col]]
    # Only keep numeric columns for prediction
    X = X.select_dtypes(include=[float, int])
    y = df[target_col]
    # Setup API and load model
    api = TrainingAPI()
    session_id = f"pretrained_{dataset_name}"
    api.current_session[session_id] = {'model': None}
    model = api.model_factory.create_model('random_forest')
    model.load_model(model_path)
    # Ensure all required features are present in X, add missing as NaN, and reorder
    required_features = model.feature_names
    for col in required_features:
        if col not in X.columns:
            X[col] = float('nan')
    X = X[required_features]
    api.current_session[session_id]['model'] = model
    # Predict with explanations
    results = model.predict(X, explain=True)
    for i, res in enumerate(results[:10]):  # Print first 10 predictions
        print(f"Sample {i+1}: label={res['label']}, confidence={res['confidence']:.3f}, top_features={res['top_features']}")

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
