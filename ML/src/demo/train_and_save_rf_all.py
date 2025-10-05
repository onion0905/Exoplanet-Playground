import pandas as pd
import os
from src.api.training_api import TrainingAPI
from src.data.include_patterns import KEPLER_INCLUDE_PATTERNS, K2_INCLUDE_PATTERNS, TESS_INCLUDE_PATTERNS

# Dataset configs
DATASETS = [
    {
        'name': 'kepler',
        'csv': 'data/kepler_trainval.csv',
        'target': 'koi_disposition',
        'include': KEPLER_INCLUDE_PATTERNS,
        'model_path': 'ML/models/pretrained/random_forest_kepler',
    },
    {
        'name': 'k2',
        'csv': 'data/k2_trainval.csv',
        'target': 'disposition',
        'include': K2_INCLUDE_PATTERNS,
        'model_path': 'ML/models/pretrained/random_forest_k2',
    },
    {
        'name': 'tess',
        'csv': 'data/tess_trainval.csv',
        'target': 'tfopwg_disp',
        'include': TESS_INCLUDE_PATTERNS,
        'model_path': 'ML/models/pretrained/random_forest_tess',
    },
]

for ds in DATASETS:
    print(f"\n=== Training Random Forest for {ds['name'].upper()} ===")
    df = pd.read_csv(ds['csv'], comment="#")
    # Filter to rows with non-null target
    df = df[df[ds['target']].notnull()].copy()
    # Ensure all include pattern columns are present
    for col in ds['include']:
        if col not in df.columns:
            df[col] = float('nan')
    # Setup API and config
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
    # Save model
    save_path = os.path.abspath(ds['model_path'])
    api.save_trained_model(session_id, f"random_forest_{ds['name']}", save_path)
    print(f"Model saved to {save_path}")
