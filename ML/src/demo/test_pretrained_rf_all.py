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
    print(f"\n=== Testing Random Forest for {ds['name'].upper()} ===")
    df = pd.read_csv(ds['csv'], comment="#")
    df = df[df[ds['target']].notnull()].copy()
    for col in ds['include']:
        if col not in df.columns:
            df[col] = float('nan')
    X = df[[col for col in ds['include'] if col != ds['target']]]
    # Only keep numeric columns for prediction
    X = X.select_dtypes(include=[float, int])
    y = df[ds['target']]
    # Load model
    api = TrainingAPI()
    model_path = os.path.abspath(ds['model_path'])
    model_id = f"random_forest_{ds['name']}"
    api.current_session[model_id] = {'model': None}  # placeholder for API
    api.current_session[model_id]['model'] = api.model_factory.create_model('random_forest')
    api.current_session[model_id]['model'].load_model(model_path)
    # Predict
    y_pred = api.current_session[model_id]['model'].predict(X)
    acc = (y_pred == y).mean()
    print(f"Accuracy on {ds['csv']}: {acc:.4f}")
    # Optionally print confusion matrix
    try:
        from sklearn.metrics import confusion_matrix
        labels = sorted(y.unique())
        cm = confusion_matrix(y, y_pred, labels=labels)
        print("Labels:", labels)
        print("Confusion matrix:\n", cm)
    except Exception as e:
        print(f"Could not compute confusion matrix: {e}")
