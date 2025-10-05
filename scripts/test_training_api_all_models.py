"""
Test all models in the TrainingAPI: report used/dropped columns, hyperparameters, and accuracies.
"""

import sys
import os
from pathlib import Path
import json

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ML.src.api.training_api import TrainingAPI

import uuid

MODEL_TYPES = [
    'random_forest',
    'decision_tree',
    'linear_regression',
    'svm',
    'xgboost',
    'deep_learning',
]

DATASET = 'kepler'
TARGET_COLUMN = 'koi_disposition'

def to_native(obj):
    import numpy as np
    if isinstance(obj, dict):
        return {to_native(k): to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_native(x) for x in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    else:
        return obj

def explain_dropped_columns(data_info, used_columns, recommended_features):
    """Explain why each dropped column was dropped."""
    dropped = {}
    columns = data_info['columns']
    dtypes = data_info.get('dtypes', {})
    missing = data_info.get('missing_values', {})
    for col in columns:
        if col in used_columns or col == TARGET_COLUMN:
            continue
        reason = []
        # Check for too many missing values
        if col in missing and data_info['shape'][0] > 0:
            miss_ratio = missing[col] / data_info['shape'][0]
            if miss_ratio > 0.5:
                reason.append(f"too many missing values ({miss_ratio:.0%})")
        # Check for dtype
        dtype = dtypes.get(col, '')
        if dtype == 'object':
            reason.append("non-numeric or high-cardinality categorical")
        # Check for patterns (simulate DataValidator patterns)
        col_lower = col.lower()
        patterns = [
            ('name/id', ['name', 'hostname', 'id', 'toi', 'kepid', 'tid']),
            ('reference', ['ref', 'delivname', 'facility', 'refname']),
            ('date', ['date', 'created', 'update', 'year', 'pubdate', 'release']),
            ('flag/lim/str', ['flag', 'lim', 'prov', 'str']),
            ('target_leakage', ['disposition', 'disp', 'score']),
            ('metadata', ['default_flag', 'controv_flag', 'ttv_flag', 'snum', 'pnum', 'soltype'])
        ]
        for label, pats in patterns:
            for pat in pats:
                if pat in col_lower:
                    reason.append(label)
        # Do not drop columns just because they are not in recommended features
        dropped[col] = reason
    return dropped

results = []


# Define hyperparameter grids for each model
from itertools import product

# Expanded parameter grids for random_forest and xgboost
rf_grid = [
    {'n_estimators': n, 'max_depth': d, 'min_samples_split': mss, 'min_samples_leaf': msl, 'criterion': c}
    for n, d, mss, msl, c in product(
        [100, 200, 300],
        [8, 16, None],
        [2, 4, 8],
        [1, 2, 4],
        ['gini', 'entropy']
    )
]
xgb_grid = [
    {'n_estimators': n, 'max_depth': d, 'learning_rate': lr, 'subsample': ss, 'colsample_bytree': cs, 'reg_alpha': ra, 'reg_lambda': rl}
    for n, d, lr, ss, cs, ra, rl in product(
        [100, 200, 300],
        [4, 8, 12],
        [0.05, 0.1, 0.2],
        [0.8, 1.0],
        [0.8, 1.0],
        [0, 0.1],
        [1, 2]
    )
]
param_grids = {
    'random_forest': rf_grid,
    'decision_tree': [
        {'max_depth': 4},
        {'max_depth': 8},
        {'min_samples_split': 4},
        {'min_samples_leaf': 2},
        {'criterion': 'entropy'},
    ],
    'svm': [
        {'C': 0.5},
        {'C': 2.0},
        {'kernel': 'linear'},
        {'kernel': 'rbf', 'gamma': 0.1},
    ],
    'xgboost': xgb_grid,
    'linear_regression': [
        {'C': 0.5},
        {'C': 2.0},
        {'solver': 'liblinear'},
        {'max_iter': 2000},
    ],
}

for model_type in MODEL_TYPES:
    print(f"\n=== Testing model: {model_type} ===")
    best_result = None
    best_val_acc = -1
    best_hyperparams = None
    best_confusion = None
    best_config = None
    best_used_columns = None
    best_dropped_columns = None
    best_dropped_explanations = None
    api = TrainingAPI()
    session_id = f"session-{model_type}-{uuid.uuid4().hex[:6]}"
    api.start_training_session(session_id)
    load_result = api.load_data_for_training(
        session_id,
        data_source="nasa",
        data_config={"datasets": [DATASET], "target_column": TARGET_COLUMN}
    )
    if load_result.get('status') != 'success':
        print(f"Failed to load data: {load_result.get('error')}")
        continue
    data_info = load_result['data_info']
    columns = data_info['columns']
    recommended_features = data_info.get('recommended_features', [])
    print(f"Recommended features: {recommended_features}")
    # For deep_learning, just run default
    if model_type == 'deep_learning':
        config_result = api.configure_training(
            session_id,
            training_config={
                "model_type": model_type,
                "target_column": TARGET_COLUMN
            }
        )
        if config_result.get('status') != 'success':
            print(f"Failed to configure training: {config_result.get('error')}")
            continue
        config = config_result['training_config']
        used_columns = config['feature_columns']
        dropped_columns = [col for col in columns if col not in used_columns and col != TARGET_COLUMN]
        dropped_explanations = explain_dropped_columns(data_info, used_columns, recommended_features)
        print(f"Used columns: {used_columns}")
        print("Dropped columns and reasons:")
        for col in dropped_columns:
            print(f"  {col}: {', '.join(dropped_explanations.get(col, []))}")
        print(f"Training parameters: {config}")
        train_result = api.start_training(session_id)
        if train_result.get('status') != 'success':
            print(f"Failed to train: {train_result.get('error')}")
            continue
        session_info = api.get_session_info(session_id, include_data=False)
        training_history = session_info.get('session_info', {}).get('model', {}).get('training_history', {})
        hyperparams = {k: v for k, v in training_history.items() if k not in ['train_accuracy', 'val_accuracy', 'feature_importances', 'tree_depth', 'n_leaves', 'oob_score']}
        print(f"Model hyperparameters: {hyperparams}")
        eval_metrics = train_result.get('evaluation_metrics', {})
        confusion = eval_metrics.get('confusion_matrix')
        if confusion is not None:
            print("Confusion matrix:")
            for row in confusion:
                print("  ", row)
        else:
            print("Confusion matrix not available.")
        val_acc = train_result.get('validation_accuracy', None)
        print(f"Validation accuracy: {val_acc}")
        results.append({
            'model_type': model_type,
            'recommended_features': recommended_features,
            'used_columns': used_columns,
            'dropped_columns': dropped_columns,
            'dropped_explanations': {col: dropped_explanations.get(col, []) for col in dropped_columns},
            'training_parameters': config,
            'hyperparameters': hyperparams,
            'validation_accuracy': val_acc,
            'confusion_matrix': confusion
        })
        continue
    # For non-neural models, try parameter grid
    tried_params = param_grids.get(model_type, [{}])
    for param_set in tried_params:
        config_result = api.configure_training(
            session_id,
            training_config={
                "model_type": model_type,
                "target_column": TARGET_COLUMN,
                "hyperparameters": param_set
            }
        )
        if config_result.get('status') != 'success':
            print(f"Failed to configure training: {config_result.get('error')} for params {param_set}")
            continue
        config = config_result['training_config']
        used_columns = config['feature_columns']
        dropped_columns = [col for col in columns if col not in used_columns and col != TARGET_COLUMN]
        dropped_explanations = explain_dropped_columns(data_info, used_columns, recommended_features)
        train_result = api.start_training(session_id)
        if train_result.get('status') != 'success':
            print(f"Failed to train: {train_result.get('error')} for params {param_set}")
            continue
        session_info = api.get_session_info(session_id, include_data=False)
        training_history = session_info.get('session_info', {}).get('model', {}).get('training_history', {})
        hyperparams = {k: v for k, v in training_history.items() if k not in ['train_accuracy', 'val_accuracy', 'feature_importances', 'tree_depth', 'n_leaves', 'oob_score']}
        eval_metrics = train_result.get('evaluation_metrics', {})
        confusion = eval_metrics.get('confusion_matrix')
        val_acc = train_result.get('validation_accuracy', None)
    # Do not print parameter/flag info
        if val_acc is not None and val_acc > best_val_acc:
            best_val_acc = val_acc
            best_result = train_result
            best_hyperparams = hyperparams
            best_confusion = confusion
            best_config = config
            best_used_columns = used_columns
            best_dropped_columns = dropped_columns
            best_dropped_explanations = dropped_explanations
    if best_result is not None:
        print(f"Best validation accuracy for {model_type}: {best_val_acc}")
        print(f"Best hyperparameters: {best_hyperparams}")
        print(f"Used columns: {best_used_columns}")
        print("Dropped columns and reasons:")
        for col in best_dropped_columns:
            print(f"  {col}: {', '.join(best_dropped_explanations.get(col, []))}")
        print(f"Training parameters: {best_config}")
        if best_confusion is not None:
            print("Confusion matrix:")
            for row in best_confusion:
                print("  ", row)
        else:
            print("Confusion matrix not available.")
        results.append({
            'model_type': model_type,
            'recommended_features': recommended_features,
            'used_columns': best_used_columns,
            'dropped_columns': best_dropped_columns,
            'dropped_explanations': {col: best_dropped_explanations.get(col, []) for col in best_dropped_columns},
            'training_parameters': best_config,
            'hyperparameters': best_hyperparams,
            'validation_accuracy': best_val_acc,
            'confusion_matrix': best_confusion
        })
    else:
        print(f"No successful training for {model_type}.")

# Save results to file (convert numpy types)
with open('model_training_results.json', 'w') as f:
    json.dump(to_native(results), f, indent=2)

print("\nAll model results saved to model_training_results.json")
