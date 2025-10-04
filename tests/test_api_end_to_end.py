"""
End-to-End API Test Script for Exoplanet Playground

This script tests all major features of the ML system via API calls:
- List datasets and models
- Train, save, and load models
- Predict exoplanet status
- Feature importance
- Subset training
- Error handling and edge cases

Usage:
    env PYTHONPATH=. python -m tests.test_api_end_to_end
"""
import sys
import os
import random
import json
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ML.src.api.user_api import ExoplanetMLAPI
from ML.src.api.training_api import TrainingAPI
from ML.src.api.prediction_api import PredictionAPI
from ML.src.api.explanation_api import ExplanationAPI

def print_header(msg):
    print("\n" + "="*40)
    print(msg)
    print("="*40)

def main():
    print_header("API: List datasets and models")
    api = ExoplanetMLAPI()
    datasets = api.list_available_datasets()
    print(f"Datasets: {datasets}")
    model_types = api.list_available_models()
    print(f"Model types: {model_types}")

    print_header("API: Train, save, and load model")
    train_api = TrainingAPI()
    import uuid
    session_id = f"session-{uuid.uuid4().hex[:8]}"
    train_api.start_training_session(session_id)
    train_api.load_data_for_training(
        session_id,
        data_source="nasa",
        data_config={"datasets": ["kepler"], "target_column": "koi_disposition"}
    )
    train_api.configure_training(
        session_id,
        training_config={
            "model_type": "random_forest",
            "target_column": "koi_disposition"
        }
    )
    train_api.start_training(session_id)
    save_info = train_api.save_trained_model(session_id, model_name="rf_kepler_test_api")
    model_path = save_info.get("model_path")
    print(f"Model saved to: {model_path}")

    print_header("API: Predict exoplanet status")
    pred_api = PredictionAPI()
    # Use ExoplanetMLAPI to get a robust sample for prediction
    user_api = ExoplanetMLAPI()
    sample_data = user_api.get_sample_data('kepler', n_samples=1)
    if 'error' in sample_data:
        print(f"Error getting sample data for prediction: {sample_data['error']}")
        sample = None
        true_label = None
    else:
        sample = sample_data['sample_data'][0]
        # Remove target column if present
        for col in ['koi_disposition', 'tfopwg_disp', 'disposition']:
            true_label = sample.pop(col, None)
    if sample is not None:
        # Load the model for prediction
        load_result = pred_api.load_model(model_path)
        if load_result.get('status') != 'success':
            print(f"Error loading model for prediction: {load_result.get('error')}")
        else:
            model_id = load_result['model_id']
            pred = pred_api.predict_single(model_id, sample)
            print(f"Sample true label: {true_label}, predicted: {pred.get('prediction', 'N/A')}")
    else:
        print("Skipping prediction due to missing sample data.")

    print_header("API: Feature importance")
    expl_api = ExplanationAPI(prediction_api=pred_api)
    # Always use prepared_data from the training session for feature importance
    session_info = train_api.get_session_info(session_id, include_data=True)
    prepared_data = session_info["session_info"].get("prepared_data")
    if prepared_data:
        import pandas as pd
        X_test = pd.DataFrame(prepared_data["X_test"])
        y_test = pd.Series(prepared_data["y_test"])
        X_train = pd.DataFrame(prepared_data["X_train"])
        y_train = pd.Series(prepared_data["y_train"])
        feat_imp = expl_api.explain_model_global(model_id, X_train, y_train, X_test, y_test)
        if feat_imp.get('status') == 'success':
            print("Feature importances (top features):")
            top_feats = feat_imp.get('top_features', [])
            # top_features is a list of (feature, score) tuples
            for feat_score in top_feats:
                feat, score = feat_score
                print(f"  {feat}: {score}")
        else:
            print(f"Error getting feature importance: {feat_imp.get('error')}")
    else:
        print("Error: prepared_data missing from training session. (API bug)")

    print_header("API: Train with subset of columns")
    subset_cols = ["koi_period", "koi_prad", "koi_teq", "koi_insol"]
    session_id2 = f"session-{uuid.uuid4().hex[:8]}"
    train_api.start_training_session(session_id2)
    train_api.load_data_for_training(
        session_id2,
        data_source="nasa",
        data_config={"datasets": ["kepler"], "target_column": "koi_disposition"}
    )
    train_api.configure_training(
        session_id2,
        training_config={
            "model_type": "random_forest",
            "target_column": "koi_disposition",
            "feature_columns": subset_cols
        }
    )
    train_api.start_training(session_id2)
    save_info2 = train_api.save_trained_model(session_id2, model_name="rf_kepler_subset_api")
    model_path2 = save_info2.get("model_path")
    print(f"Subset model saved to: {model_path2}")
    # Use ExoplanetMLAPI to get a robust sample for prediction with subset columns
    sample_data2 = user_api.get_sample_data('kepler', n_samples=1)
    if 'error' in sample_data2:
        print(f"Error getting sample data for subset model prediction: {sample_data2['error']}")
        sample2 = None
        true_label2 = None
    else:
        sample2 = sample_data2['sample_data'][0]
        # Remove target column if present
        for col in ['koi_disposition', 'tfopwg_disp', 'disposition']:
            true_label2 = sample2.pop(col, None)
    if sample2 is not None:
        # Only keep subset columns
        sample2_subset = {k: v for k, v in sample2.items() if k in subset_cols}
        load_result2 = pred_api.load_model(model_path2)
        if load_result2.get('status') != 'success':
            print(f"Error loading subset model for prediction: {load_result2.get('error')}")
        else:
            model_id2 = load_result2['model_id']
            pred2 = pred_api.predict_single(model_id2, sample2_subset)
            print(f"Sample true label: {true_label2}, predicted: {pred2.get('prediction', 'N/A')}")
    else:
        print("Skipping subset model prediction due to missing sample data.")

    print_header("API: Error handling and edge cases")
    try:
        pred_api.predict("/tmp/nonexistent_model.joblib", [sample])
        print("FAIL: No error when loading non-existent model")
    except Exception as e:
        print(f"PASS: Correctly failed to load non-existent model: {e}")
    try:
        wrong_sample = {"not_a_feature": 123}
        pred_api.predict(model_path, [wrong_sample])
        print("FAIL: No error when predicting with wrong features")
    except Exception as e:
        print(f"PASS: Correctly failed to predict with wrong features: {e}")
    session_id3 = f"session-{uuid.uuid4().hex[:8]}"
    train_api.start_training_session(session_id3)
    train_api.load_data_for_training(
        session_id3,
        data_source="nasa",
        data_config={"datasets": ["kepler"], "target_column": "koi_disposition"}
    )
    # Should fail at configure_training for empty feature columns
    result = train_api.configure_training(
        session_id3,
        training_config={
            "model_type": "random_forest",
            "target_column": "koi_disposition",
            "feature_columns": []
        }
    )
    if result.get('status') == 'error':
        print(f"PASS: Correctly failed to configure with empty feature columns: {result.get('error')}")
    else:
        print("FAIL: No error when configuring with empty feature columns")
    session_id4 = f"session-{uuid.uuid4().hex[:8]}"
    train_api.start_training_session(session_id4)
    # Should fail at load_data_for_training for wrong target column
    result2 = train_api.load_data_for_training(
        session_id4,
        data_source="nasa",
        data_config={"datasets": ["kepler"], "target_column": "not_a_column"}
    )
    if result2.get('status') == 'error':
        print(f"PASS: Correctly failed to load data with wrong target column: {result2.get('error')}")
    else:
        print("FAIL: No error when loading data with wrong target column")

if __name__ == "__main__":
    main()
