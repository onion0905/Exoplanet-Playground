"""
Comprehensive test for Exoplanet Playground ML APIs:
- Train a model and save to ML/models/user
- Load a model and predict exoplanet status
- Load a model and evaluate feature importance
- Train with a subset of columns
- Load a subset-trained model and predict
"""

import os
import sys
import random
import numpy as np

# Make ML/src available for direct script execution
ML_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '../ML/src'))
if ML_SRC not in sys.path:
    sys.path.insert(0, ML_SRC)

try:
    from api.training_api import TrainingAPI
    from api.prediction_api import PredictionAPI
    from api.explanation_api import ExplanationAPI
except ImportError as e:
    print("ImportError: {}".format(e))
    print("\nCould not import ML API modules. Make sure ML/src is present and this script is run from the project root.")
    sys.exit(1)

def print_header(msg):
    print("\n" + "="*40)
    print(msg)
    print("="*40)

def main():
    # 6. Try loading a non-existent model
    print_header("6. Error handling: load non-existent model")
    try:
        pred_api.predict("/tmp/nonexistent_model.joblib", [sample])
        print("FAIL: No error when loading non-existent model")
    except Exception as e:
        print(f"PASS: Correctly failed to load non-existent model: {e}")

    # 7. Try predicting with wrong feature set
    print_header("7. Error handling: predict with wrong features")
    try:
        wrong_sample = {"not_a_feature": 123}
        pred_api.predict(model_path, [wrong_sample])
        print("FAIL: No error when predicting with wrong features")
    except Exception as e:
        print(f"PASS: Correctly failed to predict with wrong features: {e}")

    # 8. Test model metadata file
    print_header("8. Model metadata file")
    if os.path.exists(metadata_path):
        import json
        with open(metadata_path) as f:
            meta = json.load(f)
        if "model_type" in meta and "training_config" in meta:
            print("PASS: Metadata file contains expected keys.")
        else:
            print("FAIL: Metadata file missing expected keys.")
    else:
        print("FAIL: Metadata file not found.")

    # 9. Edge case: configure with empty feature columns
    print_header("9. Edge case: empty feature columns")
    session_id3 = f"session-{uuid.uuid4().hex[:8]}"
    train_api.start_training_session(session_id3)
    train_api.load_data_for_training(
        session_id3,
        data_source="nasa",
        data_config={"datasets": ["kepler"], "target_column": "koi_disposition"}
    )
    try:
        train_api.configure_training(
            session_id3,
            training_config={
                "model_type": "random_forest",
                "target_column": "koi_disposition",
                "feature_columns": []
            }
        )
        train_api.start_training(session_id3)
        print("FAIL: No error when training with empty feature columns")
    except Exception as e:
        print(f"PASS: Correctly failed to train with empty feature columns: {e}")

    # 10. Edge case: wrong target column
    print_header("10. Edge case: wrong target column")
    session_id4 = f"session-{uuid.uuid4().hex[:8]}"
    train_api.start_training_session(session_id4)
    train_api.load_data_for_training(
        session_id4,
        data_source="nasa",
        data_config={"datasets": ["kepler"], "target_column": "not_a_column"}
    )
    try:
        train_api.configure_training(
            session_id4,
            training_config={
                "model_type": "random_forest",
                "target_column": "not_a_column"
            }
        )
        train_api.start_training(session_id4)
        print("FAIL: No error when training with wrong target column")
    except Exception as e:
        print(f"PASS: Correctly failed to train with wrong target column: {e}")
    # 1. Train a model and save to ML/models/user
    print_header("1. Train a model and save it")
    train_api = TrainingAPI()
    import uuid
    session_id = f"session-{uuid.uuid4().hex[:8]}"
    # Load data (kepler)
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
    save_info = train_api.save_trained_model(session_id, model_name="rf_kepler_test")
    model_path = save_info.get("model_path")
    metadata_path = save_info.get("metadata_path")
    print(f"Model saved to: {model_path}\nMetadata: {metadata_path}")

    # 2. Load a model and predict
    print_header("2. Load model and predict")
    pred_api = PredictionAPI()
    # Use filtered data from session for a test sample
    session_info = train_api.get_session_info(session_id, include_data=True)
    prepared_data = session_info["session_info"].get("prepared_data")
    if prepared_data is None:
        raise RuntimeError("No prepared_data found in session_info. Check training workflow.")
    X = prepared_data["X_test"]
    y = prepared_data["y_test"]
    idx = random.randint(0, len(X)-1)
    sample = X[idx]
    true_label = y[idx]
    pred = pred_api.predict(model_path, [sample])
    print(f"Sample true label: {true_label}, predicted: {pred['predictions'][0]}")

    # 3. Load a model and evaluate feature importance
    print_header("3. Feature importance")
    expl_api = ExplanationAPI()
    feat_imp = expl_api.get_feature_importance(model_path)
    print("Feature importances:")
    for feat, imp in zip(feat_imp["features"], feat_imp["importances"]):
        print(f"  {feat}: {imp:.4f}")

    # 4. Train with a subset of columns
    print_header("4. Train with subset of columns")
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
    save_info2 = train_api.save_trained_model(session_id2, model_name="rf_kepler_subset_test")
    model_path2 = save_info2.get("model_path")
    print(f"Subset model saved to: {model_path2}")

    # 5. Load subset-trained model and predict
    print_header("5. Predict with subset-trained model")
    session_info2 = train_api.get_session_info(session_id2, include_data=True)
    data2 = session_info2["session_info"]["prepared_data"]
    X2 = data2["X_test"]
    y2 = data2["y_test"]
    idx2 = random.randint(0, len(X2)-1)
    sample2 = X2[idx2]
    true_label2 = y2[idx2]
    pred2 = pred_api.predict(model_path2, [sample2])
    print(f"Sample true label: {true_label2}, predicted: {pred2['predictions'][0]}")

if __name__ == "__main__":
    main()
