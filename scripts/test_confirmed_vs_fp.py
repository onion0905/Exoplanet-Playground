"""
Test script: Only CONFIRMED vs FALSE POSITIVE training
"""
from ML.src.api.training_api import TrainingAPI
from ML.src.api.prediction_api import PredictionAPI
from ML.src.api.explanation_api import ExplanationAPI
from uuid import uuid4
import pandas as pd

sess = f"session-{uuid4().hex[:6]}"
tapi = TrainingAPI()

print("[1] Start training session...")
tapi.start_training_session(sess)

print("[2] Load Kepler data (should filter out CANDIDATE)...")
load_result = tapi.load_data_for_training(sess, data_source='nasa', data_config={'datasets':['kepler']})
print("   Data info:", load_result['data_info'])

print("[3] Configure training...")
config_result = tapi.configure_training(sess, {
    'model_type': 'random_forest',
    'target_column': 'koi_disposition'
})
print("   Config:", config_result)

print("[4] Start training...")
train_result = tapi.start_training(sess)
print("   Training metrics:", train_result.get('training_metrics'))
print("   Evaluation metrics:", train_result.get('evaluation_metrics'))

print("[5] Save model...")
save_result = tapi.save_trained_model(sess, model_name='rf_kepler_confirmed_fp')
print("   Model saved:", save_result)

print("[6] Test prediction on filtered data...")
papi = PredictionAPI()
model_path = save_result['model_path']
papi.load_model(model_path)

# Get a few samples from the filtered data
session_info = tapi.get_session_info(sess, include_data=True)
data = session_info['session_info']['data']
# Only use rows with CONFIRMED or FALSE POSITIVE
samples = data[data['koi_disposition'].isin(['CONFIRMED','FALSE POSITIVE'])].head(5)

for i, row in samples.iterrows():
    features = {k: v for k, v in row.items() if k != 'koi_disposition'}
    pred = papi.predict_single('rf_kepler_confirmed_fp', features)
    print(f"Sample {i}: true={row['koi_disposition']}, pred={pred['prediction']}, prob={pred.get('probability')}")
