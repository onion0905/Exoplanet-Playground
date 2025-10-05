# Simple test to verify three-class training integration
from ML.src.api.training_api import TrainingAPI
import pandas as pd

api = TrainingAPI()
session_id = 'integration_test'

try:
    print('Testing complete three-class integration...')
    
    # 1. Start session
    api.start_training_session(session_id)
    print('✅ Session started')
    
    # 2. Load data (kepler dataset)
    data_config = {'datasets': ['kepler'], 'target_column': 'koi_disposition'}
    load_result = api.load_data_for_training(session_id, 'nasa', data_config)
    print(f'✅ Data loaded: {load_result.get("status", load_result)}')
    
    # 3. Configure training
    training_config = {
        'model_type': 'random_forest',
        'target_column': 'koi_disposition',
        'hyperparameters': {'n_estimators': 5, 'max_depth': 3}
    }
    config_result = api.configure_training(session_id, training_config)
    print(f'✅ Training configured: {config_result.get("status", config_result)}')
    
    # 4. Start training
    training_result = api.start_training(session_id)
    print(f'✅ Training completed: {training_result.get("status", training_result)}')
    
    # 5. Validate results
    session_info = api.get_session_info(session_id, include_data=True)
    if 'prepared_data' in session_info['session_info']:
        prepared_data = session_info['session_info']['prepared_data']
        target_classes = sorted(prepared_data['y_test'].unique())
        print(f'✅ Target classes confirmed: {target_classes}')
        
        # Check if we have all three classes
        expected_classes = ['candidate', 'false_positive', 'planet']
        if all(cls in target_classes for cls in expected_classes):
            print('🎉 SUCCESS: Complete three-class system working!')
        else:
            print(f'⚠️  Warning: Expected {expected_classes}, got {target_classes}')
    else:
        print('❌ No prepared data found')

except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()