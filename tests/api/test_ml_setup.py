#!/usr/bin/env python3
"""
Simple ML API Test Script

This script provides a straightforward way to test the core ML functionality.
It imports from the new ML structure and tests basic operations.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all ML modules can be imported successfully."""
    
    print("Testing imports from ML structure...")
    
    try:
        # Test core API imports
        from ML.src.api.user_api import ExoplanetMLAPI
        print("✅ ExoplanetMLAPI imported successfully")
        
        from ML.src.api.training_api import TrainingAPI
        print("✅ TrainingAPI imported successfully")
        
        from ML.src.api.prediction_api import PredictionAPI
        print("✅ PredictionAPI imported successfully")
        
        from ML.src.api.explanation_api import ExplanationAPI
        print("✅ ExplanationAPI imported successfully")
        
        # Test utility imports
        from ML.src.utils.model_factory import ModelFactory
        print("✅ ModelFactory imported successfully")
        
        from ML.src.data.data_loader import DataLoader
        print("✅ DataLoader imported successfully")
        
        from ML.src.data.data_processor import DataProcessor
        print("✅ DataProcessor imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during import: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality of the main API."""
    
    print("\nTesting basic functionality...")
    
    try:
        from ML.src.api.user_api import ExoplanetMLAPI
        
        # Create API instance
        api = ExoplanetMLAPI()
        print("✅ API instance created successfully")
        
        # Test basic methods
        datasets = api.list_available_datasets()
        print(f"✅ Found {len(datasets)} available datasets: {datasets}")
        
        models = api.list_available_models()
        print(f"✅ Found {len(models)} available model types: {models}")
        
        trained_models = api.list_trained_models()
        print(f"✅ Found {len(trained_models)} trained models")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False


def test_configuration():
    """Test configuration and paths."""
    
    print("\nTesting configuration...")
    
    try:
        from ML.src import config
        
        print(f"✅ Config loaded successfully")
        print(f"   BASE_DIR: {config.BASE_DIR}")
        print(f"   DATA_DIR: {config.DATA_DIR}")
        print(f"   MODELS_DIR: {config.MODELS_DIR}")
        print(f"   SRC_DIR: {config.SRC_DIR}")
        
        # Check if directories exist
        data_exists = config.DATA_DIR.exists()
        models_exists = config.MODELS_DIR.exists()
        
        print(f"   Data directory exists: {data_exists}")
        print(f"   Models directory exists: {models_exists}")
        
        if data_exists:
            csv_files = list(config.DATA_DIR.glob("*.csv"))
            print(f"   Found {len(csv_files)} CSV files in data directory")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def main():
    """Run all tests."""
    
    print("=" * 50)
    print("🧪 ML API TESTING SUITE")
    print("=" * 50)
    
    success_count = 0
    total_tests = 3
    
    # Test 1: Imports
    if test_imports():
        success_count += 1
    
    # Test 2: Basic functionality  
    if test_basic_functionality():
        success_count += 1
        
    # Test 3: Configuration
    if test_configuration():
        success_count += 1
    
    print("\n" + "=" * 50)
    print(f"📊 TEST RESULTS: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 All tests passed! ML API is ready to use.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    print("=" * 50)


if __name__ == "__main__":
    main()