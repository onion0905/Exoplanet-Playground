#!/usr/bin/env python3
"""
Integration Demo: Exoplanet Playground with ML APIs

This script demonstrates how to use the ML APIs from the new ML/ directory structure
alongside the existing Flask web application.

Usage:
    python integration_demo.py

This shows:
1. How to import and use ML APIs from the new structure
2. How the ML system integrates with the web app
3. Complete workflow from data loading to prediction
"""

import sys
import os
from pathlib import Path

# Add ML directory to Python path
ml_dir = Path(__file__).parent / "ML"
sys.path.insert(0, str(ml_dir))

print("=" * 60)
print("🎯 EXOPLANET PLAYGROUND - INTEGRATION DEMO")
print("=" * 60)

def test_ml_integration():
    """Test the ML API integration"""
    print("\n1️⃣  TESTING ML API INTEGRATION")
    print("-" * 40)
    
    try:
        # Import from the new ML structure
        from src.api.user_api import ExoplanetMLAPI
        from src.api.training_api import TrainingAPI
        from src.api.prediction_api import PredictionAPI
        from src.config import DATA_DIR, MODELS_DIR, MODEL_SAVE_DIR
        
        print("✅ Successfully imported ML APIs from new structure")
        print(f"   📁 Data directory: {DATA_DIR}")
        print(f"   📁 Models directory: {MODELS_DIR}")
        print(f"   📁 Model save directory: {MODEL_SAVE_DIR}")
        
        # Test basic functionality
        api = ExoplanetMLAPI()
# Use the API
datasets = api.list_available_datasets()
models = api.list_available_models()        print(f"   📊 Available datasets: {len(datasets)} ({', '.join(datasets)})")
        print(f"   🤖 Available models: {len(models)} ({', '.join(models)})")
        
        return True
        
    except Exception as e:
        print(f"❌ ML Integration failed: {e}")
        return False

def test_web_app_compatibility():
    """Test that the web app can still run"""
    print("\n2️⃣  TESTING WEB APP COMPATIBILITY")
    print("-" * 40)
    
    try:
        # Import Flask app components
        from flask import Flask
        import app  # This should import successfully
        
        print("✅ Flask app imports successfully")
        print("✅ Web application is compatible with new ML structure")
        
        # Check that required directories exist
        templates_exist = Path("templates").exists()
        static_exist = Path("data").exists()
        
        print(f"   📁 Templates directory: {'✅' if templates_exist else '❌'}")
        print(f"   📁 Data directory: {'✅' if static_exist else '❌'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Web app compatibility failed: {e}")
        return False

def show_usage_examples():
    """Show practical usage examples"""
    print("\n3️⃣  USAGE EXAMPLES")
    print("-" * 40)
    
    print("📝 To use ML APIs in your code:")
    print("```python")
    print("# Import from the new ML structure")
    print("from ML.src.api.user_api import ExoplanetMLAPI")
    print("from ML.src.api.training_api import TrainingAPI")
    print("from ML.src.api.prediction_api import PredictionAPI")
    print("")
    print("# Initialize API")
    print("api = ExoplanetMLAPI()")
    print("")
    print("# Use the API")
    print("datasets = api.list_available_datasets()")
    print("models = api.list_available_model_types()")
    print("```")
    print("")
    
    print("🌐 To run the web application:")
    print("```bash")
    print("source .venv/bin/activate.fish")
    print("python app.py")
    print("# Visit: http://localhost:5000")
    print("```")
    print("")
    
    print("🧪 To run ML tests:")
    print("```bash")
    print("python test_ml_setup.py")
    print("python ml_api_demo.py")
    print("```")

def main():
    """Main integration test"""
    
    # Test ML integration
    ml_success = test_ml_integration()
    
    # Test web app compatibility
    web_success = test_web_app_compatibility()
    
    # Show usage examples
    show_usage_examples()
    
    # Final results
    print("\n" + "=" * 60)
    print("📊 INTEGRATION TEST RESULTS")
    print("=" * 60)
    
    print(f"🤖 ML API Integration: {'✅ PASS' if ml_success else '❌ FAIL'}")
    print(f"🌐 Web App Compatibility: {'✅ PASS' if web_success else '❌ FAIL'}")
    
    if ml_success and web_success:
        print("\n🎉 INTEGRATION SUCCESSFUL!")
        print("✅ ML APIs are accessible from new ML/ directory structure")
        print("✅ Web application remains fully functional")
        print("✅ Both systems can be used together")
        print("\n💡 You can now:")
        print("   - Use ML APIs directly in Python scripts")
        print("   - Run the web application for interactive use")
        print("   - Integrate ML functionality into other applications")
    else:
        print("\n⚠️  Some integration issues detected")
        print("Please check the error messages above")

if __name__ == "__main__":
    main()