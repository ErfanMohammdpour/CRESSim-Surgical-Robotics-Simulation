#!/usr/bin/env python3
"""
Test script for the modular surgical robotics pipeline
"""

import sys
from pathlib import Path

def test_imports():
    """Test module imports"""
    print("🧪 Testing Module Imports")
    print("=" * 30)
    
    try:
        # Add src to path
        sys.path.insert(0, str(Path.cwd() / "src"))
        
        # Test individual imports
        from src.pipeline.data_handler import DataHandler
        print("✅ DataHandler imported")
        
        from src.pipeline.trainer import ModelTrainer, SurgicalDataset
        print("✅ ModelTrainer and SurgicalDataset imported")
        
        from src.pipeline.evaluator import ModelEvaluator
        print("✅ ModelEvaluator imported")
        
        from src.pipeline.pipeline import CompletePipeline
        print("✅ CompletePipeline imported")
        
        # Test package import
        from src.pipeline import CompletePipeline, DataHandler, ModelTrainer, ModelEvaluator
        print("✅ Package imports successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_instantiation():
    """Test module instantiation"""
    print("\n🔧 Testing Module Instantiation")
    print("=" * 35)
    
    try:
        from src.pipeline import CompletePipeline, DataHandler, ModelTrainer, ModelEvaluator
        
        # Test DataHandler
        data_handler = DataHandler()
        print("✅ DataHandler instantiated")
        
        # Test ModelTrainer
        trainer = ModelTrainer()
        print("✅ ModelTrainer instantiated")
        
        # Test ModelEvaluator
        evaluator = ModelEvaluator()
        print("✅ ModelEvaluator instantiated")
        
        # Test CompletePipeline
        pipeline = CompletePipeline()
        print("✅ CompletePipeline instantiated")
        
        return True
        
    except Exception as e:
        print(f"❌ Instantiation failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Modular Pipeline Component Test")
    print("=" * 40)
    
    # Test imports
    import_success = test_imports()
    
    # Test instantiation
    instantiation_success = test_instantiation()
    
    if import_success and instantiation_success:
        print("\n🎉 All tests passed!")
        print("✅ Modular pipeline is ready to use")
        print("\nTo run the complete pipeline:")
        print("  python run_pipeline.py")
    else:
        print("\n❌ Some tests failed!")
        print("Please check the error messages above")
        sys.exit(1)

if __name__ == "__main__":
    main()
