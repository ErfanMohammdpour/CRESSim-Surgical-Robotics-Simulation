#!/usr/bin/env python3
"""
Test script for checking pipeline structure
"""

import sys
from pathlib import Path

def test_structure():
    """Test pipeline structure"""
    print("🧪 Testing Pipeline Structure")
    print("=" * 35)
    
    # Check required directories
    required_dirs = [
        "src",
        "src/pipeline",
        "configs",
        "data"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
        else:
            print(f"✅ {dir_path}/ exists")
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    
    # Check required files
    required_files = [
        "src/pipeline/__init__.py",
        "src/pipeline/data_handler.py",
        "src/pipeline/trainer.py",
        "src/pipeline/evaluator.py",
        "src/pipeline/pipeline.py",
        "run_pipeline.py",
        "manage.py",
        "requirements.txt"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path} exists")
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("\n🎉 All required files and directories exist!")
    return True

def test_basic_imports():
    """Test basic imports without heavy dependencies"""
    print("\n🔧 Testing Basic Imports")
    print("=" * 30)
    
    try:
        # Test basic Python imports
        import json
        print("✅ json imported")
        
        import logging
        print("✅ logging imported")
        
        from pathlib import Path
        print("✅ pathlib imported")
        
        # Add src to path
        sys.path.insert(0, str(Path.cwd() / "src"))
        
        # Test if we can import the modules (even if they fail later)
        try:
            from src.pipeline import CompletePipeline
            print("✅ CompletePipeline can be imported")
        except ImportError as e:
            print(f"⚠️ CompletePipeline import failed: {e}")
            print("💡 This is expected if dependencies are missing")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic imports failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Pipeline Structure Test")
    print("=" * 30)
    
    # Test structure
    structure_success = test_structure()
    
    # Test basic imports
    import_success = test_basic_imports()
    
    if structure_success and import_success:
        print("\n🎉 Structure test passed!")
        print("✅ Pipeline structure is correct")
        print("\nTo install dependencies:")
        print("  pip install -r requirements.txt")
        print("\nThen run:")
        print("  python test_pipeline.py")
    else:
        print("\n❌ Structure test failed!")
        print("Please check the error messages above")
        sys.exit(1)

if __name__ == "__main__":
    main()
