#!/usr/bin/env python3
"""
Simple script to run the complete surgical robotics pipeline
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Run the complete pipeline"""
    print("🚀 Complete Surgical Robotics Pipeline")
    print("=" * 50)
    
    # Check if src/pipeline exists
    if not Path("src/pipeline").exists():
        print("❌ Pipeline modules not found!")
        print("Please ensure src/pipeline/ directory exists with all modules")
        return
    
    # Run the pipeline
    try:
        print("📥 Downloading dataset...")
        print("🔄 Processing and splitting data...")
        print("📚 Training IL model...")
        print("🤖 Training RL model...")
        print("📊 Evaluating models...")
        print("🧹 Cleaning up...")
        
        result = subprocess.run([
            sys.executable, "-c", """
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))

from src.pipeline import CompletePipeline

# Create and run pipeline
pipeline = CompletePipeline()
success = pipeline.run_complete_pipeline(il_epochs=50, rl_timesteps=50000)

if success:
    print('\\n🎉 Pipeline completed successfully!')
    print('📁 Results saved in data/results/')
    print('🧹 Temporary files cleaned up')
else:
    print('\\n❌ Pipeline failed!')
    sys.exit(1)
"""
        ], check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        return
    except KeyboardInterrupt:
        print("\n⏹️ Pipeline interrupted by user")
        return

if __name__ == "__main__":
    main()
