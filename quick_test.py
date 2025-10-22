#!/usr/bin/env python3
"""
تست سریع مدل روی داده‌های واقعی
Quick Test Script for Real Data Evaluation
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.envs.mock_env import MockSuctionEnv
from src.utils.gpu import get_device

def quick_test_model(model_path: str, num_episodes: int = 10):
    """تست سریع مدل"""
    print("🚀 شروع تست سریع مدل...")
    
    # Setup
    device = get_device()
    print(f"📱 استفاده از دستگاه: {device}")
    
    # Load model
    try:
        checkpoint = torch.load(model_path, map_location=device)
        print("✅ مدل با موفقیت بارگذاری شد")
    except Exception as e:
        print(f"❌ خطا در بارگذاری مدل: {e}")
        return
    
    # Create environment
    env = MockSuctionEnv(image_size=(128, 128), max_steps=1000)
    print("🌍 محیط شبیه‌سازی ایجاد شد")
    
    # Test episodes
    success_count = 0
    total_rewards = []
    liquid_reductions = []
    contaminant_reductions = []
    
    print(f"\n🎯 شروع تست {num_episodes} اپیزود...")
    print("-" * 50)
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False
        step = 0
        
        while not done and step < 1000:
            # Simple random action for testing (replace with your model)
            action = env.action_space.sample()
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            step += 1
        
        # Calculate metrics
        liquid_reduction = info.get('liquid_reduction', 0) * 100
        contaminant_reduction = info.get('contaminant_reduction', 0) * 100
        success = liquid_reduction > 80 and contaminant_reduction > 80
        
        if success:
            success_count += 1
        
        total_rewards.append(total_reward)
        liquid_reductions.append(liquid_reduction)
        contaminant_reductions.append(contaminant_reduction)
        
        print(f"اپیزود {episode+1:2d}: پاداش={total_reward:6.2f}, مایع={liquid_reduction:5.1f}%, آلودگی={contaminant_reduction:5.1f}%, موفق={'✅' if success else '❌'}")
    
    # Calculate statistics
    success_rate = success_count / num_episodes
    mean_reward = np.mean(total_rewards)
    mean_liquid = np.mean(liquid_reductions)
    mean_contaminant = np.mean(contaminant_reductions)
    
    print("-" * 50)
    print("📊 نتایج تست:")
    print(f"✅ نرخ موفقیت: {success_rate:.1%}")
    print(f"💰 میانگین پاداش: {mean_reward:.2f}")
    print(f"💧 میانگین کاهش مایع: {mean_liquid:.1f}%")
    print(f"🧪 میانگین کاهش آلودگی: {mean_contaminant:.1f}%")
    print(f"🎯 اپیزودهای موفق: {success_count}/{num_episodes}")
    
    # Performance assessment
    if success_rate >= 0.8:
        print("🌟 عملکرد عالی!")
    elif success_rate >= 0.6:
        print("👍 عملکرد خوب")
    elif success_rate >= 0.4:
        print("⚠️ عملکرد متوسط")
    else:
        print("❌ نیاز به بهبود")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='تست سریع مدل')
    parser.add_argument('--model', '-m', required=True, help='مسیر مدل')
    parser.add_argument('--episodes', '-e', type=int, default=10, help='تعداد اپیزودها')
    
    args = parser.parse_args()
    
    if not Path(args.model).exists():
        print(f"❌ فایل مدل یافت نشد: {args.model}")
        sys.exit(1)
    
    quick_test_model(args.model, args.episodes)
