"""
Evaluation module for surgical robotics pipeline
Handles model evaluation and result generation
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Import project modules
from src.vision.encoders import CNNEncoder
from src.il.bc_trainer import PolicyNetwork
from src.envs.mock_env import MockSuctionEnv
from .trainer import SurgicalDataset

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Handles model evaluation and result generation"""
    
    def __init__(self, results_dir: str = "data/results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("ModelEvaluator initialized")
    
    def evaluate_models(self, il_model_path: Optional[Path] = None, 
                       rl_model_path: Optional[Path] = None,
                       val_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Evaluate both IL and RL models on validation set"""
        logger.info("📊 Evaluating models on validation set...")
        
        results = {}
        
        # Evaluate IL model
        if il_model_path and il_model_path.exists():
            il_results = self._evaluate_il_model(il_model_path, val_dir)
            results['IL'] = il_results
        
        # Evaluate RL model
        if rl_model_path and rl_model_path.exists():
            rl_results = self._evaluate_rl_model(rl_model_path)
            results['RL'] = rl_results
        
        # Save results
        self._save_evaluation_results(results)
        
        return results
    
    def _evaluate_il_model(self, model_path: Path, val_dir: Path) -> Dict[str, Any]:
        """Evaluate IL model"""
        logger.info("Evaluating IL model...")
        
        # Load model
        checkpoint = torch.load(model_path, map_location='cpu')
        
        encoder = CNNEncoder(
            input_channels=3,
            output_dim=512,
            channels=[32, 64, 128, 256],
            kernel_sizes=[3, 3, 3, 3],
            strides=[2, 2, 2, 2],
            padding=[1, 1, 1, 1]
        )
        
        model = PolicyNetwork(encoder=encoder, hidden_dim=512, action_dim=5)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Create validation dataset
        val_dataset = SurgicalDataset(val_dir, image_size=(128, 128))
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Evaluate
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for images, actions in val_dataloader:
                pred_actions = model(images)
                loss = nn.MSELoss()(pred_actions, actions)
                total_loss += loss.item()
                
                # Calculate accuracy (simplified)
                pred_classes = (pred_actions > 0.5).float()
                true_classes = (actions > 0.5).float()
                correct_predictions += (pred_classes == true_classes).sum().item()
                total_predictions += actions.numel()
        
        accuracy = correct_predictions / total_predictions
        avg_loss = total_loss / len(val_dataloader)
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'total_samples': len(val_dataset)
        }
    
    def _evaluate_rl_model(self, model_path: Path) -> Dict[str, Any]:
        """Evaluate RL model"""
        logger.info("Evaluating RL model...")
        
        # Load model
        from stable_baselines3 import PPO
        model = PPO.load(str(model_path))
        
        # Create environment
        env = MockSuctionEnv(image_size=(128, 128), max_steps=1000)
        
        # Run evaluation episodes
        num_episodes = 20
        episode_rewards = []
        success_count = 0
        
        for episode in range(num_episodes):
            obs, info = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            episode_rewards.append(episode_reward)
            
            # Check success
            liquid_reduction = info.get('liquid_reduction', 0)
            contaminant_reduction = info.get('contaminant_reduction', 0)
            if liquid_reduction > 0.8 and contaminant_reduction > 0.8:
                success_count += 1
        
        success_rate = success_count / num_episodes
        mean_reward = np.mean(episode_rewards)
        
        return {
            'success_rate': success_rate,
            'mean_reward': mean_reward,
            'episode_rewards': episode_rewards,
            'num_episodes': num_episodes
        }
    
    def _save_evaluation_results(self, results: Dict[str, Any]):
        """Save evaluation results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed results
        results_file = self.results_dir / f"evaluation_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create summary report
        self._create_summary_report(results, timestamp)
        
        logger.info(f"✅ Results saved: {results_file}")
    
    def _create_summary_report(self, results: Dict[str, Any], timestamp: str):
        """Create summary report"""
        report_file = self.results_dir / f"summary_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("SURGICAL ROBOTICS EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
            
            for model_name, model_results in results.items():
                f.write(f"{model_name} MODEL RESULTS:\n")
                f.write("-" * 30 + "\n")
                
                if model_name == 'IL':
                    f.write(f"Accuracy: {model_results['accuracy']:.4f}\n")
                    f.write(f"Loss: {model_results['loss']:.6f}\n")
                    f.write(f"Total Samples: {model_results['total_samples']}\n")
                elif model_name == 'RL':
                    f.write(f"Success Rate: {model_results['success_rate']:.2%}\n")
                    f.write(f"Mean Reward: {model_results['mean_reward']:.2f}\n")
                    f.write(f"Episodes: {model_results['num_episodes']}\n")
                
                f.write("\n")
        
        logger.info(f"📋 Summary report saved: {report_file}")
    
    def print_results(self, results: Dict[str, Any]):
        """Print evaluation results"""
        print("\n" + "=" * 60)
        print("📊 SURGICAL ROBOTICS EVALUATION RESULTS")
        print("=" * 60)
        
        for model_name, model_results in results.items():
            print(f"\n{model_name} MODEL:")
            print("-" * 30)
            
            if model_name == 'IL':
                print(f"✅ Accuracy: {model_results['accuracy']:.2%}")
                print(f"📉 Loss: {model_results['loss']:.6f}")
                print(f"📊 Samples: {model_results['total_samples']}")
            elif model_name == 'RL':
                print(f"🎯 Success Rate: {model_results['success_rate']:.2%}")
                print(f"💰 Mean Reward: {model_results['mean_reward']:.2f}")
                print(f"📈 Episodes: {model_results['num_episodes']}")
        
        print("\n" + "=" * 60)
    
    def get_results_dir(self) -> Path:
        """Get results directory path"""
        return self.results_dir
