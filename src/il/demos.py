"""
Demo generation for imperfect scripted demonstrations.
"""

import os
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging
import json
import h5py
from tqdm import tqdm

logger = logging.getLogger(__name__)


class MockUnityEnv:
    """Mock Unity environment for demo generation without Unity."""
    
    def __init__(self, image_size: int = 128):
        self.image_size = image_size
        self.reset()
    
    def reset(self) -> Tuple[np.ndarray, Dict]:
        """Reset environment and return initial observation."""
        # Generate random RGB image
        obs = np.random.randint(0, 255, (self.image_size, self.image_size, 3), dtype=np.uint8)
        
        # Add some structure to make it look more realistic
        # Add a circular "surgical area"
        center = (self.image_size // 2, self.image_size // 2)
        radius = self.image_size // 4
        cv2.circle(obs, center, radius, (100, 150, 200), -1)
        
        # Add some "tissue" texture
        noise = np.random.randint(0, 50, obs.shape, dtype=np.uint8)
        obs = np.clip(obs.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        info = {
            "liquid_mass_remaining": 100.0,
            "contaminant_mass_remaining": 50.0,
            "collisions": 0,
            "step": 0
        }
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Step environment with action."""
        # Generate next observation (slightly modified)
        obs, info = self.reset()
        
        # Add some variation based on action
        if action[4] > 0.5:  # suction toggle
            # Simulate suction effect
            obs = cv2.GaussianBlur(obs, (5, 5), 0)
        
        # Simulate progress
        info["liquid_mass_remaining"] = max(0, info["liquid_mass_remaining"] - np.random.uniform(0, 5))
        info["contaminant_mass_remaining"] = max(0, info["contaminant_mass_remaining"] - np.random.uniform(0, 3))
        info["step"] += 1
        
        # Calculate reward
        reward = self._calculate_reward(info)
        
        # Check if done
        done = info["step"] >= 100 or info["liquid_mass_remaining"] < 10
        
        return obs, reward, done, info
    
    def _calculate_reward(self, info: Dict) -> float:
        """Calculate reward based on environment state."""
        liquid_reduction = (100.0 - info["liquid_mass_remaining"]) / 100.0
        contaminant_reduction = (50.0 - info["contaminant_mass_remaining"]) / 50.0
        
        reward = liquid_reduction + contaminant_reduction - 0.01  # time penalty
        
        if info["collisions"] > 0:
            reward -= 0.1 * info["collisions"]
        
        return reward


class ScriptedPolicy:
    """Scripted policy for generating imperfect demonstrations."""
    
    def __init__(self, noise_levels: Dict[str, float]):
        self.noise_levels = noise_levels
        self.step_count = 0
    
    def reset(self):
        """Reset policy state."""
        self.step_count = 0
    
    def get_action(self, obs: np.ndarray, info: Dict) -> np.ndarray:
        """Get action from scripted policy with noise."""
        self.step_count += 1
        
        # Base action (move towards center, occasional suction)
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # [x, y, z, yaw, suction_toggle]
        
        # Move towards center of image
        center_x, center_y = obs.shape[1] // 2, obs.shape[0] // 2
        current_x, current_y = obs.shape[1] // 2 + np.random.randint(-20, 20), obs.shape[0] // 2 + np.random.randint(-20, 20)
        
        # Calculate movement direction
        dx = (center_x - current_x) / obs.shape[1]
        dy = (center_y - current_y) / obs.shape[0]
        
        action[0] = np.clip(dx, -1, 1)  # x movement
        action[1] = np.clip(dy, -1, 1)  # y movement
        action[2] = np.random.uniform(-0.5, 0.5)  # z movement
        action[3] = np.random.uniform(-30, 30)  # yaw rotation
        
        # Occasional suction
        if self.step_count % 10 == 0:
            action[4] = 1.0  # suction toggle
        else:
            action[4] = 0.0
        
        # Add noise
        action[0] += np.random.normal(0, self.noise_levels["position"])
        action[1] += np.random.normal(0, self.noise_levels["position"])
        action[2] += np.random.normal(0, self.noise_levels["position"])
        action[3] += np.random.normal(0, self.noise_levels["yaw"])
        
        # Random suction toggle with probability
        if np.random.random() < self.noise_levels["suction_toggle_prob"]:
            action[4] = 1.0 - action[4]
        
        # Clip actions to valid range
        action[0] = np.clip(action[0], -1, 1)
        action[1] = np.clip(action[1], -1, 1)
        action[2] = np.clip(action[2], -1, 1)
        action[3] = np.clip(action[3], -180, 180)
        action[4] = np.clip(action[4], 0, 1)
        
        return action


def generate_demos(output_dir: str, config: Dict[str, Any], mock: bool = True) -> None:
    """Generate imperfect scripted demonstrations."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    num_episodes = config["num_episodes"]
    
    # Extract noise levels from config
    noise_config = config.get("noise", {})
    noise_levels = {
        "position": noise_config.get("position_std", 0.01),
        "yaw": noise_config.get("rotation_std", 0.05) * 180 / np.pi,  # convert to degrees
        "suction_toggle_prob": noise_config.get("action_dropout", 0.1)
    }
    
    # Extract acceptance threshold
    filtering_config = config.get("filtering", {})
    acceptance_threshold = {
        "min_liquid_reduction": 0.01,  # very low for testing
        "max_collisions": 100  # very high for testing
    }
    
    logger.info(f"Generating {num_episodes} demo episodes...")
    
    # Initialize environment and policy
    if mock:
        env = MockUnityEnv(image_size=128)
    else:
        # TODO: Initialize real Unity environment
        raise NotImplementedError("Real Unity environment not implemented yet")
    
    policy = ScriptedPolicy(noise_levels)
    
    # Generate episodes
    episodes = []
    accepted_episodes = 0
    
    for episode_idx in tqdm(range(num_episodes), desc="Generating demos"):
        episode_data = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "infos": [],
            "episode_length": 0,
            "total_reward": 0.0,
            "final_liquid_mass": 0.0,
            "final_contaminant_mass": 0.0,
            "collisions": 0
        }
        
        obs, info = env.reset()
        policy.reset()
        
        episode_reward = 0.0
        step = 0
        
        max_steps = config.get("episode_length", 100)
        
        while step < max_steps:
            action = policy.get_action(obs, info)
            next_obs, reward, done, next_info = env.step(action)
            
            # Store data
            episode_data["observations"].append(obs.copy())
            episode_data["actions"].append(action.copy())
            episode_data["rewards"].append(reward)
            episode_data["infos"].append(info.copy())
            
            episode_reward += reward
            step += 1
            
            obs, info = next_obs, next_info
            
            if done:
                break
        
        # Calculate episode statistics
        episode_data["episode_length"] = step
        episode_data["total_reward"] = episode_reward
        episode_data["final_liquid_mass"] = info["liquid_mass_remaining"]
        episode_data["final_contaminant_mass"] = info["contaminant_mass_remaining"]
        episode_data["collisions"] = info["collisions"]
        
        # Check acceptance criteria
        liquid_reduction = (100.0 - info["liquid_mass_remaining"]) / 100.0
        contaminant_reduction = (50.0 - info["contaminant_mass_remaining"]) / 50.0
        
        if (liquid_reduction >= acceptance_threshold["min_liquid_reduction"] and
            info["collisions"] <= acceptance_threshold["max_collisions"]):
            episodes.append(episode_data)
            accepted_episodes += 1
            logger.info(f"Episode {episode_idx + 1}: Accepted (liquid: {liquid_reduction:.2f}, collisions: {info['collisions']})")
        else:
            logger.info(f"Episode {episode_idx + 1}: Rejected (liquid: {liquid_reduction:.2f}, collisions: {info['collisions']})")
    
    # Save episodes
    logger.info(f"Accepted {accepted_episodes}/{num_episodes} episodes")
    
    if accepted_episodes > 0:
        # Save as HDF5
        h5_path = output_path / "demos.h5"
        with h5py.File(h5_path, 'w') as f:
            for i, episode in enumerate(episodes):
                ep_group = f.create_group(f"episode_{i}")
                ep_group.create_dataset("observations", data=np.array(episode["observations"]))
                ep_group.create_dataset("actions", data=np.array(episode["actions"]))
                ep_group.create_dataset("rewards", data=np.array(episode["rewards"]))
                ep_group.attrs["episode_length"] = episode["episode_length"]
                ep_group.attrs["total_reward"] = episode["total_reward"]
                ep_group.attrs["final_liquid_mass"] = episode["final_liquid_mass"]
                ep_group.attrs["final_contaminant_mass"] = episode["final_contaminant_mass"]
                ep_group.attrs["collisions"] = episode["collisions"]
        
        # Save metadata
        metadata = {
            "num_episodes": accepted_episodes,
            "total_episodes_generated": num_episodes,
            "config": config,
            "generation_timestamp": str(Path().cwd())
        }
        
        with open(output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Demos saved to {h5_path}")
        logger.info(f"Metadata saved to {output_path / 'metadata.json'}")
    else:
        logger.warning("No episodes were accepted. Consider adjusting acceptance criteria.")


if __name__ == "__main__":
    # Test demo generation
    config = {
        "num_episodes": 5,
        "noise_levels": {
            "position": 0.01,
            "yaw": 5.0,
            "suction_toggle_prob": 0.1
        },
        "acceptance_threshold": {
            "min_liquid_reduction": 0.5,
            "max_collisions": 5
        }
    }
    
    generate_demos("data/demos", config, mock=True)