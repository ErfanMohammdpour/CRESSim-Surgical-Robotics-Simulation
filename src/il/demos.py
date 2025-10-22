"""
Demo generation utilities.
"""

import numpy as np
import h5py
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import logging
from tqdm import tqdm
import json
from sklearn.mixture import GaussianMixture

from ..envs.mock_env import MockSuctionEnv
from ..utils.seeding import set_seed

logger = logging.getLogger(__name__)


class ScriptedPolicy:
    """Scripted policy for generating demonstrations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.noise_level = config.get('noise_level', 0.1)
        self.suction_threshold = config.get('suction_threshold', 0.5)
        self.movement_scale = config.get('movement_scale', 0.05)
        self.suction_probability = config.get('suction_probability', 0.3)
        
        logger.info(f"ScriptedPolicy initialized: noise={self.noise_level}")
    
    def get_action(self, obs: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        """Get action from scripted policy."""
        # Simple heuristic: move towards center and toggle suction
        center_x, center_y = obs.shape[1] // 2, obs.shape[0] // 2
        
        # Calculate movement towards center
        dx = np.random.uniform(-self.movement_scale, self.movement_scale)
        dy = np.random.uniform(-self.movement_scale, self.movement_scale)
        dz = np.random.uniform(-self.movement_scale * 0.5, self.movement_scale * 0.5)
        dyaw = np.random.uniform(-0.1, 0.1)
        
        # Add noise
        if np.random.random() < self.noise_level:
            dx += np.random.normal(0, self.movement_scale * 0.5)
            dy += np.random.normal(0, self.movement_scale * 0.5)
        
        # Suction decision
        liquid_mass = info.get('liquid_mass', 0)
        suction_toggle = 1.0 if liquid_mass > 10 and np.random.random() < self.suction_probability else 0.0
        
        return np.array([dx, dy, dz, dyaw, suction_toggle], dtype=np.float32)


class DemoGenerator:
    """Generate demonstrations using scripted policy."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scripted_policy = ScriptedPolicy(config.get('scripted_policy', {}))
        
        # Environment parameters
        self.image_size = tuple(config.get('image_size', [128, 128]))
        self.episode_length = config.get('episode_length', 1000)
        
        logger.info(f"DemoGenerator initialized: {self.image_size}")
    
    def generate_episode(self, env: MockSuctionEnv, episode_id: int) -> Dict[str, Any]:
        """Generate single episode."""
        obs, info = env.reset()
        
        observations = []
        actions = []
        rewards = []
        infos = []
        safety_flags = []
        
        for step in range(self.episode_length):
            # Get action from scripted policy
            action = self.scripted_policy.get_action(obs, info)
            
            # Step environment
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            
            # Store data
            observations.append(obs.copy())
            actions.append(action.copy())
            rewards.append(reward)
            infos.append(info.copy())
            safety_flags.append(not info.get('safety_violations', 0) > 0)
            
            # Update state
            obs = next_obs
            info = next_info
            
            # Check termination
            if terminated or truncated:
                break
        
        return {
            'episode_id': episode_id,
            'observations': np.array(observations),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'infos': infos,
            'safety_flags': np.array(safety_flags),
            'length': len(observations)
        }
    
    def generate_demos(
        self,
        num_episodes: int,
        output_dir: Path,
        mock: bool = True
    ) -> None:
        """Generate demonstration episodes."""
        logger.info(f"Generating {num_episodes} demonstration episodes")
        
        # Create environment
        if mock:
            env = MockSuctionEnv(
                image_size=self.image_size,
                max_steps=self.episode_length
            )
        else:
            # TODO: Implement Unity environment
            raise NotImplementedError("Unity environment not implemented")
        
        # Generate episodes
        episodes = []
        for episode_id in tqdm(range(num_episodes), desc="Generating demos"):
            set_seed(episode_id)  # Ensure reproducibility
            
            episode = self.generate_episode(env, episode_id)
            episodes.append(episode)
        
        # Save episodes
        self._save_episodes(episodes, output_dir)
        
        # Generate weights
        self._generate_weights(episodes, output_dir)
        
        # Generate summary
        self._generate_summary(episodes, output_dir)
        
        logger.info(f"Generated {len(episodes)} episodes")
    
    def _save_episodes(self, episodes: List[Dict[str, Any]], output_dir: Path):
        """Save episodes to HDF5 file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        demos_file = output_dir / "demos.h5"
        
        with h5py.File(demos_file, 'w') as f:
            for episode in episodes:
                episode_key = f"episode_{episode['episode_id']:03d}"
                episode_group = f.create_group(episode_key)
                
                episode_group.create_dataset('observations', data=episode['observations'])
                episode_group.create_dataset('actions', data=episode['actions'])
                episode_group.create_dataset('rewards', data=episode['rewards'])
                episode_group.create_dataset('safety_flags', data=episode['safety_flags'])
                episode_group.attrs['length'] = episode['length']
        
        logger.info(f"Saved episodes to {demos_file}")
    
    def _generate_weights(self, episodes: List[Dict[str, Any]], output_dir: Path):
        """Generate quality weights for episodes."""
        logger.info("Generating quality weights")
        
        # Extract features for weighting
        features = []
        for episode in episodes:
            # Calculate episode-level features
            liquid_reduction = episode['infos'][-1].get('liquid_reduction', 0)
            contaminant_reduction = episode['infos'][-1].get('contaminant_reduction', 0)
            
            # Calculate smoothness (action variance)
            actions = episode['actions']
            smoothness = 1.0 / (1.0 + np.var(actions, axis=0).sum())
            
            # Calculate safety compliance
            safety_compliance = np.mean(episode['safety_flags'])
            
            features.append([
                liquid_reduction,
                contaminant_reduction,
                smoothness,
                safety_compliance
            ])
        
        features = np.array(features)
        
        # Fit GMM
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(features)
        
        # Calculate weights
        weights = gmm.predict_proba(features)[:, 1]  # Probability of "good" component
        
        # Save weights
        weights_file = output_dir / "weights.npz"
        np.savez(weights_file, weights=weights, features=features)
        
        logger.info(f"Saved weights to {weights_file}")
    
    def _generate_summary(self, episodes: List[Dict[str, Any]], output_dir: Path):
        """Generate episode summary."""
        logger.info("Generating episode summary")
        
        summary_data = []
        
        for episode in episodes:
            liquid_reduction = episode['infos'][-1].get('liquid_reduction', 0)
            contaminant_reduction = episode['infos'][-1].get('contaminant_reduction', 0)
            safety_violations = sum(1 for info in episode['infos'] if info.get('safety_violations', 0) > 0)
            collision_count = episode['infos'][-1].get('collision_count', 0)
            
            summary_data.append({
                'episode_id': episode['episode_id'],
                'length': episode['length'],
                'liquid_reduction': liquid_reduction,
                'contaminant_reduction': contaminant_reduction,
                'safety_violations': safety_violations,
                'collision_count': collision_count,
                'total_reward': np.sum(episode['rewards'])
            })
        
        # Save summary
        summary_file = output_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        logger.info(f"Saved summary to {summary_file}")


def generate_demos(
    output_dir: Path,
    config: Dict[str, Any],
    mock: bool = True
) -> None:
    """Generate demonstrations."""
    generator = DemoGenerator(config)
    generator.generate_demos(
        num_episodes=config.get('num_episodes', 100),
        output_dir=output_dir,
        mock=mock
    )
