"""
Tests for demo generation and filtering.
"""

import pytest
import numpy as np
import h5py
from pathlib import Path
from unittest.mock import Mock, patch
import sys
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.il.demos import ScriptedPolicy, DemoGenerator, generate_demos
from src.il.filter_weight import DemoFilter, DemoWeighting, filter_weight_demos
from src.envs.mock_env import MockSuctionEnv


class TestScriptedPolicy:
    """Test scripted policy for demo generation."""
    
    def test_policy_creation(self):
        """Test scripted policy creation."""
        config = {
            'noise_level': 0.1,
            'suction_threshold': 0.5,
            'movement_scale': 0.05,
            'suction_probability': 0.3
        }
        
        policy = ScriptedPolicy(config)
        
        assert policy.noise_level == 0.1
        assert policy.suction_threshold == 0.5
        assert policy.movement_scale == 0.05
        assert policy.suction_probability == 0.3
    
    def test_policy_action(self):
        """Test scripted policy action generation."""
        config = {
            'noise_level': 0.1,
            'suction_threshold': 0.5,
            'movement_scale': 0.05,
            'suction_probability': 0.3
        }
        
        policy = ScriptedPolicy(config)
        
        # Create dummy observation and info
        obs = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        info = {
            'liquid_mass': 50.0,
            'contaminant_mass': 25.0,
            'position': np.array([0.0, 0.0, 0.0]),
            'orientation': 0.0
        }
        
        action = policy.get_action(obs, info)
        
        assert isinstance(action, np.ndarray)
        assert action.shape == (5,)
        assert action.dtype == np.float32
        
        # Check action bounds
        assert -0.1 <= action[0] <= 0.1  # dx
        assert -0.1 <= action[1] <= 0.1  # dy
        assert -0.1 <= action[2] <= 0.1  # dz
        assert -0.2 <= action[3] <= 0.2  # dyaw
        assert action[4] in [0.0, 1.0]   # suction_toggle
    
    def test_policy_suction_logic(self):
        """Test suction logic in scripted policy."""
        config = {
            'noise_level': 0.0,  # No noise for deterministic test
            'suction_threshold': 0.5,
            'movement_scale': 0.05,
            'suction_probability': 1.0  # Always suction when conditions met
        }
        
        policy = ScriptedPolicy(config)
        
        # Test with high liquid mass (should suction)
        obs = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        info = {
            'liquid_mass': 50.0,  # High liquid mass
            'contaminant_mass': 25.0,
            'position': np.array([0.0, 0.0, 0.0]),
            'orientation': 0.0
        }
        
        action = policy.get_action(obs, info)
        
        # Should activate suction
        assert action[4] == 1.0
        
        # Test with low liquid mass (should not suction)
        info['liquid_mass'] = 5.0  # Low liquid mass
        
        action = policy.get_action(obs, info)
        
        # Should not activate suction
        assert action[4] == 0.0


class TestDemoGenerator:
    """Test demo generation."""
    
    def test_generator_creation(self):
        """Test demo generator creation."""
        config = {
            'image_size': [128, 128],
            'episode_length': 100,
            'scripted_policy': {
                'noise_level': 0.1,
                'suction_threshold': 0.5,
                'movement_scale': 0.05,
                'suction_probability': 0.3
            }
        }
        
        generator = DemoGenerator(config)
        
        assert generator.image_size == (128, 128)
        assert generator.episode_length == 100
        assert generator.scripted_policy is not None
    
    def test_episode_generation(self):
        """Test single episode generation."""
        config = {
            'image_size': [64, 64],
            'episode_length': 50,
            'scripted_policy': {
                'noise_level': 0.1,
                'suction_threshold': 0.5,
                'movement_scale': 0.05,
                'suction_probability': 0.3
            }
        }
        
        generator = DemoGenerator(config)
        
        # Create mock environment
        env = MockSuctionEnv(image_size=(64, 64), max_steps=50)
        
        # Generate episode
        episode = generator.generate_episode(env, episode_id=0)
        
        assert isinstance(episode, dict)
        assert 'episode_id' in episode
        assert 'observations' in episode
        assert 'actions' in episode
        assert 'rewards' in episode
        assert 'infos' in episode
        assert 'safety_flags' in episode
        assert 'length' in episode
        
        assert episode['episode_id'] == 0
        assert episode['length'] > 0
        assert len(episode['observations']) == episode['length']
        assert len(episode['actions']) == episode['length']
        assert len(episode['rewards']) == episode['length']
        assert len(episode['infos']) == episode['length']
        assert len(episode['safety_flags']) == episode['length']
        
        # Check data types and shapes
        assert episode['observations'].shape == (episode['length'], 64, 64, 3)
        assert episode['actions'].shape == (episode['length'], 5)
        assert episode['rewards'].shape == (episode['length'],)
        assert episode['safety_flags'].shape == (episode['length'],)
    
    def test_demo_saving(self):
        """Test demo saving to HDF5."""
        config = {
            'image_size': [64, 64],
            'episode_length': 20,
            'scripted_policy': {
                'noise_level': 0.1,
                'suction_threshold': 0.5,
                'movement_scale': 0.05,
                'suction_probability': 0.3
            }
        }
        
        generator = DemoGenerator(config)
        
        # Create mock episodes
        episodes = []
        for i in range(3):
            episode = {
                'episode_id': i,
                'observations': np.random.randint(0, 255, (20, 64, 64, 3), dtype=np.uint8),
                'actions': np.random.randn(20, 5).astype(np.float32),
                'rewards': np.random.randn(20).astype(np.float32),
                'infos': [{'liquid_mass': 50.0, 'contaminant_mass': 25.0} for _ in range(20)],
                'safety_flags': np.random.choice([True, False], 20),
                'length': 20
            }
            episodes.append(episode)
        
        # Save episodes
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            generator._save_episodes(episodes, output_dir)
            
            # Check if file was created
            demos_file = output_dir / "demos.h5"
            assert demos_file.exists()
            
            # Check file contents
            with h5py.File(demos_file, 'r') as f:
                assert len(f.keys()) == 3  # 3 episodes
                
                for i in range(3):
                    episode_key = f"episode_{i:03d}"
                    assert episode_key in f
                    
                    episode_group = f[episode_key]
                    assert 'observations' in episode_group
                    assert 'actions' in episode_group
                    assert 'rewards' in episode_group
                    assert 'safety_flags' in episode_group
                    assert episode_group.attrs['length'] == 20
    
    def test_weight_generation(self):
        """Test weight generation."""
        config = {
            'image_size': [64, 64],
            'episode_length': 20,
            'scripted_policy': {
                'noise_level': 0.1,
                'suction_threshold': 0.5,
                'movement_scale': 0.05,
                'suction_probability': 0.3
            }
        }
        
        generator = DemoGenerator(config)
        
        # Create mock episodes with different quality levels
        episodes = []
        for i in range(5):
            # Vary liquid reduction to create different quality levels
            liquid_reduction = 0.3 + i * 0.1  # 0.3 to 0.7
            
            episode = {
                'episode_id': i,
                'observations': np.random.randint(0, 255, (20, 64, 64, 3), dtype=np.uint8),
                'actions': np.random.randn(20, 5).astype(np.float32),
                'rewards': np.random.randn(20).astype(np.float32),
                'infos': [{'liquid_reduction': liquid_reduction, 'contaminant_reduction': 0.5} for _ in range(20)],
                'safety_flags': np.random.choice([True, False], 20),
                'length': 20
            }
            episodes.append(episode)
        
        # Generate weights
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            generator._generate_weights(episodes, output_dir)
            
            # Check if weights file was created
            weights_file = output_dir / "weights.npz"
            assert weights_file.exists()
            
            # Check weights contents
            weights_data = np.load(weights_file)
            assert 'weights' in weights_data
            assert 'features' in weights_data
            
            weights = weights_data['weights']
            features = weights_data['features']
            
            assert len(weights) == 5  # 5 episodes
            assert features.shape == (5, 4)  # 4 features
            
            # Weights should be between 0 and 1
            assert np.all(weights >= 0)
            assert np.all(weights <= 1)
    
    def test_summary_generation(self):
        """Test summary generation."""
        config = {
            'image_size': [64, 64],
            'episode_length': 20,
            'scripted_policy': {
                'noise_level': 0.1,
                'suction_threshold': 0.5,
                'movement_scale': 0.05,
                'suction_probability': 0.3
            }
        }
        
        generator = DemoGenerator(config)
        
        # Create mock episodes
        episodes = []
        for i in range(3):
            episode = {
                'episode_id': i,
                'observations': np.random.randint(0, 255, (20, 64, 64, 3), dtype=np.uint8),
                'actions': np.random.randn(20, 5).astype(np.float32),
                'rewards': np.random.randn(20).astype(np.float32),
                'infos': [
                    {'liquid_reduction': 0.5, 'contaminant_reduction': 0.3, 'safety_violations': 0, 'collision_count': 1}
                    for _ in range(20)
                ],
                'safety_flags': np.random.choice([True, False], 20),
                'length': 20
            }
            episodes.append(episode)
        
        # Generate summary
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            generator._generate_summary(episodes, output_dir)
            
            # Check if summary file was created
            summary_file = output_dir / "summary.json"
            assert summary_file.exists()
            
            # Check summary contents
            import json
            with open(summary_file, 'r') as f:
                summary_data = json.load(f)
            
            assert len(summary_data) == 3  # 3 episodes
            
            for episode_summary in summary_data:
                assert 'episode_id' in episode_summary
                assert 'length' in episode_summary
                assert 'liquid_reduction' in episode_summary
                assert 'contaminant_reduction' in episode_summary
                assert 'safety_violations' in episode_summary
                assert 'collision_count' in episode_summary
                assert 'total_reward' in episode_summary


class TestDemoFilter:
    """Test demo filtering."""
    
    def test_filter_creation(self):
        """Test demo filter creation."""
        config = {
            'quality_filtering': {
                'enabled': True,
                'min_liquid_reduction': 0.3,
                'min_contaminant_reduction': 0.2,
                'max_collision_rate': 0.1,
                'max_safety_violations': 0.05
            }
        }
        
        filter_obj = DemoFilter(config)
        
        assert filter_obj.enabled == True
        assert filter_obj.min_liquid_reduction == 0.3
        assert filter_obj.min_contaminant_reduction == 0.2
        assert filter_obj.max_collision_rate == 0.1
        assert filter_obj.max_safety_violations == 0.05
    
    def test_episode_filtering(self):
        """Test episode filtering."""
        config = {
            'quality_filtering': {
                'enabled': True,
                'min_liquid_reduction': 0.3,
                'min_contaminant_reduction': 0.2,
                'max_collision_rate': 0.1,
                'max_safety_violations': 0.05
            }
        }
        
        filter_obj = DemoFilter(config)
        
        # Create episodes with different quality levels
        episodes = []
        
        # Good episode
        good_episode = {
            'episode_id': 0,
            'length': 100,
            'infos': [
                {'liquid_reduction': 0.8, 'contaminant_reduction': 0.6, 'safety_violations': 0, 'collision_count': 2}
                for _ in range(100)
            ]
        }
        episodes.append(good_episode)
        
        # Bad episode (low liquid reduction)
        bad_episode = {
            'episode_id': 1,
            'length': 100,
            'infos': [
                {'liquid_reduction': 0.2, 'contaminant_reduction': 0.6, 'safety_violations': 0, 'collision_count': 2}
                for _ in range(100)
            ]
        }
        episodes.append(bad_episode)
        
        # Filter episodes
        filtered_episodes = filter_obj.filter_episodes(episodes)
        
        # Should keep only the good episode
        assert len(filtered_episodes) == 1
        assert filtered_episodes[0]['episode_id'] == 0
    
    def test_filter_disabled(self):
        """Test filter when disabled."""
        config = {
            'quality_filtering': {
                'enabled': False
            }
        }
        
        filter_obj = DemoFilter(config)
        
        episodes = [{'episode_id': i} for i in range(5)]
        filtered_episodes = filter_obj.filter_episodes(episodes)
        
        # Should return all episodes when disabled
        assert len(filtered_episodes) == 5


class TestDemoWeighting:
    """Test demo weighting."""
    
    def test_weighting_creation(self):
        """Test demo weighting creation."""
        config = {
            'weighting': {
                'enabled': True,
                'gmm_components': 2,
                'features': ['liquid_reduction', 'contaminant_reduction', 'smoothness', 'safety_compliance'],
                'weights': {
                    'liquid_reduction': 0.4,
                    'contaminant_reduction': 0.3,
                    'smoothness': 0.2,
                    'safety_compliance': 0.1
                }
            }
        }
        
        weighting_obj = DemoWeighting(config)
        
        assert weighting_obj.enabled == True
        assert weighting_obj.gmm_components == 2
        assert len(weighting_obj.features) == 4
        assert weighting_obj.feature_weights['liquid_reduction'] == 0.4
    
    def test_weight_calculation(self):
        """Test weight calculation."""
        config = {
            'weighting': {
                'enabled': True,
                'gmm_components': 2,
                'features': ['liquid_reduction', 'contaminant_reduction', 'smoothness', 'safety_compliance'],
                'weights': {
                    'liquid_reduction': 0.4,
                    'contaminant_reduction': 0.3,
                    'smoothness': 0.2,
                    'safety_compliance': 0.1
                }
            }
        }
        
        weighting_obj = DemoWeighting(config)
        
        # Create episodes with different quality levels
        episodes = []
        for i in range(5):
            episode = {
                'episode_id': i,
                'actions': np.random.randn(20, 5).astype(np.float32),
                'safety_flags': np.random.choice([True, False], 20),
                'infos': [
                    {
                        'liquid_reduction': 0.3 + i * 0.1,  # 0.3 to 0.7
                        'contaminant_reduction': 0.2 + i * 0.1  # 0.2 to 0.6
                    }
                    for _ in range(20)
                ]
            }
            episodes.append(episode)
        
        # Calculate weights
        weights = weighting_obj.calculate_weights(episodes)
        
        assert len(weights) == 5
        assert np.all(weights >= 0)
        assert np.all(weights <= 1)
    
    def test_weighting_disabled(self):
        """Test weighting when disabled."""
        config = {
            'weighting': {
                'enabled': False
            }
        }
        
        weighting_obj = DemoWeighting(config)
        
        episodes = [{'episode_id': i} for i in range(5)]
        weights = weighting_obj.calculate_weights(episodes)
        
        # Should return uniform weights when disabled
        assert len(weights) == 5
        assert np.all(weights == 1.0)


class TestGenerateDemos:
    """Test demo generation function."""
    
    def test_generate_demos(self):
        """Test demo generation function."""
        config = {
            'num_episodes': 3,
            'image_size': [64, 64],
            'episode_length': 20,
            'scripted_policy': {
                'noise_level': 0.1,
                'suction_threshold': 0.5,
                'movement_scale': 0.05,
                'suction_probability': 0.3
            }
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # Generate demos
            generate_demos(output_dir, config, mock=True)
            
            # Check if files were created
            assert (output_dir / "demos.h5").exists()
            assert (output_dir / "weights.npz").exists()
            assert (output_dir / "summary.json").exists()
            
            # Check demos file
            with h5py.File(output_dir / "demos.h5", 'r') as f:
                assert len(f.keys()) == 3  # 3 episodes
            
            # Check weights file
            weights_data = np.load(output_dir / "weights.npz")
            assert len(weights_data['weights']) == 3
            
            # Check summary file
            import json
            with open(output_dir / "summary.json", 'r') as f:
                summary_data = json.load(f)
            assert len(summary_data) == 3


if __name__ == "__main__":
    pytest.main([__file__])