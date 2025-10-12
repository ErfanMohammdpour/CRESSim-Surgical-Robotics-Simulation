"""
Tests for demo generation and processing.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile
import shutil

from src.il.demos import ScriptedPolicy, DemoGenerator, generate_demos
from src.il.filter_weight import DemoFilter, DemoWeighting, DemoProcessor
from src.il.bc_trainer import BCTrainer, DemoDataset


class TestScriptedPolicy:
    """Test scripted policy functionality."""
    
    def test_policy_creation(self):
        """Test scripted policy creation."""
        config = {
            'suction_radius': 0.05,
            'approach_speed': 0.02,
            'suction_speed': 0.01,
            'exploration_noise': 0.1,
            'success_threshold': 0.8
        }
        
        policy = ScriptedPolicy(config)
        
        assert policy.suction_radius == 0.05
        assert policy.approach_speed == 0.02
        assert policy.suction_speed == 0.01
        assert policy.exploration_noise == 0.1
        assert policy.success_threshold == 0.8
    
    def test_policy_reset(self):
        """Test policy reset."""
        config = {
            'suction_radius': 0.05,
            'approach_speed': 0.02,
            'suction_speed': 0.01,
            'exploration_noise': 0.1,
            'success_threshold': 0.8
        }
        
        policy = ScriptedPolicy(config)
        
        obs = {
            'image': np.random.rand(3, 84, 84),
            'aux': np.array([0.0, 1.0, 0.5, 0.0])
        }
        
        policy.reset(obs)
        
        assert policy.target_position is not None
        assert policy.suction_active is False
        assert policy.episode_step == 0
    
    def test_policy_action_generation(self):
        """Test action generation."""
        config = {
            'suction_radius': 0.05,
            'approach_speed': 0.02,
            'suction_speed': 0.01,
            'exploration_noise': 0.1,
            'success_threshold': 0.8
        }
        
        policy = ScriptedPolicy(config)
        
        obs = {
            'image': np.random.rand(3, 84, 84),
            'aux': np.array([0.0, 1.0, 0.5, 0.0])
        }
        
        policy.reset(obs)
        action = policy.act(obs)
        
        assert action.shape == (5,)
        assert np.all(action >= -1.0) and np.all(action <= 1.0)
        assert action.dtype == np.float32
    
    def test_action_components(self):
        """Test action component generation."""
        config = {
            'suction_radius': 0.05,
            'approach_speed': 0.02,
            'suction_speed': 0.01,
            'exploration_noise': 0.0,  # No noise for deterministic test
            'success_threshold': 0.8
        }
        
        policy = ScriptedPolicy(config)
        
        obs = {
            'image': np.random.rand(3, 84, 84),
            'aux': np.array([0.0, 1.0, 0.5, 0.0])
        }
        
        policy.reset(obs)
        action = policy.act(obs)
        
        # Check action components
        assert len(action) == 5  # [dx, dy, dz, dyaw, suction_toggle]
        assert action[4] in [0.0, 1.0]  # Suction toggle should be binary


class TestDemoGenerator:
    """Test demo generation functionality."""
    
    def test_generator_creation(self):
        """Test demo generator creation."""
        config = {
            'num_episodes': 10,
            'episode_length': 100,
            'num_workers': 2,
            'mock': True
        }
        
        generator = DemoGenerator(config)
        
        assert generator.num_episodes == 10
        assert generator.episode_length == 100
        assert generator.num_workers == 2
        assert generator.mock is True
    
    def test_mock_environment_creation(self):
        """Test mock environment creation."""
        config = {'mock': True}
        generator = DemoGenerator(config)
        
        env = generator._create_mock_env()
        assert env is not None
    
    def test_episode_generation(self):
        """Test single episode generation."""
        config = {
            'num_episodes': 1,
            'episode_length': 10,
            'mock': True,
            'scripted_policy': {
                'suction_radius': 0.05,
                'approach_speed': 0.02,
                'suction_speed': 0.01,
                'exploration_noise': 0.1,
                'success_threshold': 0.8
            }
        }
        
        generator = DemoGenerator(config)
        env = generator._create_mock_env()
        env = generator._wrap_env(env)
        
        episode_data = generator._generate_episode(env, 0)
        
        assert 'observations' in episode_data
        assert 'actions' in episode_data
        assert 'rewards' in episode_data
        assert 'dones' in episode_data
        assert 'infos' in episode_data
        assert 'episode_length' in episode_data
        assert 'episode_idx' in episode_data
        
        assert len(episode_data['observations']) > 0
        assert len(episode_data['actions']) > 0
        assert len(episode_data['rewards']) > 0
        assert len(episode_data['dones']) > 0
        assert len(episode_data['infos']) > 0
    
    def test_noise_application(self):
        """Test noise application to actions."""
        config = {
            'noise': {
                'position_std': 0.01,
                'rotation_std': 0.05,
                'action_dropout': 0.1,
                'temporal_noise': 0.05
            }
        }
        
        generator = DemoGenerator(config)
        
        action = np.array([0.1, 0.1, 0.1, 0.1, 1.0], dtype=np.float32)
        noisy_action = generator._apply_noise(action, 0)
        
        assert noisy_action.shape == action.shape
        assert np.all(noisy_action >= -1.0) and np.all(noisy_action <= 1.0)
    
    @patch('src.il.demos.MockUnityEnv')
    def test_demo_generation_integration(self, mock_env_class):
        """Test complete demo generation integration."""
        # Mock environment
        mock_env = Mock()
        mock_env.reset.return_value = (
            {'image': np.random.rand(3, 84, 84), 'aux': np.array([0.0, 1.0, 0.5, 0.0])},
            {'liquid_mass_remaining': 1.0, 'contaminant_mass_remaining': 0.5}
        )
        mock_env.step.return_value = (
            {'image': np.random.rand(3, 84, 84), 'aux': np.array([0.0, 0.9, 0.4, 0.0])},
            0.1, False, False, {'liquid_mass_remaining': 0.9, 'contaminant_mass_remaining': 0.4}
        )
        mock_env_class.return_value = mock_env
        
        config = {
            'num_episodes': 2,
            'episode_length': 5,
            'mock': True,
            'scripted_policy': {
                'suction_radius': 0.05,
                'approach_speed': 0.02,
                'suction_speed': 0.01,
                'exploration_noise': 0.1,
                'success_threshold': 0.8
            }
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = DemoGenerator(config)
            generator.generate_demos(temp_dir)
            
            # Check that files were created
            temp_path = Path(temp_dir)
            demo_files = list(temp_path.glob("demos_*.npz"))
            assert len(demo_files) > 0
            
            # Check summary file
            summary_file = temp_path / "demo_summary.json"
            assert summary_file.exists()


class TestDemoFilter:
    """Test demo filtering functionality."""
    
    def test_filter_creation(self):
        """Test demo filter creation."""
        config = {
            'method': 'gmm',
            'gmm_components': 3,
            'percentile_threshold': 0.7,
            'min_episode_length': 100,
            'max_episode_length': 1000
        }
        
        filter_obj = DemoFilter(config)
        
        assert filter_obj.method == 'gmm'
        assert filter_obj.gmm_components == 3
        assert filter_obj.percentile_threshold == 0.7
        assert filter_obj.min_episode_length == 100
        assert filter_obj.max_episode_length == 1000
    
    def test_quality_score_calculation(self):
        """Test quality score calculation."""
        config = {
            'quality_metrics': {
                'liquid_reduction_weight': 0.4,
                'contaminant_reduction_weight': 0.3,
                'smoothness_weight': 0.2,
                'safety_weight': 0.1
            }
        }
        
        filter_obj = DemoFilter(config)
        
        # Create mock demo
        demo = {
            'infos': [
                {'liquid_mass_remaining': 1.0, 'contaminant_mass_remaining': 0.5, 'collisions': 0},
                {'liquid_mass_remaining': 0.8, 'contaminant_mass_remaining': 0.3, 'collisions': 0}
            ],
            'actions': [
                np.array([0.1, 0.1, 0.1, 0.1, 1.0]),
                np.array([0.2, 0.2, 0.2, 0.2, 0.0])
            ]
        }
        
        quality_score = filter_obj._calculate_quality_scores([demo])[0]
        
        assert isinstance(quality_score, float)
        assert quality_score >= 0.0 and quality_score <= 1.0
    
    def test_length_filtering(self):
        """Test episode length filtering."""
        config = {
            'min_episode_length': 5,
            'max_episode_length': 20
        }
        
        filter_obj = DemoFilter(config)
        
        # Create demos with different lengths
        demos = [
            {'episode_length': 3},   # Too short
            {'episode_length': 10},  # Good
            {'episode_length': 25},  # Too long
            {'episode_length': 15}   # Good
        ]
        
        valid_indices = filter_obj._filter_by_length(demos)
        
        assert valid_indices[0] == False  # Too short
        assert valid_indices[1] == True   # Good
        assert valid_indices[2] == False  # Too long
        assert valid_indices[3] == True   # Good
    
    def test_gmm_filtering(self):
        """Test GMM-based filtering."""
        config = {
            'method': 'gmm',
            'gmm_components': 2
        }
        
        filter_obj = DemoFilter(config)
        
        # Create mock quality scores
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        
        weights = filter_obj._gmm_filtering(scores)
        
        assert len(weights) == len(scores)
        assert np.all(weights >= 0.0)
        assert np.all(weights <= 1.0)
    
    def test_percentile_filtering(self):
        """Test percentile-based filtering."""
        config = {
            'method': 'percentile',
            'percentile_threshold': 0.7
        }
        
        filter_obj = DemoFilter(config)
        
        # Create mock quality scores
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        
        weights = filter_obj._percentile_filtering(scores)
        
        assert len(weights) == len(scores)
        assert np.all(weights >= 0.0)
        assert np.all(weights <= 1.0)
        
        # Check that top 30% get non-zero weights
        threshold = np.percentile(scores, 30)
        high_scores = scores >= threshold
        assert np.all(weights[high_scores] > 0.0)


class TestDemoWeighting:
    """Test demo weighting functionality."""
    
    def test_weighting_creation(self):
        """Test demo weighting creation."""
        config = {
            'weighting_method': 'quality_based',
            'temperature': 1.0,
            'min_weight': 0.01,
            'max_weight': 10.0
        }
        
        weighting = DemoWeighting(config)
        
        assert weighting.weighting_method == 'quality_based'
        assert weighting.temperature == 1.0
        assert weighting.min_weight == 0.01
        assert weighting.max_weight == 10.0
    
    def test_quality_based_weighting(self):
        """Test quality-based weighting."""
        config = {
            'weighting_method': 'quality_based',
            'temperature': 1.0
        }
        
        weighting = DemoWeighting(config)
        
        demos = [{'episode_length': 100} for _ in range(5)]
        quality_scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        
        weights = weighting.calculate_weights(demos, quality_scores)
        
        assert len(weights) == len(demos)
        assert np.all(weights >= 0.0)
        assert abs(np.sum(weights) - len(weights)) < 1e-6  # Should be normalized
    
    def test_uniform_weighting(self):
        """Test uniform weighting."""
        config = {
            'weighting_method': 'uniform'
        }
        
        weighting = DemoWeighting(config)
        
        demos = [{'episode_length': 100} for _ in range(5)]
        quality_scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        
        weights = weighting.calculate_weights(demos, quality_scores)
        
        assert len(weights) == len(demos)
        assert np.allclose(weights, 1.0)  # All weights should be equal
    
    def test_length_based_weighting(self):
        """Test length-based weighting."""
        config = {
            'weighting_method': 'length_based'
        }
        
        weighting = DemoWeighting(config)
        
        demos = [
            {'episode_length': 50},
            {'episode_length': 100},
            {'episode_length': 150},
            {'episode_length': 200}
        ]
        quality_scores = np.array([0.1, 0.3, 0.5, 0.7])
        
        weights = weighting.calculate_weights(demos, quality_scores)
        
        assert len(weights) == len(demos)
        assert np.all(weights >= 0.0)
        # Longer episodes should have higher weights
        assert weights[3] > weights[2] > weights[1] > weights[0]


class TestDemoDataset:
    """Test demo dataset functionality."""
    
    def test_dataset_creation(self):
        """Test demo dataset creation."""
        # Create mock demos
        demos = [
            {
                'observations': [{'image': np.random.rand(3, 84, 84), 'aux': np.array([0.0, 1.0, 0.5, 0.0])} for _ in range(10)],
                'actions': [np.random.rand(5) for _ in range(10)],
                'rewards': [0.1] * 10,
                'infos': [{'liquid_mass_remaining': 1.0 - i*0.1, 'contaminant_mass_remaining': 0.5 - i*0.05, 'collisions': 0} for i in range(10)],
                'episode_length': 10
            }
        ]
        
        weights = np.array([1.0])
        
        dataset = DemoDataset(demos, weights, augment=False)
        
        assert len(dataset) == 10  # 10 transitions
        assert len(dataset.transitions) == 10
        assert len(dataset.transition_weights) == 10
    
    def test_dataset_getitem(self):
        """Test dataset item access."""
        # Create mock demos
        demos = [
            {
                'observations': [{'image': np.random.rand(3, 84, 84), 'aux': np.array([0.0, 1.0, 0.5, 0.0])} for _ in range(5)],
                'actions': [np.random.rand(5) for _ in range(5)],
                'rewards': [0.1] * 5,
                'infos': [{'liquid_mass_remaining': 1.0 - i*0.1, 'contaminant_mass_remaining': 0.5 - i*0.05, 'collisions': 0} for i in range(5)],
                'episode_length': 5
            }
        ]
        
        weights = np.array([1.0])
        
        dataset = DemoDataset(demos, weights, augment=False)
        
        # Test getting an item
        item = dataset[0]
        
        assert 'image' in item
        assert 'aux' in item
        assert 'action' in item
        assert 'weight' in item
        
        assert item['image'].shape == (3, 84, 84)
        assert item['aux'].shape == (4,)
        assert item['action'].shape == (5,)
        assert isinstance(item['weight'], float)


class TestDemoProcessor:
    """Test demo processing pipeline."""
    
    def test_processor_creation(self):
        """Test demo processor creation."""
        config = {
            'filtering': {
                'method': 'gmm',
                'gmm_components': 3
            },
            'weighting': {
                'weighting_method': 'quality_based',
                'temperature': 1.0
            }
        }
        
        processor = DemoProcessor(config)
        
        assert processor.filter is not None
        assert processor.weighting is not None
    
    def test_demo_processing(self):
        """Test complete demo processing."""
        config = {
            'filtering': {
                'method': 'percentile',
                'percentile_threshold': 0.7,
                'min_episode_length': 5,
                'max_episode_length': 100
            },
            'weighting': {
                'weighting_method': 'quality_based',
                'temperature': 1.0
            }
        }
        
        processor = DemoProcessor(config)
        
        # Create mock demos
        demos = [
            {
                'observations': [{'image': np.random.rand(3, 84, 84), 'aux': np.array([0.0, 1.0, 0.5, 0.0])} for _ in range(10)],
                'actions': [np.random.rand(5) for _ in range(10)],
                'rewards': [0.1] * 10,
                'infos': [{'liquid_mass_remaining': 1.0 - i*0.1, 'contaminant_mass_remaining': 0.5 - i*0.05, 'collisions': 0} for i in range(10)],
                'episode_length': 10
            },
            {
                'observations': [{'image': np.random.rand(3, 84, 84), 'aux': np.array([0.0, 1.0, 0.5, 0.0])} for _ in range(8)],
                'actions': [np.random.rand(5) for _ in range(8)],
                'rewards': [0.2] * 8,
                'infos': [{'liquid_mass_remaining': 1.0 - i*0.1, 'contaminant_mass_remaining': 0.5 - i*0.05, 'collisions': 0} for i in range(8)],
                'episode_length': 8
            }
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            filtered_demos, weights = processor.process_demos(demos, temp_dir)
            
            assert len(filtered_demos) <= len(demos)  # Some may be filtered out
            assert len(weights) == len(filtered_demos)
            assert np.all(weights >= 0.0)
            
            # Check that files were created
            temp_path = Path(temp_dir)
            assert (temp_path / "weights.npz").exists()
            assert (temp_path / "filtered_demos.npz").exists()
            assert (temp_path / "processing_stats.json").exists()


@pytest.mark.parametrize("num_episodes", [1, 5, 10])
def test_demo_generation_episodes(num_episodes):
    """Test demo generation with different numbers of episodes."""
    config = {
        'num_episodes': num_episodes,
        'episode_length': 5,
        'mock': True
    }
    
    generator = DemoGenerator(config)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        generator.generate_demos(temp_dir)
        
        # Check that files were created
        temp_path = Path(temp_dir)
        demo_files = list(temp_path.glob("demos_*.npz"))
        assert len(demo_files) > 0
        
        # Check summary file
        summary_file = temp_path / "demo_summary.json"
        assert summary_file.exists()


@pytest.mark.parametrize("weighting_method", ["quality_based", "uniform", "length_based", "reward_based"])
def test_weighting_methods(weighting_method):
    """Test different weighting methods."""
    config = {
        'weighting_method': weighting_method,
        'temperature': 1.0
    }
    
    weighting = DemoWeighting(config)
    
    demos = [
        {'episode_length': 100, 'total_reward': 10.0},
        {'episode_length': 150, 'total_reward': 15.0},
        {'episode_length': 200, 'total_reward': 20.0}
    ]
    
    quality_scores = np.array([0.3, 0.6, 0.9])
    
    weights = weighting.calculate_weights(demos, quality_scores)
    
    assert len(weights) == len(demos)
    assert np.all(weights >= 0.0)
    assert abs(np.sum(weights) - len(weights)) < 1e-6  # Should be normalized
