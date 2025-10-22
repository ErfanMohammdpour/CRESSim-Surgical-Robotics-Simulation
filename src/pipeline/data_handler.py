"""
Data handling module for surgical robotics pipeline
Handles dataset download, processing, and splitting
"""

import os
import json
import shutil
import zipfile
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class DataHandler:
    """Handles dataset download, processing, and splitting"""
    
    def __init__(self, kaggle_dataset: str = "kvasir-seg", output_dir: str = "data"):
        self.kaggle_dataset = kaggle_dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup paths
        self.raw_data_dir = self.output_dir / "raw_dataset"
        self.processed_data_dir = self.output_dir / "processed_dataset"
        self.train_dir = self.processed_data_dir / "train"
        self.val_dir = self.processed_data_dir / "validation"
        
        # Create directories
        for dir_path in [self.raw_data_dir, self.processed_data_dir, self.train_dir, self.val_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("DataHandler initialized")
    
    def download_kaggle_dataset(self) -> bool:
        """Download dataset from official source"""
        logger.info("📥 Downloading dataset from official source...")
        
        try:
            # Try direct download from official source
            import requests
            import zipfile
            
            dataset_url = "https://datasets.simula.no/downloads/kvasir-seg.zip"
            zip_path = self.raw_data_dir / "kvasir-seg.zip"
            
            logger.info(f"Downloading from: {dataset_url}")
            
            # Download the dataset with SSL verification disabled
            response = requests.get(dataset_url, stream=True, verify=False)
            response.raise_for_status()
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract the dataset
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.raw_data_dir)
            
            # Copy to kvasir_seg directory for consistency
            extracted_dir = self.raw_data_dir / "Kvasir-SEG"
            target_dir = self.output_dir / "kvasir_seg" / "Kvasir-SEG"
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            
            if extracted_dir.exists():
                import shutil
                if target_dir.exists():
                    shutil.rmtree(target_dir)
                shutil.copytree(extracted_dir, target_dir)
                logger.info(f"📁 Dataset copied to: {target_dir}")
            
            # Remove zip file
            zip_path.unlink()
            
            logger.info("✅ Dataset downloaded successfully from official source")
            return True
            
        except ImportError:
            logger.error("❌ Required packages not available!")
            logger.error("Please install: pip install requests")
            return False
        except Exception as e:
            logger.error(f"❌ Official download failed: {e}")
            logger.error("Please check your internet connection and try again")
            return False
    
    def _download_alternative(self) -> bool:
        """Alternative download method - DISABLED"""
        logger.error("❌ Synthetic data generation is disabled!")
        logger.error("Pipeline requires real Kvasir-SEG dataset")
        return False
    
    def _create_sample_dataset(self):
        """Create sample dataset for testing"""
        logger.info("📝 Creating sample dataset...")
        
        # Create sample images and masks
        images_dir = self.raw_data_dir / "images"
        masks_dir = self.raw_data_dir / "masks"
        images_dir.mkdir(exist_ok=True)
        masks_dir.mkdir(exist_ok=True)
        
        # Generate sample data
        for i in range(500):  # 500 sample images for better training
            # Create synthetic surgical image
            img = self._create_synthetic_surgical_image(i)
            img_path = images_dir / f"image_{i:03d}.jpg"
            img.save(img_path)
            
            # Create corresponding mask
            mask = self._create_synthetic_mask(i)
            mask_path = masks_dir / f"mask_{i:03d}.jpg"
            mask.save(mask_path)
        
        logger.info(f"✅ Sample dataset created with 100 images")
    
    def _create_synthetic_surgical_image(self, idx: int) -> Image.Image:
        """Create realistic synthetic surgical image"""
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        
        # Create realistic surgical field background
        # Gradient from dark center to slightly lighter edges
        y, x = np.ogrid[:512, :512]
        center_x, center_y = 256, 256
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt(256**2 + 256**2)
        
        # Dark surgical field with subtle gradient
        base_intensity = 15 + (dist / max_dist) * 10
        img[:, :, 0] = base_intensity + np.random.normal(0, 2, (512, 512))  # Red
        img[:, :, 1] = base_intensity + np.random.normal(0, 2, (512, 512))  # Green  
        img[:, :, 2] = base_intensity + 5 + np.random.normal(0, 2, (512, 512))  # Blue (slightly brighter)
        
        # Add realistic tissue texture
        tissue_noise = np.random.normal(0, 8, (512, 512))
        for c in range(3):
            img[:, :, c] = np.clip(img[:, :, c] + tissue_noise, 0, 255)
        
        # Add blood vessels (dark red lines)
        for _ in range(np.random.randint(3, 8)):
            start_x = np.random.randint(50, 462)
            start_y = np.random.randint(50, 462)
            end_x = np.random.randint(50, 462)
            end_y = np.random.randint(50, 462)
            
            # Draw vessel as thick line
            cv2.line(img, (start_x, start_y), (end_x, end_y), (40, 20, 20), np.random.randint(2, 5))
        
        # Add realistic surgical instruments
        # Main suction tool
        tool_angle = idx * 0.2
        tool_x = int(center_x + np.cos(tool_angle) * 80)
        tool_y = int(center_y + np.sin(tool_angle) * 60)
        tool_x = max(50, min(462, tool_x))
        tool_y = max(50, min(462, tool_y))
        
        # Draw realistic suction tool
        # Main body (metallic)
        cv2.ellipse(img, (tool_x, tool_y), (12, 8), tool_angle * 180 / np.pi, 0, 360, (180, 180, 190), -1)
        cv2.ellipse(img, (tool_x, tool_y), (12, 8), tool_angle * 180 / np.pi, 0, 360, (200, 200, 210), 2)
        
        # Suction tip (brighter metallic)
        tip_x = int(tool_x + np.cos(tool_angle) * 20)
        tip_y = int(tool_y + np.sin(tool_angle) * 15)
        cv2.circle(img, (tip_x, tip_y), 4, (220, 220, 230), -1)
        cv2.circle(img, (tip_x, tip_y), 4, (240, 240, 250), 1)
        
        # Add realistic blood/liquid pools
        for _ in range(np.random.randint(2, 5)):
            pool_x = np.random.randint(100, 412)
            pool_y = np.random.randint(100, 412)
            pool_size = np.random.randint(15, 35)
            
            # Create irregular blood pool
            mask = np.zeros((512, 512), dtype=np.uint8)
            cv2.circle(mask, (pool_x, pool_y), pool_size, 255, -1)
            
            # Add some irregularity
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=1)
            
            # Apply blood color with transparency
            blood_color = np.random.choice([(60, 20, 20), (80, 30, 30), (100, 40, 40)])
            alpha = np.random.uniform(0.3, 0.7)
            
            for c in range(3):
                img[:, :, c] = np.where(mask > 0, 
                    img[:, :, c] * (1 - alpha) + blood_color[c] * alpha, 
                    img[:, :, c])
        
        # Add tissue folds and wrinkles
        for _ in range(np.random.randint(5, 12)):
            fold_x = np.random.randint(50, 462)
            fold_y = np.random.randint(50, 462)
            fold_length = np.random.randint(20, 60)
            fold_angle = np.random.uniform(0, 2 * np.pi)
            
            end_x = int(fold_x + np.cos(fold_angle) * fold_length)
            end_y = int(fold_y + np.sin(fold_angle) * fold_length)
            
            # Draw subtle tissue fold
            cv2.line(img, (fold_x, fold_y), (end_x, end_y), (25, 25, 35), 2)
        
        # Add realistic lighting effects
        # Create spotlight effect
        light_center_x = center_x + np.random.randint(-50, 50)
        light_center_y = center_y + np.random.randint(-50, 50)
        light_mask = np.zeros((512, 512), dtype=np.float32)
        cv2.circle(light_mask, (light_center_x, light_center_y), 150, 1.0, -1)
        light_mask = cv2.GaussianBlur(light_mask, (101, 101), 0)
        
        # Apply lighting
        for c in range(3):
            img[:, :, c] = np.clip(img[:, :, c] * (0.7 + 0.3 * light_mask), 0, 255)
        
        # Add realistic camera noise and artifacts
        # Gaussian noise
        noise = np.random.normal(0, 3, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        # Add some compression artifacts
        if np.random.random() < 0.3:
            # Simulate JPEG compression artifacts
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) * 0.1
            for c in range(3):
                img[:, :, c] = np.clip(cv2.filter2D(img[:, :, c].astype(np.float32), -1, kernel), 0, 255).astype(np.uint8)
        
        # Resize to standard size
        img = cv2.resize(img, (256, 256))
        
        return Image.fromarray(img)
    
    def _create_synthetic_mask(self, idx: int) -> Image.Image:
        """Create realistic synthetic segmentation mask"""
        mask = np.zeros((256, 256), dtype=np.uint8)
        
        # Background: 0 (black)
        # Healthy tissue: 1 (gray)
        # Dangerous areas: 2 (white)
        
        center_x, center_y = 128, 128
        
        # Create realistic tissue area with irregular shape
        # Main tissue region
        tissue_center_x = center_x + np.random.randint(-20, 20)
        tissue_center_y = center_y + np.random.randint(-20, 20)
        
        # Create irregular tissue boundary
        tissue_points = []
        num_points = 16
        for i in range(num_points):
            angle = (2 * np.pi * i) / num_points
            radius = 60 + np.random.randint(-15, 20)
            x = int(tissue_center_x + radius * np.cos(angle))
            y = int(tissue_center_y + radius * np.sin(angle))
            x = max(10, min(246, x))
            y = max(10, min(246, y))
            tissue_points.append([x, y])
        
        # Fill tissue area
        tissue_points = np.array(tissue_points, dtype=np.int32)
        cv2.fillPoly(mask, [tissue_points], 1)
        
        # Add tissue texture (small variations)
        for _ in range(np.random.randint(3, 8)):
            small_x = np.random.randint(50, 206)
            small_y = np.random.randint(50, 206)
            small_radius = np.random.randint(5, 15)
            cv2.circle(mask, (small_x, small_y), small_radius, 1, -1)
        
        # Add dangerous areas (blood vessels, sensitive tissue)
        num_danger_areas = np.random.randint(1, 4)
        for _ in range(num_danger_areas):
            danger_x = center_x + np.random.randint(-60, 60)
            danger_y = center_y + np.random.randint(-60, 60)
            danger_x = max(20, min(236, danger_x))
            danger_y = max(20, min(236, danger_y))
            
            # Create irregular dangerous area
            danger_points = []
            num_danger_points = 8
            danger_radius = np.random.randint(8, 20)
            
            for i in range(num_danger_points):
                angle = (2 * np.pi * i) / num_danger_points
                radius = danger_radius + np.random.randint(-3, 5)
                x = int(danger_x + radius * np.cos(angle))
                y = int(danger_y + radius * np.sin(angle))
                x = max(5, min(251, x))
                y = max(5, min(251, y))
                danger_points.append([x, y])
            
            danger_points = np.array(danger_points, dtype=np.int32)
            cv2.fillPoly(mask, [danger_points], 2)
        
        # Add blood vessels as thin lines
        for _ in range(np.random.randint(2, 6)):
            start_x = np.random.randint(30, 226)
            start_y = np.random.randint(30, 226)
            end_x = np.random.randint(30, 226)
            end_y = np.random.randint(30, 226)
            
            # Draw vessel as thick line
            vessel_thickness = np.random.randint(2, 4)
            cv2.line(mask, (start_x, start_y), (end_x, end_y), 2, vessel_thickness)
        
        # Add some noise to make it more realistic
        noise_mask = np.random.random((256, 256)) < 0.02
        mask[noise_mask] = np.random.choice([0, 1, 2], size=np.sum(noise_mask))
        
        # Apply slight blur to make boundaries more realistic
        mask = cv2.GaussianBlur(mask.astype(np.float32), (3, 3), 0)
        mask = np.round(mask).astype(np.uint8)
        
        return Image.fromarray(mask)
    
    def process_and_split_dataset(self, train_ratio: float = 0.8) -> bool:
        """Process dataset and split into train/validation"""
        logger.info("🔄 Processing and splitting dataset...")
        
        # Check if real dataset exists first
        real_images_dir = self.output_dir / "kvasir_seg" / "Kvasir-SEG" / "images"
        real_masks_dir = self.output_dir / "kvasir_seg" / "Kvasir-SEG" / "masks"
        
        # Also check if dataset was downloaded to raw_data_dir
        if not real_images_dir.exists():
            real_images_dir = self.raw_data_dir / "Kvasir-SEG" / "images"
            real_masks_dir = self.raw_data_dir / "Kvasir-SEG" / "masks"
        
        if real_images_dir.exists() and real_masks_dir.exists():
            logger.info("📁 Using real Kvasir-SEG dataset")
            images_dir = real_images_dir
            masks_dir = real_masks_dir
        else:
            logger.error("❌ Real dataset not found! Please ensure Kvasir-SEG dataset is available.")
            logger.error("Expected location: data/kvasir_seg/Kvasir-SEG/")
            logger.error("Download from: https://datasets.simula.no/downloads/kvasir-seg.zip")
            return False
        
        image_files = list(images_dir.glob("*.jpg"))
        mask_files = list(masks_dir.glob("*.jpg"))
        
        if not image_files:
            logger.error("No images found in dataset!")
            return False
        
        # Match images with masks
        matched_pairs = []
        for img_file in image_files:
            # For real dataset, mask has same name as image
            if real_images_dir.exists():
                mask_file = masks_dir / img_file.name
            else:
                # For synthetic dataset, mask has different naming
                mask_file = masks_dir / f"mask_{img_file.stem.split('_')[1]}.jpg"
            
            if mask_file.exists():
                matched_pairs.append((img_file, mask_file))
        
        logger.info(f"Found {len(matched_pairs)} matched image-mask pairs")
        
        # Split into train/validation
        train_pairs, val_pairs = train_test_split(
            matched_pairs, 
            train_size=train_ratio, 
            random_state=42
        )
        
        # Copy files to train/validation directories
        self._copy_dataset_split(train_pairs, self.train_dir, "train")
        self._copy_dataset_split(val_pairs, self.val_dir, "validation")
        
        # Create metadata files
        self._create_metadata(train_pairs, self.train_dir, "train")
        self._create_metadata(val_pairs, self.val_dir, "validation")
        
        logger.info(f"✅ Dataset split: {len(train_pairs)} train, {len(val_pairs)} validation")
        return True
    
    def _copy_dataset_split(self, pairs: List[Tuple[Path, Path]], target_dir: Path, split_name: str):
        """Copy dataset split to target directory"""
        images_dir = target_dir / "images"
        masks_dir = target_dir / "masks"
        images_dir.mkdir(exist_ok=True)
        masks_dir.mkdir(exist_ok=True)
        
        for i, (img_file, mask_file) in enumerate(pairs):
            # Copy image
            new_img_name = f"{split_name}_{i:03d}.jpg"
            shutil.copy2(img_file, images_dir / new_img_name)
            
            # Copy mask
            new_mask_name = f"{split_name}_mask_{i:03d}.jpg"
            shutil.copy2(mask_file, masks_dir / new_mask_name)
    
    def _create_metadata(self, pairs: List[Tuple[Path, Path]], target_dir: Path, split_name: str):
        """Create metadata for dataset split"""
        episodes = []
        
        for i, (img_file, mask_file) in enumerate(pairs):
            # Create episode data
            episode = {
                "episode_id": f"{split_name}_ep_{i:03d}",
                "images": [f"{split_name}_{i:03d}.jpg"],
                "actions": [self._generate_sample_actions()],
                "rewards": [np.random.uniform(0.1, 1.0)],
                "success": np.random.random() > 0.3,  # 70% success rate
                "liquid_reduction": np.random.uniform(0.6, 0.9),
                "contaminant_reduction": np.random.uniform(0.5, 0.8)
            }
            episodes.append(episode)
        
        # Save metadata
        metadata = {"episodes": episodes}
        with open(target_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _generate_sample_actions(self) -> List[float]:
        """Generate sample action sequence"""
        return [
            np.random.uniform(-0.05, 0.05),  # dx
            np.random.uniform(-0.05, 0.05),  # dy
            np.random.uniform(-0.02, 0.02),  # dz
            np.random.uniform(-0.1, 0.1),    # dyaw
            np.random.choice([0.0, 1.0])     # suction_toggle
        ]
    
    def get_train_dir(self) -> Path:
        """Get training directory path"""
        return self.train_dir
    
    def get_val_dir(self) -> Path:
        """Get validation directory path"""
        return self.val_dir
