"""
Data utilities for downloading and processing datasets.
"""

import os
import zipfile
import requests
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def download_kvasir_seg(output_dir: Path, force: bool = False) -> None:
    """Download Kvasir-SEG dataset."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = output_dir / "Kvasir-SEG.zip"
    extract_dir = output_dir / "Kvasir-SEG"
    
    # Check if already downloaded
    if not force and extract_dir.exists():
        logger.info("Kvasir-SEG dataset already exists")
        return
    
    # Download dataset
    url = "https://datasets.simula.no/downloads/kvasir-seg.zip"
    logger.info(f"Downloading Kvasir-SEG from {url}")
    
    try:
        response = requests.get(url, stream=True, verify=False)
        response.raise_for_status()
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Downloaded {zip_path}")
        
        # Extract dataset
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        
        logger.info(f"Extracted to {extract_dir}")
        
        # Remove zip file
        zip_path.unlink()
        
        # Create manifest
        create_manifest(extract_dir)
        
    except Exception as e:
        logger.error(f"Failed to download Kvasir-SEG: {e}")
        raise


def create_manifest(dataset_dir: Path) -> None:
    """Create manifest file for dataset."""
    manifest = {
        'dataset': 'Kvasir-SEG',
        'images': [],
        'masks': [],
        'total_images': 0,
        'total_masks': 0,
        'checksums': {}
    }
    
    images_dir = dataset_dir / "images"
    masks_dir = dataset_dir / "masks"
    
    if images_dir.exists():
        for img_file in images_dir.glob("*.jpg"):
            manifest['images'].append(img_file.name)
            manifest['checksums'][img_file.name] = calculate_file_hash(img_file)
    
    if masks_dir.exists():
        for mask_file in masks_dir.glob("*.jpg"):
            manifest['masks'].append(mask_file.name)
            manifest['checksums'][mask_file.name] = calculate_file_hash(mask_file)
    
    manifest['total_images'] = len(manifest['images'])
    manifest['total_masks'] = len(manifest['masks'])
    
    # Save manifest
    import json
    manifest_file = dataset_dir / "manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"Created manifest: {manifest_file}")
    logger.info(f"Total images: {manifest['total_images']}")
    logger.info(f"Total masks: {manifest['total_masks']}")


def calculate_file_hash(file_path: Path) -> str:
    """Calculate SHA256 hash of file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def verify_dataset(dataset_dir: Path) -> bool:
    """Verify dataset integrity."""
    dataset_dir = Path(dataset_dir)
    manifest_file = dataset_dir / "manifest.json"
    
    if not manifest_file.exists():
        logger.error("Manifest file not found")
        return False
    
    with open(manifest_file, 'r') as f:
        manifest = json.load(f)
    
    # Check if we have enough images
    if manifest['total_images'] < 1000:
        logger.warning(f"Dataset has only {manifest['total_images']} images (expected >= 1000)")
        return False
    
    # Verify checksums
    for filename, expected_hash in manifest['checksums'].items():
        file_path = dataset_dir / "images" / filename
        if not file_path.exists():
            file_path = dataset_dir / "masks" / filename
        
        if file_path.exists():
            actual_hash = calculate_file_hash(file_path)
            if actual_hash != expected_hash:
                logger.error(f"Checksum mismatch for {filename}")
                return False
    
    logger.info("Dataset verification passed")
    return True
