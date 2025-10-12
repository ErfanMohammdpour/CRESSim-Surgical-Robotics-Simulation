"""
Data utilities for downloading and managing datasets.
"""

import os
import zipfile
import requests
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


def download_file(url: str, destination: Path, force: bool = False) -> None:
    """Download file from URL with progress bar."""
    if destination.exists() and not force:
        logger.info(f"File already exists: {destination}")
        return
    
    logger.info(f"Downloading {url} to {destination}")
    
    # Create parent directory
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    # Download with progress bar (disable SSL verification for Windows)
    response = requests.get(url, stream=True, verify=False)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=destination.name) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def extract_zip(zip_path: Path, extract_to: Path, force: bool = False) -> None:
    """Extract ZIP file."""
    if extract_to.exists() and not force:
        logger.info(f"Directory already exists: {extract_to}")
        return
    
    logger.info(f"Extracting {zip_path} to {extract_to}")
    
    extract_to.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)


def download_kvasir_seg(data_dir: str, force: bool = False) -> None:
    """Download and extract Kvasir-SEG dataset."""
    data_dir = Path(data_dir)
    zip_path = data_dir / "Kvasir-SEG.zip"
    extract_dir = data_dir / "Kvasir-SEG"
    
    # Download if needed
    if not zip_path.exists() or force:
        url = "https://datasets.simula.no/downloads/kvasir-seg.zip"
        download_file(url, zip_path, force=force)
    
    # Extract if needed
    if not extract_dir.exists() or force:
        extract_zip(zip_path, data_dir, force=force)
    
    # Validate extraction
    images_dir = extract_dir / "images"
    masks_dir = extract_dir / "masks"
    
    if not images_dir.exists() or not masks_dir.exists():
        raise RuntimeError("Invalid Kvasir-SEG extraction")
    
    # Count images
    image_count = len(list(images_dir.glob("*.jpg")))
    mask_count = len(list(masks_dir.glob("*.jpg")))
    
    logger.info(f"Kvasir-SEG dataset ready:")
    logger.info(f"  Images: {image_count}")
    logger.info(f"  Masks: {mask_count}")
    
    if image_count < 1000:
        raise RuntimeError(f"Expected >1000 images, got {image_count}")
    
    # Clean up zip file
    if zip_path.exists():
        zip_path.unlink()


def validate_dataset_structure(data_dir: str, dataset_name: str) -> bool:
    """Validate dataset directory structure."""
    data_dir = Path(data_dir)
    
    if dataset_name == "kvasir_seg":
        required_dirs = ["images", "masks"]
        dataset_path = data_dir / "Kvasir-SEG"
    elif dataset_name == "endoscapes":
        required_dirs = ["train", "val", "test"]
        dataset_path = data_dir
    elif dataset_name == "cholec80":
        required_dirs = ["videos", "annotations"]
        dataset_path = data_dir
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    if not dataset_path.exists():
        return False
    
    for required_dir in required_dirs:
        if not (dataset_path / required_dir).exists():
            return False
    
    return True


def get_dataset_info(data_dir: str, dataset_name: str) -> dict:
    """Get information about a dataset."""
    data_dir = Path(data_dir)
    
    if dataset_name == "kvasir_seg":
        dataset_path = data_dir / "Kvasir-SEG"
        images_dir = dataset_path / "images"
        masks_dir = dataset_path / "masks"
        
        if not dataset_path.exists():
            return {"status": "not_found"}
        
        image_count = len(list(images_dir.glob("*.jpg"))) if images_dir.exists() else 0
        mask_count = len(list(masks_dir.glob("*.jpg"))) if masks_dir.exists() else 0
        
        return {
            "status": "found",
            "images": image_count,
            "masks": mask_count,
            "path": str(dataset_path)
        }
    
    elif dataset_name == "endoscapes":
        dataset_path = data_dir
        if not dataset_path.exists():
            return {"status": "not_found"}
        
        # Count files in subdirectories
        info = {"status": "found", "path": str(dataset_path)}
        for split in ["train", "val", "test"]:
            split_dir = dataset_path / split
            if split_dir.exists():
                info[split] = len(list(split_dir.rglob("*.jpg")))
            else:
                info[split] = 0
        
        return info
    
    elif dataset_name == "cholec80":
        dataset_path = data_dir
        if not dataset_path.exists():
            return {"status": "not_found"}
        
        videos_dir = dataset_path / "videos"
        annotations_dir = dataset_path / "annotations"
        
        video_count = len(list(videos_dir.glob("*.mp4"))) if videos_dir.exists() else 0
        annotation_count = len(list(annotations_dir.glob("*.txt"))) if annotations_dir.exists() else 0
        
        return {
            "status": "found",
            "videos": video_count,
            "annotations": annotation_count,
            "path": str(dataset_path)
        }
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
