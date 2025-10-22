"""
Video recording utilities.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class VideoRecorder:
    """Video recorder for environment episodes."""
    
    def __init__(
        self,
        output_path: Union[str, Path],
        fps: int = 30,
        codec: str = 'mp4v',
        frame_size: Optional[tuple] = None
    ):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.fps = fps
        self.codec = codec
        self.frame_size = frame_size
        
        self.writer = None
        self.frames = []
        
        logger.info(f"VideoRecorder initialized: {output_path}")
    
    def start_recording(self, frame_size: Optional[tuple] = None):
        """Start recording."""
        if frame_size is None:
            frame_size = self.frame_size
        
        if frame_size is None:
            raise ValueError("Frame size must be specified")
        
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            frame_size
        )
        
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {self.output_path}")
        
        logger.debug(f"Started recording: {self.output_path}")
    
    def add_frame(self, frame: np.ndarray):
        """Add frame to video."""
        if self.writer is None:
            self.frames.append(frame.copy())
        else:
            # Ensure frame is in correct format
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            
            # Convert RGB to BGR if needed
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            self.writer.write(frame)
    
    def stop_recording(self):
        """Stop recording and save video."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            logger.debug(f"Stopped recording: {self.output_path}")
        elif self.frames:
            # Write frames to video file
            self._write_frames_to_video()
    
    def _write_frames_to_video(self):
        """Write collected frames to video file."""
        if not self.frames:
            return
        
        # Determine frame size from first frame
        frame_size = (self.frames[0].shape[1], self.frames[0].shape[0])
        
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            frame_size
        )
        
        for frame in self.frames:
            # Ensure frame is in correct format
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            
            # Convert RGB to BGR if needed
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            writer.write(frame)
        
        writer.release()
        self.frames.clear()
        logger.debug(f"Wrote {len(self.frames)} frames to video: {self.output_path}")


def save_video(
    frames: List[np.ndarray],
    output_path: Union[str, Path],
    fps: int = 30,
    codec: str = 'mp4v'
) -> None:
    """Save list of frames as video."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not frames:
        logger.warning("No frames to save")
        return
    
    # Determine frame size from first frame
    frame_size = (frames[0].shape[1], frames[0].shape[0])
    
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        frame_size
    )
    
    for frame in frames:
        # Ensure frame is in correct format
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        
        # Convert RGB to BGR if needed
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        writer.write(frame)
    
    writer.release()
    logger.info(f"Saved video with {len(frames)} frames: {output_path}")


def create_video_from_images(
    image_paths: List[Path],
    output_path: Union[str, Path],
    fps: int = 30,
    codec: str = 'mp4v'
) -> None:
    """Create video from list of image files."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not image_paths:
        logger.warning("No images to create video from")
        return
    
    # Load first image to determine frame size
    first_image = cv2.imread(str(image_paths[0]))
    if first_image is None:
        raise ValueError(f"Failed to load image: {image_paths[0]}")
    
    frame_size = (first_image.shape[1], first_image.shape[0])
    
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        frame_size
    )
    
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning(f"Failed to load image: {image_path}")
            continue
        
        writer.write(image)
    
    writer.release()
    logger.info(f"Created video from {len(image_paths)} images: {output_path}")
