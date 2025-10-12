"""
Video recording utilities for evaluation.
"""

import cv2
import numpy as np
from typing import Optional, List
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class VideoRecorder:
    """
    Video recorder for evaluation episodes.
    """
    
    def __init__(self, output_dir: str, fps: int = 30, resolution: tuple = (640, 480)):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.resolution = resolution
        
        self.writer = None
        self.current_file = None
        self.frame_count = 0
    
    def start_recording(self, filepath: str) -> None:
        """Start recording to file."""
        if self.writer is not None:
            self.stop_recording()
        
        self.current_file = filepath
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            filepath, 
            fourcc, 
            self.fps, 
            self.resolution
        )
        self.frame_count = 0
        
        logger.info(f"Started recording to {filepath}")
    
    def record_frame(self, frame: np.ndarray) -> None:
        """Record a single frame."""
        if self.writer is None:
            logger.warning("No active recording session")
            return
        
        # Ensure frame is in correct format
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        
        # Resize frame if necessary
        if frame.shape[:2] != self.resolution[::-1]:
            frame = cv2.resize(frame, self.resolution)
        
        # Ensure frame is BGR
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        self.writer.write(frame)
        self.frame_count += 1
    
    def stop_recording(self) -> None:
        """Stop recording and save file."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            
            logger.info(f"Stopped recording. Saved {self.frame_count} frames to {self.current_file}")
            self.current_file = None
            self.frame_count = 0
    
    def __del__(self):
        """Destructor to ensure recording is stopped."""
        self.stop_recording()


class FrameComposer:
    """
    Compose frames with multiple views for video recording.
    """
    
    def __init__(self, layout: str = "single"):
        self.layout = layout
    
    def compose_frame(
        self, 
        main_image: np.ndarray, 
        aux_data: Optional[Dict] = None,
        action: Optional[np.ndarray] = None,
        safety_info: Optional[Dict] = None
    ) -> np.ndarray:
        """Compose a frame with multiple views."""
        if self.layout == "single":
            return self._compose_single_view(main_image, aux_data, action, safety_info)
        elif self.layout == "split":
            return self._compose_split_view(main_image, aux_data, action, safety_info)
        elif self.layout == "grid":
            return self._compose_grid_view(main_image, aux_data, action, safety_info)
        else:
            return main_image
    
    def _compose_single_view(
        self, 
        main_image: np.ndarray, 
        aux_data: Optional[Dict] = None,
        action: Optional[np.ndarray] = None,
        safety_info: Optional[Dict] = None
    ) -> np.ndarray:
        """Compose single view with overlay information."""
        # Convert to displayable format
        if main_image.shape[0] == 3:  # CHW format
            frame = np.transpose(main_image, (1, 2, 0))
        else:
            frame = main_image.copy()
        
        # Ensure proper data type
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        
        # Resize for display
        frame = cv2.resize(frame, (640, 480))
        
        # Add text overlay
        y_offset = 30
        line_height = 25
        
        # Action information
        if action is not None:
            action_text = f"Action: [{action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f}, {action[3]:.3f}, {action[4]:.1f}]"
            cv2.putText(frame, action_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += line_height
        
        # Safety information
        if safety_info is not None:
            safety_level = safety_info.get('safety_level', 'unknown')
            distance = safety_info.get('distance', 0)
            action_scale = safety_info.get('action_scale', 1.0)
            
            safety_text = f"Safety: {safety_level} | Dist: {distance:.3f} | Scale: {action_scale:.3f}"
            color = (0, 255, 0) if safety_level == 'safe' else (0, 255, 255) if safety_level == 'warning' else (0, 0, 255)
            cv2.putText(frame, safety_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += line_height
        
        # Auxiliary data
        if aux_data is not None:
            liquid_mass = aux_data.get('liquid_mass_remaining', 0)
            contaminant_mass = aux_data.get('contaminant_mass_remaining', 0)
            collisions = aux_data.get('collisions', 0)
            
            aux_text = f"Liquid: {liquid_mass:.3f} | Contaminant: {contaminant_mass:.3f} | Collisions: {collisions}"
            cv2.putText(frame, aux_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += line_height
        
        return frame
    
    def _compose_split_view(
        self, 
        main_image: np.ndarray, 
        aux_data: Optional[Dict] = None,
        action: Optional[np.ndarray] = None,
        safety_info: Optional[Dict] = None
    ) -> np.ndarray:
        """Compose split view with main image and information panel."""
        # Main image
        if main_image.shape[0] == 3:  # CHW format
            main_frame = np.transpose(main_image, (1, 2, 0))
        else:
            main_frame = main_image.copy()
        
        if main_frame.dtype != np.uint8:
            main_frame = (main_frame * 255).astype(np.uint8)
        
        main_frame = cv2.resize(main_frame, (480, 360))
        
        # Information panel
        info_panel = np.zeros((360, 160, 3), dtype=np.uint8)
        
        y_offset = 30
        line_height = 25
        
        # Action information
        if action is not None:
            cv2.putText(info_panel, "Action:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += line_height
            for i, val in enumerate(action):
                cv2.putText(info_panel, f"  {i}: {val:.3f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                y_offset += 20
        
        # Safety information
        if safety_info is not None:
            y_offset += 10
            cv2.putText(info_panel, "Safety:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += line_height
            
            safety_level = safety_info.get('safety_level', 'unknown')
            color = (0, 255, 0) if safety_level == 'safe' else (0, 255, 255) if safety_level == 'warning' else (0, 0, 255)
            cv2.putText(info_panel, f"  Level: {safety_level}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            y_offset += 20
            
            distance = safety_info.get('distance', 0)
            cv2.putText(info_panel, f"  Dist: {distance:.3f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 20
            
            action_scale = safety_info.get('action_scale', 1.0)
            cv2.putText(info_panel, f"  Scale: {action_scale:.3f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Combine frames
        combined_frame = np.hstack([main_frame, info_panel])
        
        return combined_frame
    
    def _compose_grid_view(
        self, 
        main_image: np.ndarray, 
        aux_data: Optional[Dict] = None,
        action: Optional[np.ndarray] = None,
        safety_info: Optional[Dict] = None
    ) -> np.ndarray:
        """Compose grid view with multiple information panels."""
        # Main image (top-left)
        if main_image.shape[0] == 3:  # CHW format
            main_frame = np.transpose(main_image, (1, 2, 0))
        else:
            main_frame = main_image.copy()
        
        if main_frame.dtype != np.uint8:
            main_frame = (main_frame * 255).astype(np.uint8)
        
        main_frame = cv2.resize(main_frame, (320, 240))
        
        # Action visualization (top-right)
        action_frame = self._create_action_visualization(action)
        
        # Safety visualization (bottom-left)
        safety_frame = self._create_safety_visualization(safety_info)
        
        # Data visualization (bottom-right)
        data_frame = self._create_data_visualization(aux_data)
        
        # Combine into grid
        top_row = np.hstack([main_frame, action_frame])
        bottom_row = np.hstack([safety_frame, data_frame])
        combined_frame = np.vstack([top_row, bottom_row])
        
        return combined_frame
    
    def _create_action_visualization(self, action: Optional[np.ndarray]) -> np.ndarray:
        """Create action visualization panel."""
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        
        if action is None:
            cv2.putText(frame, "No Action Data", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            return frame
        
        # Action bar chart
        action_names = ['dx', 'dy', 'dz', 'dyaw', 'suction']
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
        bar_width = 40
        bar_spacing = 50
        start_x = 20
        
        for i, (name, value, color) in enumerate(zip(action_names, action, colors)):
            x = start_x + i * bar_spacing
            height = int(abs(value) * 100) + 10
            y = 200 - height
            
            # Draw bar
            cv2.rectangle(frame, (x, y), (x + bar_width, 200), color, -1)
            
            # Draw value
            cv2.putText(frame, f"{value:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Draw label
            cv2.putText(frame, name, (x, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.putText(frame, "Action Visualization", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def _create_safety_visualization(self, safety_info: Optional[Dict]) -> np.ndarray:
        """Create safety visualization panel."""
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        
        if safety_info is None:
            cv2.putText(frame, "No Safety Data", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            return frame
        
        # Safety level indicator
        safety_level = safety_info.get('safety_level', 'unknown')
        distance = safety_info.get('distance', 0)
        action_scale = safety_info.get('action_scale', 1.0)
        
        # Color based on safety level
        if safety_level == 'safe':
            color = (0, 255, 0)
        elif safety_level == 'warning':
            color = (0, 255, 255)
        elif safety_level == 'critical':
            color = (0, 0, 255)
        else:
            color = (128, 128, 128)
        
        # Draw safety circle
        center = (160, 120)
        radius = int(distance * 200) + 20
        cv2.circle(frame, center, radius, color, 3)
        cv2.circle(frame, center, 10, color, -1)
        
        # Draw text
        cv2.putText(frame, f"Safety: {safety_level}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"Distance: {distance:.3f}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Scale: {action_scale:.3f}", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def _create_data_visualization(self, aux_data: Optional[Dict]) -> np.ndarray:
        """Create data visualization panel."""
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        
        if aux_data is None:
            cv2.putText(frame, "No Data", (100, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            return frame
        
        # Extract data
        liquid_mass = aux_data.get('liquid_mass_remaining', 1.0)
        contaminant_mass = aux_data.get('contaminant_mass_remaining', 0.5)
        collisions = aux_data.get('collisions', 0)
        
        # Draw progress bars
        y_start = 50
        bar_width = 200
        bar_height = 20
        bar_spacing = 40
        
        # Liquid mass bar
        liquid_progress = int((1.0 - liquid_mass) * bar_width)
        cv2.rectangle(frame, (60, y_start), (60 + bar_width, y_start + bar_height), (100, 100, 100), -1)
        cv2.rectangle(frame, (60, y_start), (60 + liquid_progress, y_start + bar_height), (0, 255, 0), -1)
        cv2.putText(frame, f"Liquid: {liquid_mass:.3f}", (10, y_start + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Contaminant mass bar
        contaminant_progress = int((1.0 - contaminant_mass) * bar_width)
        cv2.rectangle(frame, (60, y_start + bar_spacing), (60 + bar_width, y_start + bar_spacing + bar_height), (100, 100, 100), -1)
        cv2.rectangle(frame, (60, y_start + bar_spacing), (60 + contaminant_progress, y_start + bar_spacing + bar_height), (0, 0, 255), -1)
        cv2.putText(frame, f"Contaminant: {contaminant_mass:.3f}", (10, y_start + bar_spacing + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Collisions indicator
        collision_color = (0, 0, 255) if collisions > 0 else (0, 255, 0)
        cv2.circle(frame, (280, y_start + 2 * bar_spacing), 15, collision_color, -1)
        cv2.putText(frame, f"Collisions: {collisions}", (10, y_start + 2 * bar_spacing + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.putText(frame, "Data Visualization", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
