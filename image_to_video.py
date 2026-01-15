"""
Image to Video Converter Module
Convert sequence of images with timestamps to video file
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

def calculate_fps_from_timestamps(timestamp_list: List[int]) -> float:
    """
    Calculate FPS from timestamp differences.
    
    Args:
        timestamp_list: List of timestamps in milliseconds
        
    Returns:
        FPS value (float)
    """
    if len(timestamp_list) < 2:
        return 30.0  # Default FPS
    
    # Calculate differences between consecutive timestamps (in seconds)
    diffs = []
    for i in range(1, len(timestamp_list)):
        dt = (timestamp_list[i] - timestamp_list[i-1]) / 1000.0  # Convert ms to seconds
        if dt > 0:  # Only positive differences
            diffs.append(dt)
    
    if not diffs:
        return 30.0
    
    # Use median to avoid outliers
    median_dt = np.median(diffs)
    
    # Calculate FPS and clamp to reasonable range
    fps = 1.0 / median_dt if median_dt > 0 else 30.0
    fps = min(60.0, max(1.0, fps))  # Clamp between 1 and 60
    
    logger.info(f"Calculated FPS: {fps:.2f} (median frame interval: {median_dt:.3f}s)")
    return fps


def get_sorted_images(image_dir: Path) -> tuple[List[Path], List[int]]:
    """
    Get sorted list of images from directory.
    
    Args:
        image_dir: Path to image directory
        
    Returns:
        Tuple of (sorted file paths, timestamps)
    """
    image_files = sorted(image_dir.glob("*.jpg"))
    
    # Extract timestamps from filenames
    timestamps = []
    for file in image_files:
        try:
            ts = int(file.stem)  # Filename without extension
            timestamps.append(ts)
        except ValueError:
            logger.warning(f"Skipping file with invalid timestamp: {file.name}")
            continue
    
    logger.info(f"Found {len(image_files)} images in {image_dir.name}")
    return image_files, timestamps


def create_video_from_images(
    image_dir: Path,
    output_path: Path,
    auto_fps: bool = True,
    fps: float = 30.0,
    codec: str = "mp4v"
) -> bool:
    """
    Create video from sequence of images.
    
    Args:
        image_dir: Path to directory containing images
        output_path: Path for output video file
        auto_fps: Whether to calculate FPS from timestamps
        fps: Fixed FPS if auto_fps is False (default: 30.0)
        codec: Video codec (default: "mp4v" for H.264)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        image_files, timestamps = get_sorted_images(image_dir)
        
        if len(image_files) == 0:
            logger.error("No images found in directory")
            return False
        
        # Read first image to get dimensions
        first_frame = cv2.imread(str(image_files[0]))
        if first_frame is None:
            logger.error(f"Cannot read first image: {image_files[0]}")
            return False
        
        height, width = first_frame.shape[:2]
        logger.info(f"Video dimensions: {width}x{height}")
        
        # Calculate or set FPS
        if auto_fps and len(timestamps) > 1:
            video_fps = calculate_fps_from_timestamps(timestamps)
        else:
            video_fps = fps
            logger.info(f"Using fixed FPS: {video_fps}")
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(
            str(output_path),
            fourcc,
            video_fps,
            (width, height)
        )
        
        if not out.isOpened():
            logger.error("Cannot create video writer")
            return False
        
        # Write frames
        logger.info("Writing frames to video...")
        for i, image_file in enumerate(image_files):
            frame = cv2.imread(str(image_file))
            if frame is None:
                logger.warning(f"Skipping unreadable frame: {image_file.name}")
                continue
            
            # Resize if necessary
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            
            out.write(frame)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(image_files)} frames")
        
        out.release()
        
        logger.info(f"Video created successfully: {output_path}")
        logger.info(f"Total frames: {len(image_files)}, FPS: {video_fps:.2f}")
        
        # Calculate and log video duration
        duration_seconds = len(image_files) / video_fps
        logger.info(f"Video duration: {duration_seconds:.2f}s ({duration_seconds/60:.2f}m)")
        
        return True
        
    except Exception as e:
        logger.error(f"Error creating video: {str(e)}", exc_info=True)
        return False


def create_video_from_device_folder(
    device_id: str,
    base_image_dir: str = "image",
    output_dir: str = "output",
    auto_fps: bool = True,
    fps: float = 30.0
) -> Optional[Path]:
    """
    Create video from device-specific image folder.
    
    Args:
        device_id: Device ID (e.g., "IMAGE_001")
        base_image_dir: Base directory containing device folders
        output_dir: Directory for output video
        auto_fps: Whether to calculate FPS from timestamps
        fps: Fixed FPS if auto_fps is False
        
    Returns:
        Path to created video, or None if failed
    """
    image_dir = Path(base_image_dir) / device_id
    
    if not image_dir.exists():
        logger.error(f"Image directory not found: {image_dir}")
        return None
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename
    video_file = output_path / f"{device_id}_{video_file.suffix or 'video'}.mp4"
    
    logger.info(f"Creating video for device: {device_id}")
    
    if create_video_from_images(
        image_dir,
        video_file,
        auto_fps=auto_fps,
        fps=fps
    ):
        return video_file
    
    return None


if __name__ == "__main__":
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    if len(sys.argv) > 1:
        device_id = sys.argv[1]
    else:
        device_id = "IMAGE_001"
    
    video_path = create_video_from_device_folder(
        device_id=device_id,
        auto_fps=True
    )
    
    if video_path:
        print(f"\n✓ Video created: {video_path}")
    else:
        print("\n✗ Failed to create video")
