"""
Test script for image to video converter
Demonstrates how to use the image_to_video module
"""

import sys
import os
from pathlib import Path
import logging

# Add parent directory to path so we can import image_to_video
sys.path.insert(0, str(Path(__file__).parent.parent))

from image_to_video import (
    create_video_from_images,
    create_video_from_device_folder,
    get_sorted_images,
    calculate_fps_from_timestamps
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_calculate_fps():
    """Test FPS calculation from timestamps"""
    logger.info("=" * 60)
    logger.info("Test 1: FPS Calculation")
    logger.info("=" * 60)
    
    # Simulate timestamps: 200ms interval (5 FPS)
    timestamps = [i * 200 for i in range(10)]  # 0, 200, 400, 600, ...
    fps = calculate_fps_from_timestamps(timestamps)
    logger.info(f"Input timestamps (ms): {timestamps[:5]}...")
    logger.info(f"Calculated FPS: {fps:.2f}")
    logger.info(f"Expected: ~5 FPS\n")


def test_get_sorted_images():
    """Test image collection and sorting"""
    logger.info("=" * 60)
    logger.info("Test 2: Image Collection and Sorting")
    logger.info("=" * 60)
    
    image_dir = Path(__file__).parent.parent / "image" / "IMAGE_001"
    
    if not image_dir.exists():
        logger.error(f"Image directory not found: {image_dir}")
        return
    
    files, timestamps = get_sorted_images(image_dir)
    
    logger.info(f"Total images: {len(files)}")
    logger.info(f"First image: {files[0].name} (timestamp: {timestamps[0]})")
    logger.info(f"Last image: {files[-1].name} (timestamp: {timestamps[-1]})")
    
    # Calculate duration
    duration_ms = timestamps[-1] - timestamps[0]
    logger.info(f"Time span: {duration_ms}ms ({duration_ms/1000:.2f}s)\n")


def test_create_video():
    """Test video creation"""
    logger.info("=" * 60)
    logger.info("Test 3: Video Creation")
    logger.info("=" * 60)
    
    # Set up paths
    image_dir = Path(__file__).parent.parent / "image" / "IMAGE_001"
    output_dir = Path(__file__).parent / "output"
    output_video = output_dir / "test_video.mp4"
    
    if not image_dir.exists():
        logger.error(f"Image directory not found: {image_dir}")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Input images: {image_dir}")
    logger.info(f"Output video: {output_video}")
    logger.info("Creating video (auto FPS from timestamps)...\n")
    
    success = create_video_from_images(
        image_dir=image_dir,
        output_path=output_video,
        auto_fps=True,
        codec="mp4v"
    )
    
    if success:
        # Check file size
        file_size_mb = output_video.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Video created successfully!")
        logger.info(f"✓ File size: {file_size_mb:.2f} MB\n")
        return output_video
    else:
        logger.error("✗ Failed to create video\n")
        return None


def test_create_video_from_device():
    """Test device folder to video conversion"""
    logger.info("=" * 60)
    logger.info("Test 4: Device Folder Conversion")
    logger.info("=" * 60)
    
    # Change to parent directory to use relative paths
    original_cwd = os.getcwd()
    test_dir = Path(__file__).parent.parent
    os.chdir(test_dir)
    
    try:
        logger.info("Converting IMAGE_001 device folder to video...")
        
        video_path = create_video_from_device_folder(
            device_id="IMAGE_001",
            base_image_dir="image",
            output_dir=str(Path(__file__).parent / "output"),
            auto_fps=True
        )
        
        if video_path:
            logger.info(f"✓ Video created: {video_path}\n")
            return video_path
        else:
            logger.error("✗ Failed to create video from device folder\n")
            return None
    finally:
        os.chdir(original_cwd)


def main():
    """Run all tests"""
    logger.info("\n")
    logger.info("╔" + "=" * 58 + "╗")
    logger.info("║" + " " * 58 + "║")
    logger.info("║" + "  Image to Video Converter - Test Suite".center(58) + "║")
    logger.info("║" + " " * 58 + "║")
    logger.info("╚" + "=" * 58 + "╝")
    logger.info("\n")
    
    try:
        # Test 1: FPS Calculation
        test_calculate_fps()
        
        # Test 2: Image Collection
        test_get_sorted_images()
        
        # Test 3: Video Creation
        test_create_video()
        
        # Summary
        logger.info("=" * 60)
        logger.info("Test Summary")
        logger.info("=" * 60)
        logger.info("✓ All tests completed!")
        logger.info("\nOutput location:")
        logger.info(f"  {Path(__file__).parent / 'output'}\n")
        
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()
