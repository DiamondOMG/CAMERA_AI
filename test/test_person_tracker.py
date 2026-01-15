"""
Test Person Tracker on Image Sequence and Video

Tests the person tracking functionality on:
1. Image sequence from image/IMAGE_001 folder
2. (Optional) Video files if available
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from person_tracker import PersonTracker, process_images, process_video


def test_tracking_on_images():
    """Test tracking on sequential images"""
    print("=" * 80)
    print("TEST 1: Person Tracking on Image Sequence")
    print("=" * 80 + "\n")
    
    # Setup paths
    img_dir = Path(__file__).parent.parent / "image" / "IMAGE_001"
    output_dir = Path(__file__).parent / "output" / "tracked_images"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if not img_dir.exists():
        print(f"❌ Image directory not found: {img_dir}")
        return
    
    # Get all images sorted by filename
    image_files = sorted(
        img_dir.glob("*.jpg"),
        key=lambda p: int(p.stem)
    )[:100]  # Test with first 100 images
    
    if not image_files:
        print(f"❌ No images found in {img_dir}")
        return
    
    print(f"✓ Found {len(image_files)} images\n")
    
    # Process images
    try:
        summary = process_images(
            image_paths=[str(p) for p in image_files],
            output_dir=str(output_dir),
            confidence=0.5
        )
        
        # Print summary
        print("\n" + "=" * 80)
        print("TRACKING SUMMARY")
        print("=" * 80)
        print(f"\n📊 Total unique persons detected: {summary['total_unique_persons']}\n")
        
        if summary['persons']:
            print("Person Details:")
            print("-" * 80)
            for person in summary['persons']:
                print(f"  Person #{person['id']}:")
                print(f"    - First seen: {person['first_seen']}")
                print(f"    - Last seen: {person['last_seen']}")
                print(f"    - Duration: {person['duration_seconds']}s")
                print(f"    - Appeared in {person['total_frames']} frames")
                print()
        
        # Save summary to JSON
        summary_file = Path(__file__).parent / "output" / "tracking_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Summary saved to: {summary_file}")
        print(f"✓ Annotated images saved to: {output_dir}\n")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nMake sure you have installed:")
        print("  pip install ultralytics supervision")
        return
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return


def test_tracking_on_video():
    """Test tracking on video file (if available)"""
    print("\n" + "=" * 80)
    print("TEST 2: Person Tracking on Video")
    print("=" * 80 + "\n")
    
    # Look for video files in test directory
    test_dir = Path(__file__).parent
    video_files = list(test_dir.glob("*.mp4")) + list(test_dir.glob("*.avi"))
    
    if not video_files:
        print("ℹ️  No video files found in test directory")
        print(f"   To test video tracking, add a video file to: {test_dir}")
        print("   Supported formats: .mp4, .avi\n")
        return
    
    video_path = video_files[0]
    output_path = test_dir / "output" / "tracked_video.mp4"
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    print(f"📹 Testing with video: {video_path.name}\n")
    
    try:
        summary = process_video(
            video_path=str(video_path),
            output_path=str(output_path),
            confidence=0.5,
            display=False  # Set to True to see real-time display
        )
        
        # Print summary
        print("\n" + "=" * 80)
        print("VIDEO TRACKING SUMMARY")
        print("=" * 80)
        print(f"\n📊 Total unique persons detected: {summary['total_unique_persons']}\n")
        
        if summary['persons']:
            print("Person Details:")
            print("-" * 80)
            for person in summary['persons']:
                print(f"  Person #{person['id']}:")
                print(f"    - Duration: {person['duration_seconds']}s")
                print(f"    - Appeared in {person['total_frames']} frames")
                print()
        
        print(f"✓ Output video saved to: {output_path}\n")
        
    except Exception as e:
        print(f"❌ Error during video processing: {e}")
        import traceback
        traceback.print_exc()
        return


def test_basic_tracker():
    """Test basic tracker functionality with single image"""
    print("=" * 80)
    print("TEST 3: Basic Tracker Test (Single Image)")
    print("=" * 80 + "\n")
    
    img_dir = Path(__file__).parent.parent / "image" / "IMAGE_001"
    
    if not img_dir.exists():
        print(f"❌ Image directory not found: {img_dir}")
        return
    
    # Get first image
    image_files = sorted(img_dir.glob("*.jpg"))
    if not image_files:
        print(f"❌ No images found in {img_dir}")
        return
    
    test_image = image_files[0]
    
    print(f"Testing with image: {test_image.name}")
    
    try:
        import cv2
        
        # Initialize tracker
        tracker = PersonTracker(confidence=0.5)
        
        # Load image
        frame = cv2.imread(str(test_image))
        if frame is None:
            print(f"❌ Cannot read image: {test_image}")
            return
        
        # Detect and track
        detections, annotated_frame = tracker.detect_and_track(frame)
        
        # Print results
        print(f"\n✓ Detection complete!")
        print(f"  - Persons detected: {len(detections)}")
        
        if hasattr(detections, 'tracker_id') and detections.tracker_id is not None:
            print(f"  - Tracking IDs: {[int(id) for id in detections.tracker_id]}")
        
        # Save annotated image
        output_path = Path(__file__).parent / "output" / "test_tracking_single.jpg"
        output_path.parent.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(str(output_path), annotated_frame)
        
        print(f"  - Annotated image saved: {output_path}\n")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nMake sure you have installed:")
        print("  pip install ultralytics supervision")
        return
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    print("\n🚀 PERSON TRACKER TESTING SUITE\n")
    
    # Run tests
    test_basic_tracker()
    test_tracking_on_images()
    test_tracking_on_video()
    
    print("=" * 80)
    print("✅ ALL TESTS COMPLETE!")
    print("=" * 80)
