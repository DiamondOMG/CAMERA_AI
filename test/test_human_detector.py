"""
Test Human Detection on Image Dataset

Tests the human detector on images from image/IMAGE_001 folder
and saves annotated results to test/output/
"""

import sys
from pathlib import Path
import cv2
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from human_detector import HumanDetector


def test_human_detection():
    """Test detection on sample images from dataset"""
    
    # Setup paths
    img_dir = Path(__file__).parent.parent / "image" / "IMAGE_001"
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    if not img_dir.exists():
        print(f"❌ Image directory not found: {img_dir}")
        return
    
    # Get sample images (first 10 or all if less)
    image_files = sorted(
        img_dir.glob("*.jpg"),
        key=lambda p: int(p.stem)
    )[:100]
    
    if not image_files:
        print(f"❌ No images found in {img_dir}")
        return
    
    print(f"✓ Found {len(image_files)} images")
    print(f"✓ Initializing YOLOv8 detector... (first run will download model ~35MB)")
    
    try:
        detector = HumanDetector(conf_threshold=0.3)
    except ImportError as e:
        print(f"❌ {e}")
        print("\nInstall ultralytics with:")
        print("  pip install ultralytics")
        return
    
    print("✓ Detector initialized\n")
    print("Running detection on sample images...\n")
    
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "dataset_path": str(img_dir),
        "num_images": len(image_files),
        "results": {}
    }
    
    total_humans = 0
    images_with_humans = 0
    
    for idx, img_path in enumerate(image_files, 1):
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [{idx}/{len(image_files)}] ⚠ Failed to read {img_path.name}")
            continue
        
        # Detect
        detections = detector.detect(img)
        count = detections['count']
        
        total_humans += count
        if count > 0:
            images_with_humans += 1
        
        status = f"✓ {count} human(s)" if count > 0 else "  (no humans)"
        print(f"  [{idx}/{len(image_files)}] {img_path.name:25} {status}")
        
        # Store result
        all_results['results'][img_path.name] = {
            'timestamp': int(img_path.stem),
            'detections': detections['detections'],
            'count': count
        }
        
        # Draw annotations and save
        if count > 0:
            annotated = detector.draw_detections(img, detections)
            output_path = output_dir / f"annotated_{img_path.name}"
            cv2.imwrite(str(output_path), annotated)
    
    # Print summary
    print("\n" + "="*60)
    print("📊 DETECTION SUMMARY")
    print("="*60)
    print(f"Total images processed:     {len(image_files)}")
    print(f"Images with humans:         {images_with_humans}")
    print(f"Total humans detected:      {total_humans}")
    print(f"Average per image:          {total_humans/len(image_files):.2f}")
    print("="*60)
    
    # Save detailed results as JSON
    json_path = output_dir / "detection_results.json"
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Results saved to:")
    print(f"  - {json_path}")
    print(f"  - Annotated images in {output_dir}/")
    
    return all_results


if __name__ == "__main__":
    test_human_detection()
