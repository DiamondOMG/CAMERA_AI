"""
Test Face Recognition + Person Tracking

Tests face recognition capabilities:
1. Basic face detection and recognition
2. Face recognition with person tracking
3. Re-identification across frames (person leaves and returns)
"""

import sys
import json
import cv2
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from face_recognizer import FaceRecognizer, compare_faces
from person_tracker import PersonTracker, process_images


def test_basic_face_recognition():
    """Test basic face recognition on single images"""
    print("=" * 80)
    print("TEST 1: Basic Face Recognition")
    print("=" * 80 + "\n")
    
    img_dir = Path(__file__).parent.parent / "image" / "IMAGE_001"
    output_dir = Path(__file__).parent / "output" / "face_recognition"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if not img_dir.exists():
        print(f"❌ Image directory not found: {img_dir}")
        return
    
    # Get first few images
    image_files = sorted(img_dir.glob("*.jpg"), key=lambda p: int(p.stem))[:10]
    
    if not image_files:
        print(f"❌ No images found in {img_dir}")
        return
    
    print(f"✓ Found {len(image_files)} images\n")
    
    try:
        recognizer = FaceRecognizer(tolerance=0.6, model="hog")
        
        for idx, img_path in enumerate(image_files, 1):
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            
            # Detect faces
            faces = recognizer.detect_faces(frame)
            
            if faces:
                print(f"Image {idx}: Found {len(faces)} face(s)")
                for face in faces:
                    print(f"  - Face ID: {face.face_id}, Confidence: {face.confidence:.2f}")
                
                # Draw and save
                annotated = recognizer.draw_faces(frame, faces)
                output_path = output_dir / f"face_{img_path.name}"
                cv2.imwrite(str(output_path), annotated)
            else:
                print(f"Image {idx}: No faces detected")
        
        # Print statistics
        stats = recognizer.get_statistics()
        print(f"\n📊 Face Recognition Statistics:")
        print(f"  - Total unique faces: {stats['total_faces']}")
        print(f"  - Named faces: {stats['named_faces']}")
        print(f"  - Unnamed faces: {stats['unnamed_faces']}")
        
        # Save face database
        db_path = Path(__file__).parent / "output" / "face_database.pkl"
        recognizer.save_database(str(db_path))
        print(f"\n✓ Annotated images saved to: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_face_comparison():
    """Test comparing faces between two images"""
    print("\n" + "=" * 80)
    print("TEST 2: Face Comparison")
    print("=" * 80 + "\n")
    
    img_dir = Path(__file__).parent.parent / "image" / "IMAGE_001"
    
    if not img_dir.exists():
        print(f"❌ Image directory not found: {img_dir}")
        return
    
    image_files = sorted(img_dir.glob("*.jpg"), key=lambda p: int(p.stem))[:20]
    
    if len(image_files) < 2:
        print(f"❌ Need at least 2 images")
        return
    
    # Compare first image with several others
    base_image = str(image_files[0])
    
    print(f"Base image: {image_files[0].name}\n")
    
    for img_file in image_files[1:6]:
        result = compare_faces(base_image, str(img_file), tolerance=0.6)
        
        if 'error' in result:
            print(f"❌ {img_file.name}: {result['error']}")
        elif not result['match']:
            print(f"✗ NO MATCH - {img_file.name}")
            print(f"  Reason: {result.get('reason', 'Unknown')}")
        else:
            match_status = "✓ MATCH" if result['match'] else "✗ NO MATCH"
            print(f"{match_status} - {img_file.name}")
            print(f"  Distance: {result['distance']:.4f}, Confidence: {result['confidence']:.2f}")


def test_tracking_with_face_recognition():
    """Test person tracking WITH face recognition enabled"""
    print("\n" + "=" * 80)
    print("TEST 3: Person Tracking + Face Recognition")
    print("=" * 80 + "\n")
    
    img_dir = Path(__file__).parent.parent / "image" / "IMAGE_001"
    output_dir = Path(__file__).parent / "output" / "tracked_with_faces"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if not img_dir.exists():
        print(f"❌ Image directory not found: {img_dir}")
        return
    
    # Get images
    image_files = sorted(
        img_dir.glob("*.jpg"),
        key=lambda p: int(p.stem)
    )[:100]
    
    if not image_files:
        print(f"❌ No images found in {img_dir}")
        return
    
    print(f"✓ Processing {len(image_files)} images with face recognition...\n")
    
    try:
        summary = process_images(
            image_paths=[str(p) for p in image_files],
            output_dir=str(output_dir),
            confidence=0.5,
            use_face_recognition=True  # Enable face recognition!
        )
        
        # Print summary
        print("\n" + "=" * 80)
        print("TRACKING SUMMARY (With Face Recognition)")
        print("=" * 80)
        print(f"\n📊 Statistics:")
        print(f"  - Total tracking IDs assigned: {summary['total_tracked_ids']}")
        print(f"  - Total UNIQUE persons (face-based): {summary['total_unique_persons']}")
        print(f"  - Face recognition: {'Enabled' if summary['face_recognition_enabled'] else 'Disabled'}")
        
        if summary['persons']:
            print(f"\n👥 Person Details:")
            print("-" * 80)
            
            # Group by unique_id
            unique_persons = {}
            for person in summary['persons']:
                uid = person.get('unique_id')
                if uid not in unique_persons:
                    unique_persons[uid] = []
                unique_persons[uid].append(person)
            
            for unique_id, tracks in sorted(unique_persons.items()):
                if unique_id is None:
                    continue
                    
                total_frames = sum(p['total_frames'] for p in tracks)
                track_ids = [p['track_id'] for p in tracks]
                face_ids = set(p.get('face_id') for p in tracks if p.get('face_id'))
                
                print(f"\n  Unique Person #{unique_id}:")
                print(f"    - Tracking IDs: {track_ids} (merged)")
                print(f"    - Face IDs: {list(face_ids)}")
                print(f"    - Total frames: {total_frames}")
                print(f"    - First seen: {tracks[0]['first_seen']}")
                print(f"    - Last seen: {tracks[-1]['last_seen']}")
        
        # Save summary
        summary_file = Path(__file__).parent / "output" / "tracking_with_faces_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Summary saved to: {summary_file}")
        print(f"✓ Annotated images saved to: {output_dir}\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_comparison_no_face_vs_with_face():
    """Compare tracking WITHOUT vs WITH face recognition"""
    print("\n" + "=" * 80)
    print("TEST 4: Comparison - Tracking Only vs Tracking + Face Recognition")
    print("=" * 80 + "\n")
    
    img_dir = Path(__file__).parent.parent / "image" / "IMAGE_001"
    
    if not img_dir.exists():
        print(f"❌ Image directory not found: {img_dir}")
        return
    
    image_files = sorted(
        img_dir.glob("*.jpg"),
        key=lambda p: int(p.stem)
    )[:50]
    
    if not image_files:
        print(f"❌ No images found in {img_dir}")
        return
    
    print(f"Processing {len(image_files)} images...\n")
    
    try:
        # Test WITHOUT face recognition
        print("🔵 Test 1: Tracking ONLY (no face recognition)")
        summary_no_face = process_images(
            image_paths=[str(p) for p in image_files],
            output_dir=None,
            confidence=0.5,
            use_face_recognition=False
        )
        
        # Test WITH face recognition
        print("\n🟢 Test 2: Tracking + Face Recognition")
        summary_with_face = process_images(
            image_paths=[str(p) for p in image_files],
            output_dir=None,
            confidence=0.5,
            use_face_recognition=True
        )
        
        # Compare results
        print("\n" + "=" * 80)
        print("COMPARISON RESULTS")
        print("=" * 80)
        print(f"\n📊 Without Face Recognition:")
        print(f"  - Unique persons detected: {summary_no_face['total_unique_persons']}")
        
        print(f"\n📊 With Face Recognition:")
        print(f"  - Tracking IDs assigned: {summary_with_face['total_tracked_ids']}")
        print(f"  - Unique persons (face-based): {summary_with_face['total_unique_persons']}")
        
        improvement = summary_no_face['total_unique_persons'] - summary_with_face['total_unique_persons']
        if improvement > 0:
            print(f"\n✅ Face recognition reduced false positives by {improvement} persons")
            print(f"   (merged multiple tracking IDs into same person)")
        elif improvement < 0:
            print(f"\n⚠️  Face recognition found {abs(improvement)} more unique persons")
        else:
            print(f"\n➖ No difference detected")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n🚀 FACE RECOGNITION + TRACKING TESTING SUITE\n")
    
    # Run tests
    test_basic_face_recognition()
    test_face_comparison()
    test_tracking_with_face_recognition()
    test_comparison_no_face_vs_with_face()
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS COMPLETE!")
    print("=" * 80)
    print("\nℹ️  Key Insights:")
    print("  - Face recognition helps identify same person across different tracking IDs")
    print("  - Useful when person leaves frame and returns (gets new tracking ID)")
    print("  - Trade-off: slower processing but better accuracy")
