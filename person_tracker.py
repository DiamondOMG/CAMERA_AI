"""
Person Tracker using YOLOv8 + ByteTrack + Face Recognition

This module provides real-time person tracking across video frames.
Each detected person gets a unique ID that persists across frames.
Optional face recognition can be enabled to maintain identity across re-entries.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime

try:
    from face_recognizer import FaceRecognizer, FaceData
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    FaceRecognizer = None
    FaceData = None


@dataclass
class PersonInfo:
    """Information about a tracked person"""
    track_id: int
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    total_frames: int = 0
    bbox_history: List[Tuple[int, int, int, int]] = field(default_factory=list)
    face_id: Optional[int] = None  # Face recognition ID
    unique_id: Optional[int] = None  # Unique person ID (merged from face+track)


class PersonTracker:
    """Track multiple persons across video frames using YOLOv8 + ByteTrack + Face Recognition"""
    
    def __init__(
        self, 
        model_path: str = "yolov8n.pt", 
        confidence: float = 0.5,
        use_face_recognition: bool = False,
        face_tolerance: float = 0.6
    ):
        """
        Initialize the person tracker
        
        Args:
            model_path: Path to YOLOv8 model file
            confidence: Minimum confidence threshold for detection (0-1)
            use_face_recognition: Enable face recognition for person re-identification
            face_tolerance: Face matching tolerance (0-1, lower is stricter)
        """
        print(f"🔧 Initializing PersonTracker...")
        
        # Initialize YOLOv8 model
        self.model = YOLO(model_path)
        self.confidence = confidence
        
        # Initialize ByteTrack tracker from supervision
        self.tracker = sv.ByteTrack()
        
        # Initialize annotators for visualization
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        
        # Track persons info
        self.persons: Dict[int, PersonInfo] = {}
        
        # Face recognition
        self.use_face_recognition = use_face_recognition and FACE_RECOGNITION_AVAILABLE
        self.face_recognizer = None
        
        if self.use_face_recognition:
            if not FACE_RECOGNITION_AVAILABLE:
                print("⚠️  Face recognition requested but not available")
                print("    Install: pip install face-recognition")
                self.use_face_recognition = False
            else:
                self.face_recognizer = FaceRecognizer(tolerance=face_tolerance, model="hog")
                print("✓ Face recognition enabled")
        
        # Track ID mappings for merging
        self.track_id_to_unique_id: Dict[int, int] = {}
        self.next_unique_id = 1
        
        print(f"✓ PersonTracker initialized")
        print(f"  - Model: {model_path}")
        print(f"  - Confidence threshold: {confidence}")
        print(f"  - Tracker: ByteTrack")
        print(f"  - Face recognition: {'Enabled' if self.use_face_recognition else 'Disabled'}\n")
    
    def detect_and_track(self, frame: np.ndarray) -> Tuple[sv.Detections, np.ndarray]:
        """
        Detect and track persons in a single frame
        
        Args:
            frame: Input image/frame (BGR format)
            
        Returns:
            Tuple of (detections with tracking IDs, annotated frame)
        """
        # Run YOLOv8 detection
        results = self.model(frame, conf=self.confidence, classes=[0], verbose=False)[0]
        
        # Convert to supervision Detections format
        detections = sv.Detections.from_ultralytics(results)
        
        # Update tracker with new detections
        detections = self.tracker.update_with_detections(detections)
        
        # Face recognition on detected persons (if enabled)
        if self.use_face_recognition and detections.tracker_id is not None:
            self._process_faces(frame, detections)
        
        # Update person info
        self._update_person_info(detections)
        
        # Annotate frame
        annotated_frame = self._annotate_frame(frame.copy(), detections)
        
        return detections, annotated_frame
    
    def _process_faces(self, frame: np.ndarray, detections: sv.Detections):
        """Process face recognition for detected persons"""
        for idx, track_id in enumerate(detections.tracker_id):
            track_id = int(track_id)
            bbox = detections.xyxy[idx]
            
            # Extract person crop
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            person_crop = frame[y1:y2, x1:x2]
            
            if person_crop.size == 0:
                continue
            
            # Detect faces in person crop
            faces = self.face_recognizer.detect_faces(person_crop)
            
            if faces:
                # Use first detected face
                face = faces[0]
                face_id = face.face_id
                
                # Get or create unique ID
                if track_id not in self.track_id_to_unique_id:
                    # Check if this face has been seen before with different track_id
                    existing_track_id = self._find_track_id_by_face(face_id)
                    
                    if existing_track_id is not None:
                        # Merge: use existing unique_id
                        unique_id = self.track_id_to_unique_id[existing_track_id]
                        self.track_id_to_unique_id[track_id] = unique_id
                    else:
                        # New unique person
                        unique_id = self.next_unique_id
                        self.next_unique_id += 1
                        self.track_id_to_unique_id[track_id] = unique_id
                
                # Update person info with face_id
                if track_id in self.persons:
                    self.persons[track_id].face_id = face_id
                    self.persons[track_id].unique_id = self.track_id_to_unique_id[track_id]
    
    def _find_track_id_by_face(self, face_id: int) -> Optional[int]:
        """Find existing track_id that has the same face_id"""
        for tid, person in self.persons.items():
            if person.face_id == face_id:
                return tid
        return None
    
    def _update_person_info(self, detections: sv.Detections):
        """Update tracking information for each detected person"""
        current_time = datetime.now()
        
        if detections.tracker_id is None:
            return
        
        for idx, track_id in enumerate(detections.tracker_id):
            track_id = int(track_id)
            bbox = detections.xyxy[idx]
            
            if track_id not in self.persons:
                # New person detected
                self.persons[track_id] = PersonInfo(
                    track_id=track_id,
                    first_seen=current_time,
                    last_seen=current_time,
                    total_frames=1,
                    bbox_history=[tuple(bbox)]
                )
            else:
                # Update existing person
                person = self.persons[track_id]
                person.last_seen = current_time
                person.total_frames += 1
                person.bbox_history.append(tuple(bbox))
                
                # Keep only last 30 bboxes to save memory
                if len(person.bbox_history) > 30:
                    person.bbox_history = person.bbox_history[-30:]
    
    def _annotate_frame(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        """Draw bounding boxes and labels on frame"""
        if detections.tracker_id is None or len(detections) == 0:
            return frame
        
        # Prepare labels with person ID and confidence
        labels = []
        for idx, track_id in enumerate(detections.tracker_id):
            confidence = detections.confidence[idx] if detections.confidence is not None else 0
            person_info = self.persons.get(int(track_id))
            
            if person_info:
                duration = (person_info.last_seen - person_info.first_seen).total_seconds()
                
                # Use unique_id if face recognition is enabled, otherwise use track_id
                if self.use_face_recognition and person_info.unique_id is not None:
                    label = f"ID#{person_info.unique_id} (T{track_id}) | {confidence:.2f}"
                else:
                    label = f"Person #{track_id} | {confidence:.2f} | {duration:.1f}s"
            else:
                label = f"Person #{track_id} | {confidence:.2f}"
            
            labels.append(label)
        
        # Draw boxes and labels
        frame = self.box_annotator.annotate(scene=frame, detections=detections)
        frame = self.label_annotator.annotate(scene=frame, detections=detections, labels=labels)
        
        return frame
    
    def get_tracking_summary(self) -> Dict:
        """Get summary of all tracked persons"""
        
        # Count unique persons (by unique_id if face recognition enabled)
        if self.use_face_recognition:
            unique_ids = set()
            for person in self.persons.values():
                if person.unique_id is not None:
                    unique_ids.add(person.unique_id)
            total_unique = len(unique_ids) if unique_ids else len(self.persons)
        else:
            total_unique = len(self.persons)
        
        summary = {
            "total_unique_persons": total_unique,
            "total_tracked_ids": len(self.persons),
            "face_recognition_enabled": self.use_face_recognition,
            "persons": []
        }
        
        for track_id, person in sorted(self.persons.items()):
            duration = (person.last_seen - person.first_seen).total_seconds()
            person_data = {
                "track_id": track_id,
                "first_seen": person.first_seen.isoformat(),
                "last_seen": person.last_seen.isoformat(),
                "duration_seconds": round(duration, 2),
                "total_frames": person.total_frames
            }
            
            if self.use_face_recognition:
                person_data["face_id"] = person.face_id
                person_data["unique_id"] = person.unique_id
            
            summary["persons"].append(person_data)
        
        return summary
    
    def reset(self):
        """Reset tracker state"""
        self.tracker = sv.ByteTrack()
        self.persons.clear()
        self.track_id_to_unique_id.clear()
        self.next_unique_id = 1
        
        if self.face_recognizer:
            # Don't reset face database to maintain recognition across sessions
            pass
        
        print("✓ Tracker reset")
    
    def save_face_database(self, filepath: str):
        """Save face recognition database"""
        if self.face_recognizer:
            self.face_recognizer.save_database(filepath)
    
    def load_face_database(self, filepath: str):
        """Load face recognition database"""
        if self.face_recognizer:
            return self.face_recognizer.load_database(filepath)
        return False


def process_video(
    video_path: str,
    output_path: Optional[str] = None,
    model_path: str = "yolov8n.pt",
    confidence: float = 0.5,
    display: bool = True,
    use_face_recognition: bool = False
) -> Dict:
    """
    Process a video file and track all persons
    
    Args:
        video_path: Path to input video file
        output_path: Optional path to save annotated video
        model_path: Path to YOLOv8 model
        confidence: Detection confidence threshold
        display: Whether to display video while processing
        use_face_recognition: Enable face recognition for re-identification
        
    Returns:
        Dictionary with tracking summary
    """
    # Initialize tracker
    tracker = PersonTracker(
        model_path=model_path, 
        confidence=confidence,
        use_face_recognition=use_face_recognition
    )
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Processing video: {video_path}")
    print(f"  - Resolution: {width}x{height}")
    print(f"  - FPS: {fps}")
    print(f"  - Total frames: {total_frames}\n")
    
    # Setup video writer if output path provided
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect and track
            detections, annotated_frame = tracker.detect_and_track(frame)
            
            # Write to output video
            if writer:
                writer.write(annotated_frame)
            
            # Display frame
            if display:
                # Add frame counter
                cv2.putText(
                    annotated_frame,
                    f"Frame: {frame_count}/{total_frames} | Persons: {len(tracker.persons)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                
                cv2.imshow("Person Tracking", annotated_frame)
                
                # Press 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n⚠️  Stopped by user")
                    break
            
            # Print progress
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% | Unique persons: {len(tracker.persons)}")
    
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
    
    print(f"\n✓ Video processing complete!")
    print(f"  - Frames processed: {frame_count}")
    print(f"  - Unique persons detected: {len(tracker.persons)}")
    
    if output_path:
        print(f"  - Output saved: {output_path}")
    
    return tracker.get_tracking_summary()


def process_images(
    image_paths: List[str],
    output_dir: Optional[str] = None,
    model_path: str = "yolov8n.pt",
    confidence: float = 0.5,
    use_face_recognition: bool = False
) -> Dict:
    """
    Process multiple images as if they were video frames
    
    Args:
        image_paths: List of image file paths (should be in sequence order)
        output_dir: Optional directory to save annotated images
        model_path: Path to YOLOv8 model
        confidence: Detection confidence threshold
        use_face_recognition: Enable face recognition for re-identification
        
    Returns:
        Dictionary with tracking summary
    """
    tracker = PersonTracker(
        model_path=model_path, 
        confidence=confidence,
        use_face_recognition=use_face_recognition
    )
    
    print(f"📸 Processing {len(image_paths)} images...")
    
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    for idx, img_path in enumerate(image_paths, 1):
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"⚠️  Cannot read image: {img_path}")
            continue
        
        # Detect and track
        detections, annotated_frame = tracker.detect_and_track(frame)
        
        # Save annotated image
        if output_dir:
            import os
            filename = os.path.basename(img_path)
            output_path = os.path.join(output_dir, f"tracked_{filename}")
            cv2.imwrite(output_path, annotated_frame)
        
        if idx % 10 == 0:
            print(f"Processed {idx}/{len(image_paths)} images | Unique persons: {len(tracker.persons)}")
    
    print(f"\n✓ Image processing complete!")
    print(f"  - Images processed: {len(image_paths)}")
    print(f"  - Unique persons detected: {len(tracker.persons)}")
    
    if output_dir:
        print(f"  - Output saved to: {output_dir}")
    
    return tracker.get_tracking_summary()
