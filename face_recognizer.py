"""
Face Recognition Module

This module provides face detection, encoding, and recognition capabilities
using the face_recognition library (based on dlib).
"""

import face_recognition
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
import pickle
from pathlib import Path


@dataclass
class FaceData:
    """Data structure for storing face information"""
    face_id: int
    encoding: np.ndarray
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    name: Optional[str] = None
    confidence: float = 0.0


class FaceRecognizer:
    """Face detection and recognition using face_recognition library"""
    
    def __init__(self, tolerance: float = 0.6, model: str = "hog"):
        """
        Initialize face recognizer
        
        Args:
            tolerance: How much distance between faces to consider it a match (0-1)
                      Lower is more strict. 0.6 is typical best performance.
            model: Face detection model - 'hog' (faster, CPU) or 'cnn' (more accurate, GPU)
        """
        self.tolerance = tolerance
        self.model = model
        
        # Database of known faces
        self.known_face_encodings: List[np.ndarray] = []
        self.known_face_ids: List[int] = []
        self.known_face_names: Dict[int, str] = {}
        
        # Counter for assigning new face IDs
        self.next_face_id = 1
        
        print(f"🔧 FaceRecognizer initialized")
        print(f"  - Model: {model}")
        print(f"  - Tolerance: {tolerance}")
        print(f"  - Known faces: {len(self.known_face_encodings)}\n")
    
    def detect_faces(self, frame: np.ndarray) -> List[FaceData]:
        """
        Detect and recognize faces in an image
        
        Args:
            frame: Input image in BGR format (OpenCV)
            
        Returns:
            List of FaceData objects containing face information
        """
        # Convert BGR to RGB (face_recognition uses RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect face locations
        face_locations = face_recognition.face_locations(rgb_frame, model=self.model)
        
        if not face_locations:
            return []
        
        # Get face encodings
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        faces = []
        
        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            # Convert to (x1, y1, x2, y2) format
            bbox = (left, top, right, bottom)
            
            # Try to match with known faces
            face_id, confidence = self._match_face(encoding)
            
            # If no match found, assign new ID
            if face_id is None:
                face_id = self._add_new_face(encoding)
            
            name = self.known_face_names.get(face_id)
            
            faces.append(FaceData(
                face_id=face_id,
                encoding=encoding,
                bbox=bbox,
                name=name,
                confidence=confidence
            ))
        
        return faces
    
    def _match_face(self, encoding: np.ndarray) -> Tuple[Optional[int], float]:
        """
        Match a face encoding against known faces
        
        Returns:
            Tuple of (face_id, confidence) or (None, 0.0) if no match
        """
        if not self.known_face_encodings:
            return None, 0.0
        
        # Calculate distances to all known faces
        distances = face_recognition.face_distance(self.known_face_encodings, encoding)
        
        # Find the best match
        best_match_idx = np.argmin(distances)
        best_distance = distances[best_match_idx]
        
        # Check if within tolerance
        if best_distance <= self.tolerance:
            face_id = self.known_face_ids[best_match_idx]
            confidence = 1.0 - best_distance  # Convert distance to confidence
            return face_id, confidence
        
        return None, 0.0
    
    def _add_new_face(self, encoding: np.ndarray) -> int:
        """Add a new face to the known faces database"""
        face_id = self.next_face_id
        self.next_face_id += 1
        
        self.known_face_encodings.append(encoding)
        self.known_face_ids.append(face_id)
        
        return face_id
    
    def set_face_name(self, face_id: int, name: str):
        """Assign a name to a face ID"""
        self.known_face_names[face_id] = name
    
    def merge_faces(self, face_id_1: int, face_id_2: int):
        """
        Merge two face IDs (when they're determined to be the same person)
        Keep face_id_1, remove face_id_2
        """
        # Find indices
        indices_to_remove = [i for i, fid in enumerate(self.known_face_ids) if fid == face_id_2]
        
        # Remove face_id_2 entries
        for idx in sorted(indices_to_remove, reverse=True):
            del self.known_face_encodings[idx]
            del self.known_face_ids[idx]
        
        # Transfer name if exists
        if face_id_2 in self.known_face_names:
            if face_id_1 not in self.known_face_names:
                self.known_face_names[face_id_1] = self.known_face_names[face_id_2]
            del self.known_face_names[face_id_2]
    
    def save_database(self, filepath: str):
        """Save known faces database to file"""
        data = {
            'encodings': self.known_face_encodings,
            'ids': self.known_face_ids,
            'names': self.known_face_names,
            'next_id': self.next_face_id,
            'tolerance': self.tolerance
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✓ Face database saved to {filepath}")
    
    def load_database(self, filepath: str):
        """Load known faces database from file"""
        if not Path(filepath).exists():
            print(f"⚠️  Database file not found: {filepath}")
            return False
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.known_face_encodings = data['encodings']
        self.known_face_ids = data['ids']
        self.known_face_names = data.get('names', {})
        self.next_face_id = data['next_id']
        self.tolerance = data.get('tolerance', self.tolerance)
        
        print(f"✓ Face database loaded from {filepath}")
        print(f"  - Known faces: {len(self.known_face_encodings)}")
        
        return True
    
    def get_statistics(self) -> Dict:
        """Get statistics about known faces"""
        return {
            'total_faces': len(self.known_face_encodings),
            'named_faces': len(self.known_face_names),
            'unnamed_faces': len(self.known_face_encodings) - len(self.known_face_names),
            'next_face_id': self.next_face_id
        }
    
    def draw_faces(self, frame: np.ndarray, faces: List[FaceData]) -> np.ndarray:
        """Draw face bounding boxes and labels on frame"""
        annotated = frame.copy()
        
        for face in faces:
            x1, y1, x2, y2 = face.bbox
            
            # Choose color based on whether face is named
            color = (0, 255, 0) if face.name else (255, 0, 0)
            
            # Draw rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            if face.name:
                label = f"{face.name} (#{face.face_id})"
            else:
                label = f"Person #{face.face_id}"
            
            if face.confidence > 0:
                label += f" {face.confidence:.2f}"
            
            # Draw label background
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            
            # Draw label text
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )
        
        return annotated


def compare_faces(image1_path: str, image2_path: str, tolerance: float = 0.6) -> Dict:
    """
    Compare faces in two images
    
    Returns:
        Dictionary with comparison results
    """
    import cv2
    
    recognizer = FaceRecognizer(tolerance=tolerance)
    
    # Load images
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    
    if img1 is None or img2 is None:
        return {'error': 'Could not load one or both images'}
    
    # Detect faces
    faces1 = recognizer.detect_faces(img1)
    faces2 = recognizer.detect_faces(img2)
    
    if not faces1 or not faces2:
        return {
            'match': False,
            'reason': 'No faces detected in one or both images',
            'faces_in_image1': len(faces1),
            'faces_in_image2': len(faces2)
        }
    
    # Compare first face in each image
    encoding1 = faces1[0].encoding
    encoding2 = faces2[0].encoding
    
    distance = face_recognition.face_distance([encoding1], encoding2)[0]
    match = distance <= tolerance
    
    return {
        'match': match,
        'distance': float(distance),
        'tolerance': tolerance,
        'confidence': float(1.0 - distance) if match else 0.0,
        'faces_in_image1': len(faces1),
        'faces_in_image2': len(faces2)
    }
