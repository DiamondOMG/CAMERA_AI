"""
Human Detection Module using YOLOv8

This module detects human objects in images and returns bounding boxes
along with confidence scores using YOLOv8 object detection.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional


class HumanDetector:
    """
    Detects humans in images using YOLOv8.
    
    Attributes:
        model: YOLOv8 model instance
        conf_threshold: Confidence threshold for detections (default 0.5)
    """
    
    def __init__(self, conf_threshold: float = 0.5):
        """
        Initialize the YOLOv8 detector.
        
        Args:
            conf_threshold: Confidence threshold for detections
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics not installed. Install with: pip install ultralytics"
            )
        
        # Load YOLOv8 nano model (fastest, ~6MB)
        # Options: yolov8n.pt (nano), yolov8s.pt (small), yolov8m.pt (medium)
        self.model = YOLO('yolov8n.pt')
        self.conf_threshold = conf_threshold
    
    def detect(self, image: np.ndarray) -> Dict[str, any]:
        """
        Detect humans in a single image using YOLOv8.
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            Dictionary containing:
            - 'detections': List of dicts with keys:
                - 'bbox': [x1, y1, x2, y2] in pixel coords
                - 'confidence': float [0, 1]
                - 'class_id': int (0 for person)
                - 'class_name': str ('person')
            - 'count': int (number of humans detected)
            - 'image_shape': tuple (height, width, channels)
        """
        # Run inference
        results = self.model(image, conf=self.conf_threshold, verbose=False)
        
        detections = []
        
        # Extract person detections (class_id = 0 in COCO dataset)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                
                # Only keep person detections (class_id = 0)
                if class_id == 0:
                    # Get bbox coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    
                    detections.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': 'person'
                    })
        
        return {
            'detections': detections,
            'count': len(detections),
            'image_shape': image.shape
        }
    
    def detect_batch(self, image_paths: List[Path]) -> Dict[Path, Dict]:
        """
        Detect humans in multiple images.
        
        Args:
            image_paths: List of paths to images
            
        Returns:
            Dictionary mapping image path to detection results
        """
        results = {}
        for img_path in image_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                results[img_path] = {'error': f'Failed to read {img_path}'}
                continue
            results[img_path] = self.detect(img)
        return results
    
    def draw_detections(
        self, 
        image: np.ndarray, 
        detections: Dict
    ) -> np.ndarray:
        """
        Draw bounding boxes on image.
        
        Args:
            image: Input image (BGR)
            detections: Detection results from detect()
            
        Returns:
            Image with drawn bounding boxes
        """
        img_copy = image.copy()
        
        for det in detections['detections']:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            
            # Draw rectangle
            cv2.rectangle(
                img_copy, 
                (int(x1), int(y1)), 
                (int(x2), int(y2)), 
                (0, 255, 0), 
                2
            )
            
            # Draw label with confidence
            label = f"{det['class_name']} {conf:.2f}"
            cv2.putText(
                img_copy,
                label,
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
        
        return img_copy


if __name__ == "__main__":
    # Simple test
    print("HumanDetector module loaded successfully")
