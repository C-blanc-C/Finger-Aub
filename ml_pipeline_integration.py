import cv2
import torch
import numpy as np
from ultralytics import YOLO
from collections import deque
import time

class RoboticHandMLPipeline:
    def __init__(self, camera_index=0):
        # Initialize models
        self.yolo_model = YOLO('yolov8m.pt')  # Medium model for balance
        self.depth_model = self._load_depth_model()
        self.shape_detector = ShapeDetector()
        
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Performance tracking
        self.fps_buffer = deque(maxlen=30)
        
    def _load_depth_model(self):
        # Load MiDaS model
        model_type = "DPT_Hybrid"
        midas = torch.hub.load("intel-isl/MiDaS", model_type)
        midas.eval()
        
        # Load transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.dpt_transform
        
        return midas
    
    def process_frame(self, frame):
        """Main processing pipeline for a single frame"""
        start_time = time.time()
        
        # 1. Object Detection
        detections = self._detect_objects(frame)
        
        # 2. Depth Estimation
        depth_map = self._estimate_depth(frame)
        
        # 3. Shape Analysis for each detection
        shape_features = []
        for det in detections:
            shape_info = self._analyze_shape(frame, det)
            shape_features.append(shape_info)
        
        # 4. Feature Fusion
        control_commands = self._fuse_features(detections, depth_map, shape_features)
        
        # Performance tracking
        inference_time = (time.time() - start_time) * 1000
        self.fps_buffer.append(inference_time)
        
        return control_commands, inference_time
    
    def _detect_objects(self, frame):
        """YOLOv8 object detection"""
        results = self.yolo_model(frame, conf=0.5)
        detections = []
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'class': self.yolo_model.names[cls],
                        'confidence': conf
                    })
        
        return detections
    
    def _estimate_depth(self, frame):
        """MiDaS depth estimation"""
        # Prepare input
        input_batch = self.transform(frame).unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            prediction = self.depth_model(input_batch)
        
        # Resize to original dimensions
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        
        # Normalize to real-world distances (assuming calibrated camera)
        # This requires camera calibration - placeholder values
        depth_map = self._normalize_depth(depth_map)
        
        return depth_map
    
    def _normalize_depth(self, depth_map, min_distance=0.1, max_distance=2.0):
        """Convert relative depth to meters"""
        # Normalize to 0-1 range
        depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        
        # Map to real-world distances
        depth_meters = depth_normalized * (max_distance - min_distance) + min_distance
        
        return depth_meters
    
    def _analyze_shape(self, frame, detection):
        """Analyze shape characteristics of detected object"""
        x1, y1, x2, y2 = map(int, detection['bbox'])
        roi = frame[y1:y2, x1:x2]
        
        return self.shape_detector.analyze(roi)
    
    def _fuse_features(self, detections, depth_map, shape_features):
        """Combine all features to generate actuator commands"""
        commands = {
            'thumb': 0,
            'index': 0,
            'middle': 0,
            'ring': 0,
            'pinky': 0
        }
        
        if not detections:
            return commands
        
        # Get primary object (closest/largest)
        primary_obj = self._get_primary_object(detections, depth_map)
        primary_shape = shape_features[0] if shape_features else None
        
        # Generate commands based on object type and shape
        commands = self._generate_grip_pattern(
            primary_obj, 
            primary_shape,
            self._get_object_distance(primary_obj, depth_map)
        )
        
        return commands
    
    def _get_primary_object(self, detections, depth_map):
        """Select primary object based on size and distance"""
        if not detections:
            return None
            
        # Calculate priority score for each detection
        scores = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            
            # Get average depth in bbox region
            depth_roi = depth_map[int(y1):int(y2), int(x1):int(x2)]
            avg_depth = np.mean(depth_roi)
            
            # Calculate area
            area = (x2 - x1) * (y2 - y1)
            
            # Priority score: larger and closer objects have higher priority
            score = area / (avg_depth + 0.1)  # Add small value to avoid division by zero
            scores.append(score)
        
        # Return detection with highest score
        best_idx = np.argmax(scores)
        return detections[best_idx]
    
    def _get_object_distance(self, detection, depth_map):
        """Get average distance of object"""
        if detection is None:
            return float('inf')
            
        x1, y1, x2, y2 = map(int, detection['bbox'])
        depth_roi = depth_map[y1:y2, x1:x2]
        return np.mean(depth_roi)
    
    def _generate_grip_pattern(self, obj, shape, distance):
        """Generate grip pattern based on object properties"""
        # Default open position
        commands = {
            'thumb': 0,
            'index': 0,
            'middle': 0,
            'ring': 0,
            'pinky': 0
        }
        
        if obj is None:
            return commands
        
        # Distance-based scaling (closer = more closed)
        distance_factor = 1.0 - min(distance / 1.0, 1.0)  # Normalize to 0-1
        
        # Object-specific grip patterns
        grip_patterns = {
            'bottle': {'thumb': 0.7, 'index': 0.8, 'middle': 0.8, 'ring': 0.7, 'pinky': 0.6},
            'cup': {'thumb': 0.6, 'index': 0.7, 'middle': 0.7, 'ring': 0.6, 'pinky': 0.5},
            'pen': {'thumb': 0.9, 'index': 0.9, 'middle': 0.3, 'ring': 0.1, 'pinky': 0.1},
            'ball': {'thumb': 0.5, 'index': 0.6, 'middle': 0.6, 'ring': 0.6, 'pinky': 0.5},
            'book': {'thumb': 0.4, 'index': 0.5, 'middle': 0.5, 'ring': 0.5, 'pinky': 0.4},
            'phone': {'thumb': 0.3, 'index': 0.4, 'middle': 0.4, 'ring': 0.4, 'pinky': 0.3}
        }
        
        # Get base pattern
        base_pattern = grip_patterns.get(obj['class'], 
                                       {'thumb': 0.5, 'index': 0.5, 'middle': 0.5, 'ring': 0.5, 'pinky': 0.5})
        
        # Apply distance scaling
        for finger, base_value in base_pattern.items():
            commands[finger] = base_value * distance_factor * 100  # Convert to percentage
        
        # Apply shape modifiers
        if shape and shape.get('circularity', 0) > 0.8:
            # More uniform grip for circular objects
            avg_closure = np.mean(list(commands.values()))
            for finger in commands:
                commands[finger] = 0.8 * commands[finger] + 0.2 * avg_closure
        
        return commands


class ShapeDetector:
    """Advanced shape detection using contours and geometric features"""
    
    def analyze(self, roi):
        """Analyze shape characteristics of ROI"""
        if roi.size == 0:
            return {}
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {}
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate shape features
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        # Circularity
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Aspect ratio
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = w / h if h > 0 else 1
        
        # Convexity
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        convexity = area / hull_area if hull_area > 0 else 0
        
        # Approximate polygon
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        num_vertices = len(approx)
        
        return {
            'circularity': circularity,
            'aspect_ratio': aspect_ratio,
            'convexity': convexity,
            'num_vertices': num_vertices,
            'area': area,
            'perimeter': perimeter
        }


class ActuatorController:
    """Smooth actuator control with filtering"""
    
    def __init__(self, serial_port='/dev/ttyUSB0', baud_rate=115200):
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        
        # Previous commands for smoothing
        self.prev_commands = {
            'thumb': 0,
            'index': 0,
            'middle': 0,
            'ring': 0,
            'pinky': 0
        }
        
        # Smoothing factor (0-1, higher = more smoothing)
        self.alpha = 0.7
        
    def send_commands(self, commands):
        """Send smoothed commands to actuators"""
        smoothed_commands = self._smooth_commands(commands)
        
        # Format command string for Arduino
        # Format: T:XX,I:XX,M:XX,R:XX,P:XX\n
        cmd_str = f"T:{smoothed_commands['thumb']:.0f},"
        cmd_str += f"I:{smoothed_commands['index']:.0f},"
        cmd_str += f"M:{smoothed_commands['middle']:.0f},"
        cmd_str += f"R:{smoothed_commands['ring']:.0f},"
        cmd_str += f"P:{smoothed_commands['pinky']:.0f}\n"
        
        # Send via serial (implement actual serial communication)
        print(f"Actuator commands: {cmd_str.strip()}")
        
        # Update previous commands
        self.prev_commands = smoothed_commands.copy()
        
        return smoothed_commands
    
    def _smooth_commands(self, commands):
        """Apply exponential smoothing to commands"""
        smoothed = {}
        
        for finger, new_value in commands.items():
            prev_value = self.prev_commands.get(finger, 0)
            smoothed[finger] = self.alpha * prev_value + (1 - self.alpha) * new_value
            
            # Clamp to valid range
            smoothed[finger] = max(0, min(100, smoothed[finger]))
        
        return smoothed