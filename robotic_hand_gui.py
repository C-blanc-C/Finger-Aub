import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import torch
from ultralytics import YOLO
from pathlib import Path
import json

class FingerWidget(QWidget):
    """Custom widget for a single finger with percentage bar"""
    
    def __init__(self, finger_name, color, parent=None):
        super().__init__(parent)
        self.finger_name = finger_name
        self.color = color
        self.percentage = 0
        self.target_percentage = 0
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.animate_percentage)
        self.animation_timer.start(20)  # 50 FPS animation
        
        self.setMinimumSize(80, 200)
        
    def set_percentage(self, value):
        """Set target percentage for smooth animation"""
        self.target_percentage = max(0, min(100, value))
        
    def animate_percentage(self):
        """Smooth animation of percentage changes"""
        if abs(self.percentage - self.target_percentage) > 0.5:
            diff = self.target_percentage - self.percentage
            self.percentage += diff * 0.15  # Smooth factor
            self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw finger outline
        finger_rect = QRect(10, 10, 60, 150)
        
        # Finger shape (rounded rectangle)
        painter.setPen(QPen(QColor(100, 100, 100), 2))
        painter.setBrush(QBrush(QColor(240, 220, 200)))
        painter.drawRoundedRect(finger_rect, 20, 20)
        
        # Draw percentage bar
        bar_height = int((self.percentage / 100) * 140)
        bar_rect = QRect(15, 155 - bar_height, 50, bar_height)
        
        # Gradient for bar
        gradient = QLinearGradient(0, 155, 0, 15)
        gradient.setColorAt(0, QColor(self.color).darker(150))
        gradient.setColorAt(1, QColor(self.color))
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(bar_rect, 15, 15)
        
        # Draw percentage text
        painter.setPen(QPen(Qt.black, 1))
        painter.setFont(QFont("Arial", 10, QFont.Bold))
        painter.drawText(finger_rect, Qt.AlignCenter, f"{self.percentage:.0f}%")
        
        # Draw finger name
        painter.setFont(QFont("Arial", 9))
        name_rect = QRect(0, 165, 80, 20)
        painter.drawText(name_rect, Qt.AlignCenter, self.finger_name)


class HandVisualizationWidget(QWidget):
    """Widget showing the complete hand with all fingers"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Create finger widgets
        self.fingers = {
            'thumb': FingerWidget('Thumb', '#FF6B6B'),
            'index': FingerWidget('Index', '#4ECDC4'),
            'middle': FingerWidget('Middle', '#45B7D1'),
            'ring': FingerWidget('Ring', '#96CEB4'),
            'pinky': FingerWidget('Pinky', '#FECA57')
        }
        
        # Layout
        layout = QHBoxLayout()
        layout.setSpacing(15)
        
        # Add fingers in order
        for finger_name in ['thumb', 'index', 'middle', 'ring', 'pinky']:
            layout.addWidget(self.fingers[finger_name])
        
        # Palm visualization
        self.palm_widget = QWidget()
        self.palm_widget.setMinimumSize(100, 100)
        
        main_layout = QVBoxLayout()
        main_layout.addLayout(layout)
        main_layout.addWidget(self.palm_widget)
        
        self.setLayout(main_layout)
        
    def set_finger_percentages(self, percentages):
        """Update all finger percentages"""
        for finger, percentage in percentages.items():
            if finger in self.fingers:
                self.fingers[finger].set_percentage(percentage)
    
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw palm
        palm_rect = QRect(50, 220, 350, 80)
        painter.setPen(QPen(QColor(100, 100, 100), 2))
        painter.setBrush(QBrush(QColor(240, 220, 200)))
        painter.drawRoundedRect(palm_rect, 20, 20)
        
        # Draw palm label
        painter.setPen(QPen(Qt.black, 1))
        painter.setFont(QFont("Arial", 12))
        painter.drawText(palm_rect, Qt.AlignCenter, "Palm")


class MLProcessor(QObject):
    """ML processing in separate thread"""
    
    results_ready = pyqtSignal(dict)
    status_update = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.yolo_model = None
        self.depth_model = None
        self.initialized = False
        
    def initialize_models(self):
        """Load ML models"""
        try:
            self.status_update.emit("Loading YOLO model...")
            self.yolo_model = YOLO('yolov8m.pt')
            
            self.status_update.emit("Loading depth model...")
            # Load MiDaS for depth estimation
            self.depth_model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid", trust_repo=True)
            self.depth_model.eval()
            
            # Load transforms
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            self.transform = midas_transforms.dpt_transform
            
            self.initialized = True
            self.status_update.emit("Models loaded successfully!")
            
        except Exception as e:
            self.status_update.emit(f"Error loading models: {str(e)}")
    
    def process_image(self, image_path):
        """Process image and return grip recommendations"""
        if not self.initialized:
            self.status_update.emit("Models not initialized!")
            return
        
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                self.status_update.emit("Failed to load image!")
                return
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 1. Object Detection
            self.status_update.emit("Detecting objects...")
            detections = self._detect_objects(image_rgb)
            
            if not detections:
                self.status_update.emit("No objects detected!")
                self.results_ready.emit({
                    'thumb': 0, 'index': 0, 'middle': 0, 'ring': 0, 'pinky': 0,
                    'object': 'None', 'distance': 0
                })
                return
            
            # 2. Depth Estimation
            self.status_update.emit("Estimating depth...")
            depth_map = self._estimate_depth(image_rgb)
            
            # 3. Get primary object and distance
            primary_obj = detections[0]  # Use first detection
            distance = self._get_object_distance(primary_obj, depth_map)
            
            # 4. Generate grip pattern
            self.status_update.emit(f"Generating grip for {primary_obj['class']} at {distance:.2f}m")
            grip_pattern = self._generate_grip_pattern(primary_obj, distance)
            
            # Add metadata
            grip_pattern['object'] = primary_obj['class']
            grip_pattern['distance'] = distance
            grip_pattern['confidence'] = primary_obj['confidence']
            
            self.results_ready.emit(grip_pattern)
            self.status_update.emit("Processing complete!")
            
        except Exception as e:
            self.status_update.emit(f"Processing error: {str(e)}")
    
    def _detect_objects(self, image):
        """Run YOLO object detection"""
        results = self.yolo_model(image, conf=0.5)
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
    
    def _estimate_depth(self, image):
        """Estimate depth using MiDaS"""
        # Prepare input
        input_batch = self.transform(image).unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            prediction = self.depth_model(input_batch)
        
        # Resize to original dimensions
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        
        # Normalize to real-world distances (assuming calibrated camera)
        # This is a simplified conversion - in practice, you'd calibrate
        depth_map = self._normalize_depth(depth_map)
        
        return depth_map
    
    def _normalize_depth(self, depth_map, min_distance=0.1, max_distance=2.0):
        """Convert relative depth to meters"""
        depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        depth_meters = depth_normalized * (max_distance - min_distance) + min_distance
        return depth_meters
    
    def _get_object_distance(self, detection, depth_map):
        """Get average distance of detected object"""
        x1, y1, x2, y2 = map(int, detection['bbox'])
        depth_roi = depth_map[y1:y2, x1:x2]
        return np.mean(depth_roi)
    
    def _generate_grip_pattern(self, obj, distance):
        """Generate grip pattern based on object and distance"""
        # Distance-based scaling (closer = more closed)
        distance_factor = 1.0 - min(distance / 1.0, 1.0)  # Normalize to 0-1
        
        # Predefined grip patterns for different objects
        grip_patterns = {
            'bottle': {'thumb': 0.7, 'index': 0.8, 'middle': 0.8, 'ring': 0.7, 'pinky': 0.6},
            'cup': {'thumb': 0.6, 'index': 0.7, 'middle': 0.7, 'ring': 0.6, 'pinky': 0.5},
            'cell phone': {'thumb': 0.3, 'index': 0.4, 'middle': 0.4, 'ring': 0.4, 'pinky': 0.3},
            'book': {'thumb': 0.4, 'index': 0.5, 'middle': 0.5, 'ring': 0.5, 'pinky': 0.4},
            'scissors': {'thumb': 0.8, 'index': 0.8, 'middle': 0.2, 'ring': 0.2, 'pinky': 0.2},
            'spoon': {'thumb': 0.7, 'index': 0.8, 'middle': 0.4, 'ring': 0.3, 'pinky': 0.2},
            'keyboard': {'thumb': 0.2, 'index': 0.3, 'middle': 0.3, 'ring': 0.3, 'pinky': 0.2},
            'mouse': {'thumb': 0.5, 'index': 0.6, 'middle': 0.6, 'ring': 0.4, 'pinky': 0.3},
            'pen': {'thumb': 0.9, 'index': 0.9, 'middle': 0.3, 'ring': 0.1, 'pinky': 0.1},
            'apple': {'thumb': 0.5, 'index': 0.6, 'middle': 0.6, 'ring': 0.6, 'pinky': 0.5},
            'banana': {'thumb': 0.6, 'index': 0.7, 'middle': 0.7, 'ring': 0.6, 'pinky': 0.5},
            'orange': {'thumb': 0.5, 'index': 0.6, 'middle': 0.6, 'ring': 0.6, 'pinky': 0.5}
        }
        
        # Get base pattern or use default
        base_pattern = grip_patterns.get(
            obj['class'].lower(), 
            {'thumb': 0.5, 'index': 0.5, 'middle': 0.5, 'ring': 0.5, 'pinky': 0.5}
        )
        
        # Apply distance scaling and convert to percentage
        grip_commands = {}
        for finger, base_value in base_pattern.items():
            grip_commands[finger] = base_value * distance_factor * 100
        
        return grip_commands


class RoboticHandGUI(QMainWindow):
    """Main GUI application"""
    
    def __init__(self):
        super().__init__()
        self.ml_processor = MLProcessor()
        self.ml_thread = QThread()
        self.ml_processor.moveToThread(self.ml_thread)
        
        # Connect signals
        self.ml_processor.results_ready.connect(self.update_hand_visualization)
        self.ml_processor.status_update.connect(self.update_status)
        
        self.ml_thread.started.connect(self.ml_processor.initialize_models)
        self.ml_thread.start()
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Robotic Hand Control System")
        self.setGeometry(100, 100, 900, 700)
        
        # Set application style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QLabel {
                color: #333;
            }
        """)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Title
        title = QLabel("Robotic Hand ML Control System")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 20, QFont.Bold))
        title.setStyleSheet("color: #2c3e50; margin: 20px;")
        main_layout.addWidget(title)
        
        # Content layout
        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout)
        
        # Left panel - Image display and controls
        left_panel = QVBoxLayout()
        content_layout.addLayout(left_panel, 1)
        
        # Image display
        self.image_label = QLabel("No image loaded")
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setMaximumSize(600, 450)
        self.image_label.setScaledContents(True)
        self.image_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #ccc;
                border-radius: 10px;
                background-color: white;
                padding: 20px;
            }
        """)
        self.image_label.setAlignment(Qt.AlignCenter)
        left_panel.addWidget(self.image_label)
        
        # Upload button
        self.upload_btn = QPushButton("Upload Image")
        self.upload_btn.clicked.connect(self.upload_image)
        self.upload_btn.setMinimumHeight(40)
        left_panel.addWidget(self.upload_btn)
        
        # Process button
        self.process_btn = QPushButton("Process Image")
        self.process_btn.clicked.connect(self.process_image)
        self.process_btn.setEnabled(False)
        self.process_btn.setMinimumHeight(40)
        left_panel.addWidget(self.process_btn)
        
        # Results display
        results_group = QGroupBox("Detection Results")
        results_layout = QVBoxLayout()
        results_group.setLayout(results_layout)
        
        self.object_label = QLabel("Object: -")
        self.distance_label = QLabel("Distance: -")
        self.confidence_label = QLabel("Confidence: -")
        
        for label in [self.object_label, self.distance_label, self.confidence_label]:
            label.setFont(QFont("Arial", 12))
            results_layout.addWidget(label)
        
        left_panel.addWidget(results_group)
        
        # Right panel - Hand visualization
        right_panel = QVBoxLayout()
        content_layout.addLayout(right_panel, 1)
        
        hand_group = QGroupBox("Hand Visualization")
        hand_layout = QVBoxLayout()
        hand_group.setLayout(hand_layout)
        
        self.hand_widget = HandVisualizationWidget()
        hand_layout.addWidget(self.hand_widget)
        
        right_panel.addWidget(hand_group)
        
        # Manual control
        manual_group = QGroupBox("Manual Control")
        manual_layout = QVBoxLayout()
        manual_group.setLayout(manual_layout)
        
        # Sliders for manual control
        self.sliders = {}
        for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']:
            slider_layout = QHBoxLayout()
            
            label = QLabel(f"{finger.capitalize()}:")
            label.setMinimumWidth(60)
            slider_layout.addWidget(label)
            
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(0)
            slider.valueChanged.connect(lambda v, f=finger: self.manual_control(f, v))
            self.sliders[finger] = slider
            slider_layout.addWidget(slider)
            
            value_label = QLabel("0%")
            value_label.setMinimumWidth(40)
            slider.valueChanged.connect(lambda v, l=value_label: l.setText(f"{v}%"))
            slider_layout.addWidget(value_label)
            
            manual_layout.addLayout(slider_layout)
        
        right_panel.addWidget(manual_group)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Image path storage
        self.current_image_path = None
        
    def upload_image(self):
        """Handle image upload"""
        file_dialog = QFileDialog()
        image_path, _ = file_dialog.getOpenFileName(
            self, 
            "Select Image", 
            "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if image_path:
            # Display image
            pixmap = QPixmap(image_path)
            scaled_pixmap = pixmap.scaled(
                self.image_label.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
            
            self.current_image_path = image_path
            self.process_btn.setEnabled(True)
            self.update_status(f"Image loaded: {Path(image_path).name}")
    
    def process_image(self):
        """Process the uploaded image"""
        if self.current_image_path:
            self.process_btn.setEnabled(False)
            self.update_status("Processing image...")
            
            # Process in ML thread
            self.ml_processor.process_image(self.current_image_path)

    
    def update_hand_visualization(self, results):
        """Update hand visualization with ML results"""
        # Update finger positions
        finger_percentages = {
            'thumb': results.get('thumb', 0),
            'index': results.get('index', 0),
            'middle': results.get('middle', 0),
            'ring': results.get('ring', 0),
            'pinky': results.get('pinky', 0)
        }
        
        self.hand_widget.set_finger_percentages(finger_percentages)
        
        # Update sliders
        for finger, percentage in finger_percentages.items():
            if finger in self.sliders:
                self.sliders[finger].setValue(int(percentage))
        
        # Update results display
        self.object_label.setText(f"Object: {results.get('object', 'Unknown')}")
        self.distance_label.setText(f"Distance: {results.get('distance', 0):.2f}m")
        self.confidence_label.setText(f"Confidence: {results.get('confidence', 0):.1%}")
        
        self.process_btn.setEnabled(True)
    
    def manual_control(self, finger, value):
        """Handle manual slider control"""
        self.hand_widget.fingers[finger].set_percentage(value)
    
    def update_status(self, message):
        """Update status bar"""
        self.status_bar.showMessage(message)
    
    def closeEvent(self, event):
        """Clean up on close"""
        self.ml_thread.quit()
        self.ml_thread.wait()
        event.accept()


def main():
    app = QApplication(sys.argv)
    
    # Set application icon and style
    app.setApplicationName("Robotic Hand Control")
    app.setStyle('Fusion')
    
    # Create and show main window
    window = RoboticHandGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        window = RoboticHandGUI()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print("\n=== ERROR OCCURRED ===")
        print(e)
