import cv2
import numpy as np
import json
import os
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader

class RoboticHandDataset(Dataset):
    """Custom dataset for robotic hand control training"""
    
    def __init__(self, data_dir, transform=None, mode='train'):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.mode = mode
        
        # Load annotations
        self.annotations = self._load_annotations()
        
        # Define augmentation pipeline
        self.augmentation = self._get_augmentation_pipeline()
    
    def _load_annotations(self):
        """Load dataset annotations"""
        ann_file = self.data_dir / 'annotations.json'
        with open(ann_file, 'r') as f:
            return json.load(f)
    
    def _get_augmentation_pipeline(self):
        """Define augmentation strategies for robustness"""
        if self.mode == 'train':
            return A.Compose([
                # Lighting variations
                A.RandomBrightnessContrast(p=0.5),
                A.RandomGamma(p=0.3),
                A.HueSaturationValue(p=0.3),
                
                # Geometric transformations
                A.Rotate(limit=15, p=0.5),
                A.Perspective(p=0.3),
                
                # Noise and blur
                A.GaussNoise(p=0.2),
                A.MotionBlur(p=0.2),
                
                # Shadows and occlusions
                A.RandomShadow(p=0.3),
                A.CoarseDropout(max_holes=3, max_height=50, max_width=50, p=0.2),
                
                # Resize to model input size
                A.Resize(640, 480),
                
                # Normalize
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(640, 480),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        """Get a single sample"""
        ann = self.annotations[idx]
        
        # Load image
        img_path = self.data_dir / 'images' / ann['image']
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load depth map if available
        depth_path = self.data_dir / 'depth' / ann['depth']
        if depth_path.exists():
            depth = np.load(str(depth_path))
        else:
            # Generate synthetic depth if not available
            depth = self._generate_synthetic_depth(image, ann)
        
        # Apply augmentations
        if self.augmentation:
            augmented = self.augmentation(image=image)
            image = augmented['image']
        
        # Prepare target
        target = {
            'object_class': ann['object_class'],
            'bbox': torch.tensor(ann['bbox'], dtype=torch.float32),
            'shape_features': torch.tensor(ann['shape_features'], dtype=torch.float32),
            'grip_pattern': torch.tensor(ann['grip_pattern'], dtype=torch.float32),
            'depth': torch.tensor(depth, dtype=torch.float32)
        }
        
        return image, target
    
    def _generate_synthetic_depth(self, image, annotation):
        """Generate synthetic depth map for training"""
        h, w = image.shape[:2]
        depth = np.ones((h, w), dtype=np.float32) * 1.0  # Default 1m distance
        
        # Create depth gradient for object
        x1, y1, x2, y2 = annotation['bbox']
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        # Generate radial gradient
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)
        max_dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2) / 2
        
        # Apply gradient to bbox region
        mask = (X >= x1) & (X <= x2) & (Y >= y1) & (Y <= y2)
        depth[mask] = 0.3 + 0.4 * (dist_from_center[mask] / max_dist)
        
        return depth


class DatasetCreator:
    """Tools for creating and annotating datasets"""
    
    def __init__(self, output_dir='./dataset'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'images').mkdir(exist_ok=True)
        (self.output_dir / 'depth').mkdir(exist_ok=True)
        
        self.annotations = []
        
    def capture_training_data(self, camera_index=0):
        """Interactive tool for capturing and annotating training data"""
        cap = cv2.VideoCapture(camera_index)
        
        # Object classes for grip patterns
        object_classes = [
            'bottle', 'cup', 'pen', 'ball', 'book', 'phone',
            'scissors', 'spoon', 'key', 'card'
        ]
        
        print("Dataset Creation Tool")
        print("====================")
        print("Controls:")
        print("SPACE - Capture frame")
        print("1-9,0 - Select object class")
        print("ESC - Exit")
        print("\nObject classes:")
        for i, cls in enumerate(object_classes):
            print(f"{i+1}: {cls}")
        
        frame_count = 0
        current_class = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Display current selection
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Class: {object_classes[current_class]}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Frames: {frame_count}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Dataset Creator', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == ord(' '):  # SPACE - Capture
                # Save image
                img_name = f"frame_{frame_count:06d}.jpg"
                img_path = self.output_dir / 'images' / img_name
                cv2.imwrite(str(img_path), frame)
                
                # Interactive annotation
                annotation = self._annotate_frame(frame, img_name, object_classes[current_class])
                if annotation:
                    self.annotations.append(annotation)
                    frame_count += 1
                    print(f"Captured frame {frame_count}")
                
            elif ord('1') <= key <= ord('9'):
                current_class = key - ord('1')
            elif key == ord('0'):
                current_class = 9
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Save annotations
        self._save_annotations()
    
    def _annotate_frame(self, frame, img_name, object_class):
        """Interactive annotation for a single frame"""
        print(f"\nAnnotating {img_name} as {object_class}")
        
        # Simple bounding box annotation
        bbox = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select Object")
        
        if bbox[2] == 0 or bbox[3] == 0:
            return None
        
        # Get grip pattern for object class
        grip_pattern = self._get_default_grip_pattern(object_class)
        
        # Extract shape features from ROI
        x, y, w, h = bbox
        roi = frame[y:y+h, x:x+w]
        shape_features = self._extract_shape_features(roi)
        
        annotation = {
            'image': img_name,
            'object_class': object_class,
            'bbox': list(bbox),
            'grip_pattern': grip_pattern,
            'shape_features': shape_features,
            'depth': img_name.replace('.jpg', '_depth.npy')
        }
        
        return annotation
    
    def _get_default_grip_pattern(self, object_class):
        """Get default grip pattern for object class"""
        patterns = {
            'bottle': [70, 80, 80, 70, 60],  # thumb, index, middle, ring, pinky
            'cup': [60, 70, 70, 60, 50],
            'pen': [90, 90, 30, 10, 10],
            'ball': [50, 60, 60, 60, 50],
            'book': [40, 50, 50, 50, 40],
            'phone': [30, 40, 40, 40, 30],
            'scissors': [80, 80, 20, 20, 20],
            'spoon': [70, 80, 40, 30, 20],
            'key': [80, 90, 30, 20, 10],
            'card': [60, 70, 30, 20, 10]
        }
        return patterns.get(object_class, [50, 50, 50, 50, 50])
    
    def _extract_shape_features(self, roi):
        """Extract basic shape features"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return [0, 1, 0, 4, 0, 0]
        
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = w / h if h > 0 else 1
        
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        convexity = area / hull_area if hull_area > 0 else 0
        
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        num_vertices = len(approx)
        
        return [circularity, aspect_ratio, convexity, num_vertices, area, perimeter]
    
    def _save_annotations(self):
        """Save annotations to JSON file"""
        ann_file = self.output_dir / 'annotations.json'
        with open(ann_file, 'w') as f:
            json.dump(self.annotations, f, indent=2)
        print(f"Saved {len(self.annotations)} annotations to {ann_file}")


# Example training script
def train_grip_predictor(dataset_path, epochs=50):
    """Train a neural network for grip pattern prediction"""
    
    # Create dataset
    train_dataset = RoboticHandDataset(dataset_path, mode='train')
    val_dataset = RoboticHandDataset(dataset_path, mode='val')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Initialize model (simplified example)
    model = GripPredictorNet()
    
    # Training loop would go here...
    print(f"Training on {len(train_dataset)} samples...")


class GripPredictorNet(torch.nn.Module):
    """Neural network for predicting grip patterns"""
    
    def __init__(self, num_classes=10, num_fingers=5):
        super().__init__()
        
        # Feature extraction backbone (can use pre-trained ResNet)
        self.backbone = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(128, 256, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Shape feature processing
        self.shape_fc = torch.nn.Sequential(
            torch.nn.Linear(6, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 64)
        )
        
        # Fusion and prediction
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(256 + 64 + num_classes, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, num_fingers)
        )
        
    def forward(self, image, shape_features, object_class):
        # Extract image features
        img_features = self.backbone(image).squeeze()
        
        # Process shape features
        shape_encoded = self.shape_fc(shape_features)
        
        # One-hot encode object class
        class_onehot = torch.nn.functional.one_hot(object_class, num_classes=10)
        
        # Concatenate all features
        combined = torch.cat([img_features, shape_encoded, class_onehot], dim=1)
        
        # Predict grip pattern (0-100% for each finger)
        grip_pattern = torch.sigmoid(self.fusion(combined)) * 100
        
        return grip_pattern