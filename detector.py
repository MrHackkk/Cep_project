"""
Detection Backend Module
Handles PPE detection using AI model and camera
"""

import cv2
import numpy as np
import tensorflow as tf
import os
from datetime import datetime
import random


class PPEDetector:
    """Main PPE detection class"""
    
    def __init__(self, model_path="ppe_model/best_model.h5"):
        self.model = None
        self.load_model(model_path)
        self.input_size = (224, 224)
        self.class_names = [
            'person', 'helmet', 'safety_vest', 'gloves', 'safety_glasses',
            'no_helmet', 'no_vest', 'no_gloves', 'no_glasses'
        ]
        self.safety_items = {
            'helmet': {'name': 'Hard Hat', 'required': True},
            'safety_vest': {'name': 'Safety Vest', 'required': True},
            'gloves': {'name': 'Safety Gloves', 'required': True},
            'safety_glasses': {'name': 'Safety Glasses', 'required': True}
        }
        self.camera = None
        self.detection_active = False
        
    def load_model(self, model_path):
        """Load the trained PPE detection model"""
        try:
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
                print(f"Model loaded successfully from {model_path}")
            else:
                print(f"Model not found at {model_path}. Using mock predictions.")
                self.model = None
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def preprocess_frame(self, frame):
        """Preprocess frame for model prediction"""
        resized = cv2.resize(frame, self.input_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype('float32') / 255.0
        return np.expand_dims(normalized, axis=0)
    
    def predict_ppe(self, frame):
        """Predict PPE items in the frame"""
        if self.model is None:
            return self.mock_prediction()
        
        try:
            processed_frame = self.preprocess_frame(frame)
            predictions = self.model.predict(processed_frame, verbose=0)
            
            top_indices = np.argsort(predictions[0])[-4:][::-1]
            results = {}
            
            for idx in top_indices:
                class_name = self.class_names[idx] if idx < len(self.class_names) else f"class_{idx}"
                confidence = float(predictions[0][idx])
                if confidence > 0.3:
                    results[class_name] = confidence
            
            return results
        except Exception as e:
            print(f"Prediction error: {e}")
            return self.mock_prediction()
    
    def mock_prediction(self):
        """Mock prediction when model is not available"""
        mock_results = {}
        
        if random.random() > 0.3:
            mock_results['helmet'] = random.uniform(0.6, 0.9)
        if random.random() > 0.4:
            mock_results['safety_vest'] = random.uniform(0.5, 0.8)
        if random.random() > 0.5:
            mock_results['gloves'] = random.uniform(0.4, 0.7)
        if random.random() > 0.6:
            mock_results['safety_glasses'] = random.uniform(0.3, 0.6)
        
        return mock_results
    
    def analyze_safety_compliance(self, detections):
        """Analyze safety compliance based on detections"""
        compliance = {}
        missing_items = []
        present_items = []
        
        for item, data in self.safety_items.items():
            is_present = item in detections and detections[item] > 0.5
            is_missing = f"no_{item}" in detections and detections[f"no_{item}"] > 0.5
            
            if is_present:
                present_items.append({
                    'name': data['name'],
                    'confidence': detections[item],
                    'status': 'present'
                })
                compliance[item] = True
            elif is_missing or not is_present:
                missing_items.append({
                    'name': data['name'],
                    'confidence': detections.get(f"no_{item}", 0.8),
                    'status': 'missing'
                })
                compliance[item] = False
        
        return {
            'compliance': compliance,
            'present_items': present_items,
            'missing_items': missing_items,
            'is_compliant': len(missing_items) == 0,
            'compliance_percentage': (len(present_items) / len(self.safety_items)) * 100
        }
    
    def start_camera(self, camera_index=0):
        """Start camera capture"""
        try:
            self.camera = cv2.VideoCapture(camera_index)
            if not self.camera.isOpened():
                raise Exception("Could not open camera")
            
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            print("Camera started successfully")
            return True
        except Exception as e:
            print(f"Camera error: {e}")
            return False
    
    def stop_camera(self):
        """Stop camera capture"""
        if self.camera:
            self.camera.release()
            self.camera = None
            print("Camera stopped")
    
    def get_frame(self):
        """Get current frame from camera"""
        if self.camera and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                return frame
        return None
    
    def process_frame(self, frame):
        """Process frame and return detection results"""
        if frame is None:
            return None
        
        detections = self.predict_ppe(frame)
        compliance_analysis = self.analyze_safety_compliance(detections)
        
        compliance_analysis['timestamp'] = datetime.now().strftime("%H:%M:%S")
        
        return compliance_analysis
    
    def save_violation_screenshot(self, frame):
        """Save screenshot when violation is detected"""
        try:
            os.makedirs('violations', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'violations/violation_{timestamp}.jpg'
            cv2.imwrite(filename, frame)
            return filename
        except Exception as e:
            print(f"Error saving screenshot: {e}")
            return None

