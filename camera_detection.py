import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, Response, jsonify
import threading
import time
import os
from datetime import datetime
import json

class SmartPPEDetector:
    def __init__(self, model_path="ppe_model/best_model.h5"):
        self.model = None
        self.load_model(model_path)
        self.input_size = (224, 224)
        self.class_names = [
            'person', 'helmet', 'safety_vest', 'gloves', 'safety_glasses',
            'no_helmet', 'no_vest', 'no_gloves', 'no_glasses'
        ]
        self.safety_items = {
            'helmet': {'name': 'Hard Hat', 'required': True, 'icon': '👷'},
            'safety_vest': {'name': 'Safety Vest', 'required': True, 'icon': '🦺'},
            'gloves': {'name': 'Safety Gloves', 'required': True, 'icon': '🧤'},
            'safety_glasses': {'name': 'Safety Glasses', 'required': True, 'icon': '🥽'}
        }
        self.detection_history = []
        self.camera = None
        self.frame = None
        self.detection_active = False
        
    def load_model(self, model_path):
        """Load the trained PPE detection model"""
        try:
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
                print(f"✅ Model loaded successfully from {model_path}")
            else:
                print(f"⚠️ Model not found at {model_path}. Using mock predictions.")
                self.model = None
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
    
    def preprocess_frame(self, frame):
        """Preprocess frame for model prediction"""
        # Resize frame to model input size
        resized = cv2.resize(frame, self.input_size)
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Normalize pixel values
        normalized = rgb.astype('float32') / 255.0
        # Add batch dimension
        return np.expand_dims(normalized, axis=0)
    
    def predict_ppe(self, frame):
        """Predict PPE items in the frame"""
        if self.model is None:
            # Mock prediction for demonstration
            return self.mock_prediction()
        
        try:
            processed_frame = self.preprocess_frame(frame)
            predictions = self.model.predict(processed_frame, verbose=0)
            
            # Get top predictions
            top_indices = np.argsort(predictions[0])[-4:][::-1]
            results = {}
            
            for idx in top_indices:
                class_name = self.class_names[idx] if idx < len(self.class_names) else f"class_{idx}"
                confidence = float(predictions[0][idx])
                if confidence > 0.3:  # Threshold for detection
                    results[class_name] = confidence
            
            return results
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return self.mock_prediction()
    
    def mock_prediction(self):
        """Mock prediction for demonstration when model is not available"""
        import random
        mock_results = {}
        
        # Simulate random detections
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
                    'icon': data['icon'],
                    'confidence': detections[item],
                    'status': 'present'
                })
                compliance[item] = True
            elif is_missing or not is_present:
                missing_items.append({
                    'name': data['name'],
                    'icon': data['icon'],
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
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            print("✅ Camera started successfully")
            return True
        except Exception as e:
            print(f"❌ Camera error: {e}")
            return False
    
    def stop_camera(self):
        """Stop camera capture"""
        if self.camera:
            self.camera.release()
            self.camera = None
            print("📹 Camera stopped")
    
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
        
        # Get PPE predictions
        detections = self.predict_ppe(frame)
        
        # Analyze safety compliance
        compliance_analysis = self.analyze_safety_compliance(detections)
        
        # Add timestamp
        compliance_analysis['timestamp'] = datetime.now().strftime("%H:%M:%S")
        
        # Store in history
        self.detection_history.append(compliance_analysis)
        if len(self.detection_history) > 100:  # Keep last 100 detections
            self.detection_history.pop(0)
        
        return compliance_analysis

# Flask application
app = Flask(__name__)

# Global detector instance
detector = SmartPPEDetector()

@app.route('/')
def index():
    """Main page with camera interface"""
    return render_template('camera_interface.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    def generate_frames():
        while detector.detection_active:
            frame = detector.get_frame()
            if frame is not None:
                # Process frame for detection
                detection_results = detector.process_frame(frame)
                
                # Draw detection overlay on frame
                annotated_frame = draw_detection_overlay(frame, detection_results)
                
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', annotated_frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)  # ~30 FPS
    
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_detection')
def start_detection():
    """Start PPE detection"""
    if not detector.camera:
        success = detector.start_camera()
        if not success:
            return jsonify({'success': False, 'message': 'Could not start camera'})
    
    detector.detection_active = True
    return jsonify({'success': True, 'message': 'Detection started'})

@app.route('/stop_detection')
def stop_detection():
    """Stop PPE detection"""
    detector.detection_active = False
    return jsonify({'success': True, 'message': 'Detection stopped'})

@app.route('/get_detection_status')
def get_detection_status():
    """Get current detection status"""
    if detector.detection_history:
        latest = detector.detection_history[-1]
        return jsonify(latest)
    return jsonify({'message': 'No detections yet'})

@app.route('/get_detection_history')
def get_detection_history():
    """Get detection history"""
    return jsonify(detector.detection_history[-10:])  # Last 10 detections

def draw_detection_overlay(frame, detection_results):
    """Draw detection results overlay on frame"""
    if detection_results is None:
        return frame
    
    # Create overlay
    overlay = frame.copy()
    height, width = frame.shape[:2]
    
    # Draw compliance status
    is_compliant = detection_results.get('is_compliant', False)
    compliance_pct = detection_results.get('compliance_percentage', 0)
    
    # Status bar
    status_color = (0, 255, 0) if is_compliant else (0, 0, 255)
    cv2.rectangle(overlay, (10, 10), (width-10, 80), status_color, -1)
    cv2.rectangle(overlay, (10, 10), (width-10, 80), (255, 255, 255), 2)
    
    # Status text
    status_text = "✅ SAFETY COMPLIANT" if is_compliant else "❌ SAFETY VIOLATION"
    cv2.putText(overlay, status_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(overlay, f"Compliance: {compliance_pct:.1f}%", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw present items
    present_items = detection_results.get('present_items', [])
    y_offset = 100
    for item in present_items:
        text = f"✅ {item['name']} ({item['confidence']:.2f})"
        cv2.putText(overlay, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_offset += 25
    
    # Draw missing items
    missing_items = detection_results.get('missing_items', [])
    for item in missing_items:
        text = f"❌ {item['name']} - WEAR REQUIRED"
        cv2.putText(overlay, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        y_offset += 25
    
    # Blend overlay with frame
    alpha = 0.7
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

if __name__ == '__main__':
    print(" Smart AI Safety Kit Detection System")
    print("=" * 50)
    print("Starting camera-based PPE detection...")
    
    # Start camera
    if detector.start_camera():
        print("✅ Camera initialized successfully")
        print("🌐 Starting web interface...")
        print("📱 Open your browser and go to: http://localhost:5000")
        print("🛑 Press Ctrl+C to stop")
        
        try:
            app.run(debug=False, host='0.0.0.0', port=5000)
        except KeyboardInterrupt:
            print("\n Shutting down...")
        finally:
            detector.stop_camera()
    else:
        print("❌ Failed to initialize camera")
        print(" Make sure your camera is connected and not being used by another application")
