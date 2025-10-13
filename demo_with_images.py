#!/usr/bin/env python3
"""
Demo script for Smart AI Safety Kit Detection System
Works with static images when camera is not available
"""

import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
import random
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image

class ImageBasedPPEDetector:
    def __init__(self):
        self.safety_items = {
            'helmet': {'name': 'Hard Hat', 'required': True, 'icon': '🦺'},
            'safety_vest': {'name': 'Safety Vest', 'required': True, 'icon': '🦺'},
            'gloves': {'name': 'Safety Gloves', 'required': True, 'icon': '🧤'},
            'safety_glasses': {'name': 'Safety Glasses', 'required': True, 'icon': '🥽'}
        }
        self.detection_history = []
        
    def mock_prediction(self):
        """Generate mock predictions for demonstration"""
        mock_results = {}
        
        # Simulate realistic detection patterns
        detection_probabilities = {
            'helmet': 0.7,
            'safety_vest': 0.6,
            'gloves': 0.5,
            'safety_glasses': 0.4
        }
        
        for item, prob in detection_probabilities.items():
            if random.random() < prob:
                mock_results[item] = random.uniform(0.6, 0.9)
            else:
                mock_results[f"no_{item}"] = random.uniform(0.6, 0.9)
        
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
            'compliance_percentage': (len(present_items) / len(self.safety_items)) * 100,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        }
    
    def process_image(self, image_path):
        """Process uploaded image and return detection results"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Get mock predictions
            detections = self.mock_prediction()
            
            # Analyze compliance
            compliance_analysis = self.analyze_safety_compliance(detections)
            
            # Store in history
            self.detection_history.append(compliance_analysis)
            if len(self.detection_history) > 100:
                self.detection_history.pop(0)
            
            return compliance_analysis
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return None

# Flask application
app = Flask(__name__)
detector = ImageBasedPPEDetector()

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    """Main page with image upload interface"""
    return render_template('image_demo.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload and detection"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    if file:
        # Save uploaded file
        filename = f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process image
        results = detector.process_image(filepath)
        
        if results:
            # Convert image to base64 for display
            with open(filepath, 'rb') as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            
            return jsonify({
                'success': True,
                'image': f"data:image/jpeg;base64,{img_base64}",
                'results': results
            })
        else:
            return jsonify({'error': 'Failed to process image'}), 500

@app.route('/get_detection_history')
def get_detection_history():
    """Get detection history"""
    return jsonify(detector.detection_history[-10:])

@app.route('/demo_detection')
def demo_detection():
    """Generate a demo detection without image upload"""
    detections = detector.mock_prediction()
    results = detector.analyze_safety_compliance(detections)
    return jsonify(results)

if __name__ == '__main__':
    print(" Smart AI Safety Kit Detection System - Image Demo")
    print("=" * 60)
    print("This demo works with uploaded images when camera is not available")
    print(" Open your browser and go to: http://localhost:5001")
    print(" Press Ctrl+C to stop")
    print("=" * 60)
    
    try:
        app.run(debug=False, host='0.0.0.0', port=5001)
    except KeyboardInterrupt:
        print("\n Demo stopped")
