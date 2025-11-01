"""
Detection Interface
Camera-based PPE detection - No login required
"""

from flask import Flask, render_template, Response, jsonify
import cv2
import time
from datetime import datetime
import winsound
import os

from detector import PPEDetector
from storage import DataStorage

app = Flask(__name__)

# Initialize detector and storage
detector = PPEDetector()
storage = DataStorage()

# Track entry/exit times
entry_times = {}
current_status = {}


@app.route('/')
def index():
    """Main detection page - no login required"""
    return render_template('detect.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route - clean camera feed only"""
    def generate_frames():
        # Create black frame if camera not available
        import numpy as np
        black_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(black_frame, "Camera not available", (400, 360), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(black_frame, "Using mock detection data", (350, 420), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        while detector.detection_active:
            # Check if detection was stopped
            if not detector.detection_active:
                break
            frame = detector.get_frame()
            if frame is None:
                # Use mock detection if no camera
                detection_results = detector.mock_prediction()
                detection_results = detector.analyze_safety_compliance(detection_results)
                detection_results['timestamp'] = datetime.now().strftime("%H:%M:%S")
                frame_to_show = black_frame
            else:
                # Get detection results
                detection_results = detector.process_frame(frame)
                # Show clean camera feed only - no overlays
                frame_to_show = frame
            
            # Save detection to storage
            if detection_results:
                storage.save_detection(detection_results)
                
                # Track entry/exit
                now = datetime.now()
                date_str = now.strftime('%Y-%m-%d')
                time_str = now.strftime('%H:%M:%S')
                
                if date_str not in entry_times:
                    entry_times[date_str] = {'in': None, 'out': None}
                
                # Mark entry/exit based on compliance
                if detection_results.get('is_compliant'):
                    if entry_times[date_str]['in'] is None:
                        entry_times[date_str]['in'] = time_str
                    if current_status.get(date_str) != 'in':
                        current_status[date_str] = 'in'
                        entry_times[date_str]['out'] = None
                else:
                    if entry_times[date_str]['in'] is not None and current_status.get(date_str) == 'in':
                        if entry_times[date_str]['out'] is None:
                            entry_times[date_str]['out'] = time_str
                        current_status[date_str] = 'out'
                
                # Save violation screenshot and play alert (only if camera available)
                if not detection_results.get('is_compliant') and frame is not None:
                    detector.save_violation_screenshot(frame)
                    try:
                        winsound.Beep(1000, 500)
                    except:
                        pass
            
            # Encode and send clean frame
            ret, buffer = cv2.imencode('.jpg', frame_to_show, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)
    
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_detection')
def start_detection():
    """Start PPE detection"""
    if not detector.camera:
        success = detector.start_camera()
        if not success:
            detector.detection_active = True
            return jsonify({'success': True, 'message': 'Detection started (using mock data)'})
    
    detector.detection_active = True
    return jsonify({'success': True, 'message': 'Detection started'})


@app.route('/stop_detection')
def stop_detection():
    """Stop PPE detection"""
    detector.detection_active = False
    if detector.camera:
        detector.stop_camera()
        detector.camera = None
    return jsonify({'success': True, 'message': 'Detection stopped'})


@app.route('/get_detection_status')
def get_detection_status():
    """Get current detection status"""
    if storage.data:
        latest = storage.data[-1]
        return jsonify(latest)
    return jsonify({'message': 'No detections yet'})


if __name__ == '__main__':
    print("Starting Detection Interface...")
    if detector.start_camera():
        print("Camera initialized successfully")
        print("Open your browser and go to: http://localhost:5000")
    else:
        print("Warning: Camera not available. Using mock detection.")
    
    try:
        app.run(debug=False, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        detector.stop_camera()

    