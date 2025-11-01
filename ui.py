"""
UI Module
Flask routes and web interface
"""

from flask import Flask, render_template, Response, jsonify, send_file, request, session, redirect
import cv2
import time
from datetime import datetime
import winsound
import os

from detector import PPEDetector
from storage import DataStorage
from admin import require_login, login_user, logout_user, is_admin

app = Flask(__name__)
app.secret_key = 'safety-detection-system-secret-key-2024'

# Initialize detector and storage
detector = PPEDetector()
storage = DataStorage()

# Track entry/exit times
entry_times = {}
current_status = {}


def draw_detection_overlay(frame, detection_results):
    """Draw detection results on frame"""
    if detection_results is None:
        return frame
    
    overlay = frame.copy()
    height, width = frame.shape[:2]
    
    is_compliant = detection_results.get('is_compliant', False)
    compliance_pct = detection_results.get('compliance_percentage', 0)
    
    # Status bar
    status_color = (0, 255, 0) if is_compliant else (0, 0, 255)
    cv2.rectangle(overlay, (10, 10), (width-10, 80), status_color, -1)
    cv2.rectangle(overlay, (10, 10), (width-10, 80), (255, 255, 255), 2)
    
    # Status text
    status_text = "SAFETY COMPLIANT" if is_compliant else "SAFETY VIOLATION"
    cv2.putText(overlay, status_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(overlay, f"Compliance: {compliance_pct:.1f}%", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw present items
    present_items = detection_results.get('present_items', [])
    y_offset = 100
    for item in present_items:
        text = f"{item['name']} ({item['confidence']:.2f})"
        cv2.putText(overlay, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_offset += 25
    
    # Draw missing items
    missing_items = detection_results.get('missing_items', [])
    for item in missing_items:
        text = f"{item['name']} - WEAR REQUIRED"
        cv2.putText(overlay, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        y_offset += 25
    
    alpha = 0.7
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if login_user(username, password):
            return redirect('/')
        else:
            return render_template('login.html', error='Invalid username or password')
    
    return render_template('login.html')


@app.route('/logout')
def logout():
    """Logout"""
    logout_user()
    return redirect('/login')


@app.route('/')
@require_login
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/video_feed')
@require_login
def video_feed():
    """Video streaming route"""
    def generate_frames():
        while detector.detection_active:
            frame = detector.get_frame()
            if frame is not None:
                detection_results = detector.process_frame(frame)
                
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
                    
                    # Save violation screenshot and play alert
                    if not detection_results.get('is_compliant'):
                        detector.save_violation_screenshot(frame)
                        try:
                            winsound.Beep(1000, 500)
                        except:
                            pass
                
                annotated_frame = draw_detection_overlay(frame, detection_results)
                ret, buffer = cv2.imencode('.jpg', annotated_frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)
    
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_detection')
@require_login
def start_detection():
    """Start PPE detection"""
    if not detector.camera:
        success = detector.start_camera()
        if not success:
            return jsonify({'success': False, 'message': 'Could not start camera'})
    
    detector.detection_active = True
    return jsonify({'success': True, 'message': 'Detection started'})


@app.route('/stop_detection')
@require_login
def stop_detection():
    """Stop PPE detection"""
    detector.detection_active = False
    return jsonify({'success': True, 'message': 'Detection stopped'})


@app.route('/get_status')
@require_login
def get_status():
    """Get current detection status"""
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Get today's entry/exit times
    in_time = entry_times.get(today, {}).get('in', 'Not recorded')
    out_time = entry_times.get(today, {}).get('out', 'Not recorded')
    current = current_status.get(today, 'out')
    
    # Get today's detections
    today_detections = storage.get_today_detections()
    
    # Get statistics
    stats = storage.get_statistics()
    
    return jsonify({
        'date': today,
        'current_time': datetime.now().strftime('%H:%M:%S'),
        'in_time': in_time,
        'out_time': out_time,
        'current_status': current,
        'today_detections': len(today_detections),
        'today_violations': stats['today_violations'],
        'compliance_rate': stats['compliance_rate'],
        'total_detections': stats['total_detections']
    })


@app.route('/get_daily_entries')
@require_login
def get_daily_entries():
    """Get daily entry history"""
    today = datetime.now().strftime('%Y-%m-%d')
    today_detections = storage.get_today_detections()
    
    entries = []
    for det in today_detections:
        entries.append({
            'date': det.get('date', today),
            'time': det.get('time', ''),
            'compliant': det.get('is_compliant', False),
            'compliance_percentage': det.get('compliance_percentage', 0)
        })
    
    return jsonify(entries)


@app.route('/export_csv')
@require_login
def export_csv():
    """Export detection history to CSV"""
    if storage.export_to_csv('detectionsexport.csv'):
        return send_file('detectionsexport.csv', 
                        as_attachment=True,
                        download_name=f'detections_{datetime.now().strftime("%Y%m%d")}.csv')
    return jsonify({'error': 'Export failed'}), 500


if __name__ == '__main__':
    print("Starting Safety Detection System...")
    if detector.start_camera():
        print("Camera initialized successfully")
        print("Open your browser and go to: http://localhost:5000")
        try:
            app.run(debug=False, host='0.0.0.0', port=5000)
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            detector.stop_camera()
    else:
        print("Failed to initialize camera")

