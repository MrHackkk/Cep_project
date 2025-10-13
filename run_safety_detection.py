#!/usr/bin/env python3
"""
Smart AI Safety Kit Detection System Launcher
Easy-to-use launcher for the construction site safety monitoring system
"""

import os
import sys
import subprocess
import webbrowser
import time
import threading
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    print(" Checking dependencies...")
    
    required_packages = [
        'tensorflow',
        'opencv-python',
        'flask',
        'numpy',
        'PIL'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
            print("✅ Dependencies installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies. Please run: pip install -r requirements.txt")
            return False
    
    return True

def check_camera():
    """Check if camera is available"""
    print("\n Checking camera availability...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print("✅ Camera is working properly")
                return True
            else:
                print("❌ Camera is not responding")
                return False
        else:
            print("❌ Could not access camera")
            return False
    except Exception as e:
        print(f"❌ Camera error: {e}")
        return False

def check_model():
    """Check if trained model exists"""
    print("\n Checking AI model...")
    
    model_paths = [
        "ppe_model/best_model.h5",
        "ppe_model/final_model.h5",
        "enhanced_ppe_model/best_model.h5",
        "enhanced_ppe_model/final_model.h5"
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            print(f"✅ Found model: {path}")
            return True
    
    print("⚠️ No trained model found. System will use mock predictions.")
    print(" To train a model, run: python train_enhanced_model.py")
    return False

def open_browser():
    """Open browser after a short delay"""
    time.sleep(3)
    webbrowser.open('http://localhost:5000')

def main():
    """Main launcher function"""
    print(" Smart AI Safety Kit Detection System")
    print("=" * 50)
    print("Initializing system...")
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Dependency check failed. Please install requirements manually.")
        return
    
    # Check camera
    if not check_camera():
        print("\n⚠️ Camera issues detected. Please check your camera connection.")
        print("The system will still start but camera functionality may be limited.")
    
    # Check model
    model_available = check_model()
    
    print("\n Starting Safety Detection System...")
    print("=" * 50)
    
    if model_available:
        print("✅ AI model loaded - High accuracy detection enabled")
    else:
        print("⚠️ Using mock predictions - Train a model for better accuracy")
    
    print("\n Web interface will open automatically...")
    print(" Manual access: http://localhost:5000")
    print(" Press Ctrl+C to stop the system")
    print("=" * 50)
    
    # Open browser in background
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    try:
        # Import and run the camera detection system
        from camera_detection import app, detector
        
        # Start the Flask app
        app.run(debug=False, host='0.0.0.0', port=5000)
        
    except KeyboardInterrupt:
        print("\n\n Shutting down Safety Detection System...")
        if 'detector' in locals():
            detector.stop_camera()
        print("✅ System stopped successfully")
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("Please ensure all files are in the correct location")
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check the error and try again")

if __name__ == "__main__":
    main()
