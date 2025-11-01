"""
Main Running File
Start the Safety Detection System
"""

import os
import sys
import webbrowser
import time
import threading

def check_dependencies():
    """Check if required packages are installed"""
    print("Checking dependencies...")
    
    required = ['tensorflow', 'opencv-python', 'flask', 'numpy']
    missing = []
    
    for package in required:
        try:
            __import__(package.replace('-', '_'))
            print(f"OK: {package}")
        except ImportError:
            missing.append(package)
            print(f"MISSING: {package}")
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Installing missing packages...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing)
            print("Dependencies installed successfully")
        except:
            print("Failed to install dependencies. Please run: pip install -r requirements.txt")
            return False
    
    return True

def open_browser():
    """Open browser after delay"""
    time.sleep(3)
    webbrowser.open('http://localhost:5000')

def main():
    """Main function"""
    print("=" * 50)
    print("Safety Detection System")
    print("=" * 50)
    
    if not check_dependencies():
        return
    
    print("\nStarting system...")
    print("Web interface will open automatically")
    print("Manual access: http://localhost:5000")
    print("Press Ctrl+C to stop")
    print("=" * 50)
    
    # Open browser in background
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    try:
        from ui import app, detector
        
        # Try to start camera, but continue even if it fails
        if not detector.camera:
            if detector.start_camera():
                print("Camera initialized successfully")
            else:
                print("Warning: Could not start camera.")
                print("System will still run with mock detection data.")
                print("Make sure camera is connected and not being used by another application.")
        
        print("\nSystem is ready!")
        print("Login credentials are shown above and on the login page.")
        app.run(debug=False, host='0.0.0.0', port=5000)
        
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        if 'detector' in locals():
            detector.stop_camera()
        print("System stopped")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()

