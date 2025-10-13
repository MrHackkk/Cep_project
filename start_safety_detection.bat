@echo off
title Smart AI Safety Kit Detection System
color 0A

echo.
echo ========================================
echo  Smart AI Safety Kit Detection System
echo ========================================
echo.
echo Starting system initialization...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org
    pause
    exit /b 1
)

echo Python found: 
python --version

REM Check if required files exist
if not exist "camera_detection.py" (
    echo ERROR: camera_detection.py not found
    echo Please ensure you're running this from the correct directory
    pause
    exit /b 1
)

echo.
echo Installing/updating dependencies...
pip install -r requirements.txt

echo.
echo Starting Safety Detection System...
echo.
echo The web interface will open automatically in your browser
echo Manual access: http://localhost:5000
echo.
echo Press Ctrl+C to stop the system
echo.

python run_safety_detection.py

echo.
echo System stopped.
pause
