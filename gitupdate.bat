@echo off
echo Updating GitHub Repository
echo ===========================
echo.
echo This script will:
echo 1. Remove old files from Git
echo 2. Add all new files
echo 3. Commit changes
echo 4. Push to GitHub
echo.
echo Make sure you have:
echo - Git installed
echo - Repository initialized
echo - GitHub credentials set up
echo.
pause

echo.
echo Step 1: Removing old files...
git rm camera_detection.py 2>nul
git rm run_safety_detection.py 2>nul
git rm detect.py 2>nul
git rm demo_with_images.py 2>nul
git rm main.py 2>nul
git rm models.py 2>nul
git rm enhanced_models.py 2>nul
git rm train.py 2>nul
git rm train_enhanced_model.py 2>nul
git rm simple_data_storage.py 2>nul
git rm README_Enhanced.md 2>nul
git rm GITHUB_SETUP.md 2>nul
git rm ENHANCEMENT_SUGGESTIONS.md 2>nul
git rm IMPLEMENTATION_PRIORITY.md 2>nul
git rm QUICK_START_GUIDE.md 2>nul
git rm SIMPLE_ENHANCEMENTS_SUMMARY.md 2>nul
git rm STUDENT_FRIENDLY_ENHANCEMENTS.md 2>nul
git rm start_safety_detection.bat 2>nul
git rm -r templates/camera_interface.html 2>nul
git rm templates/detection_result.html 2>nul
git rm templates/image_demo.html 2>nul

echo.
echo Step 2: Adding all new files...
git add storage.py
git add detector.py
git add ui.py
git add admin.py
git add run.py
git add start.bat
git add requirements.txt
git add templates/login.html
git add templates/index.html
git add ProjectIdeology.txt
git add TrailBasedProject.txt
git add SimpleFunctionalProject.txt
git add FinalProject.txt
git add developer.txt

echo.
echo Step 3: Committing changes...
git commit -m "Project restructure: Simplified file structure, added new features, updated documentation"

echo.
echo Step 4: Pushing to GitHub...
git push origin main

echo.
echo Done! Check your GitHub repository.
pause

