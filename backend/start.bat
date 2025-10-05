@echo off
REM Quick start script for Exoplanet Playground Backend

echo ========================================
echo Exoplanet Playground Backend
echo ========================================
echo.

echo Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)
echo.

echo Installing dependencies...
pip install -r ../requirements.txt
pip install -r requirements.txt
echo.

echo Starting Flask server...
python app.py

pause
