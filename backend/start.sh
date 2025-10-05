#!/bin/bash
# Quick start script for Exoplanet Playground Backend

echo "========================================"
echo "Exoplanet Playground Backend"
echo "========================================"
echo ""

echo "Checking Python installation..."
python3 --version
if [ $? -ne 0 ]; then
    echo "ERROR: Python is not installed or not in PATH"
    exit 1
fi
echo ""

echo "Installing dependencies..."
pip install -r ../requirements.txt
pip install -r requirements.txt
echo ""

echo "Starting Flask server..."
python3 app.py
