@echo off
echo ==========================================
echo    EV Price Predictor - Startup Script
echo ==========================================

:: 1. Create a Virtual Environment if it doesn't exist
if not exist "venv" (
    echo [1/3] Creating Virtual Environment...
    python -m venv venv
)

:: 2. Activate and Install Requirements
echo [2/3] Installing/Updating Libraries...
call venv\Scripts\activate
pip install -r requirements.txt --quiet

:: 3. Launch the Web Server
echo [3/3] Starting Flask Server...
echo ------------------------------------------
echo Dashboard will be live at: http://127.0.0.1:5000
echo ------------------------------------------
python app.py

pause