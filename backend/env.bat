@echo off
REM Windows Batch Environment Configuration for Exoplanet ML Backend
REM Usage: call env.bat

REM Core Flask Configuration
set SECRET_KEY=exoplanet-ml-secret-key-dev-change-this-in-production
set FLASK_ENV=development
set FLASK_DEBUG=True

REM ML Configuration
set MAX_UPLOAD_SIZE=16777216
set SESSION_TIMEOUT=3600
set MAX_CONCURRENT_SESSIONS=10

REM Logging Configuration
set LOG_LEVEL=INFO
set LOG_FILE=logs/app.log

REM CORS Origins
set CORS_ORIGINS=http://localhost:3000,http://localhost:5173

REM Python Path
set PYTHONPATH=%CD%;%CD%\..

echo Environment variables loaded for Exoplanet ML Backend
echo Flask Environment: %FLASK_ENV%
echo Debug Mode: %FLASK_DEBUG%
echo CORS Origins: %CORS_ORIGINS%