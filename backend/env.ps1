# PowerShell Environment Configuration for Exoplanet ML Backend
# Usage: . .\env.ps1  (dot-source this file to load variables)

# Core Flask Configuration
$env:SECRET_KEY = "exoplanet-ml-secret-key-dev-change-this-in-production"
$env:FLASK_ENV = "development"
$env:FLASK_DEBUG = "True"

# Database settings (if needed in future)
# $env:DATABASE_URL = "sqlite:///exoplanet.db"

# ML Configuration
$env:MAX_UPLOAD_SIZE = "16777216"        # 16MB in bytes
$env:SESSION_TIMEOUT = "3600"           # 1 hour in seconds  
$env:MAX_CONCURRENT_SESSIONS = "10"

# Logging Configuration
$env:LOG_LEVEL = "INFO"
$env:LOG_FILE = "logs/app.log"

# CORS Origins (comma-separated)
$env:CORS_ORIGINS = "http://localhost:3000,http://localhost:5173"

# Additional PowerShell-specific settings
$env:PYTHONPATH = "$PWD;$PWD\.."        # Add current and parent directory to Python path

Write-Host "Environment variables loaded for Exoplanet ML Backend" -ForegroundColor Green
Write-Host "Flask Environment: $env:FLASK_ENV" -ForegroundColor Cyan
Write-Host "Debug Mode: $env:FLASK_DEBUG" -ForegroundColor Cyan
Write-Host "CORS Origins: $env:CORS_ORIGINS" -ForegroundColor Cyan