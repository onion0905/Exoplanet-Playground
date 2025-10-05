# Windows Environment Setup Guide

This guide shows you how to set up environment variables for the Exoplanet ML Backend on Windows.

## Quick Start

### Option 1: PowerShell (Recommended)

1. **Copy the template:**
   ```powershell
   Copy-Item env.ps1 env-local.ps1
   ```

2. **Edit your local file:**
   ```powershell
   notepad env-local.ps1
   ```

3. **Load environment variables:**
   ```powershell
   . .\env-local.ps1
   ```

4. **Run the application:**
   ```powershell
   python app.py
   ```

### Option 2: Command Prompt/Batch

1. **Copy the template:**
   ```cmd
   copy env.bat env-local.bat
   ```

2. **Edit your local file:**
   ```cmd
   notepad env-local.bat
   ```

3. **Load environment variables:**
   ```cmd
   call env-local.bat
   ```

4. **Run the application:**
   ```cmd
   python app.py
   ```

### Option 3: Traditional .env (with python-dotenv)

1. **Copy the template:**
   ```powershell
   Copy-Item .env.example .env
   ```

2. **Edit the file:**
   ```powershell
   notepad .env
   ```

3. **Install python-dotenv:**
   ```powershell
   pip install python-dotenv
   ```

4. **The application will automatically load .env**

## Complete Workflow Example

### PowerShell
```powershell
# Navigate to backend directory
cd backend

# Set up environment
Copy-Item env.ps1 env-local.ps1
# Edit env-local.ps1 with your preferred editor

# Load variables and run
. .\env-local.ps1
python app.py
```

### Command Prompt
```cmd
# Navigate to backend directory
cd backend

# Set up environment
copy env.bat env-local.bat
# Edit env-local.bat with your preferred editor

# Load variables and run
call env-local.bat
python app.py
```

## Verification

After loading environment variables, you can verify they're set:

### PowerShell
```powershell
echo $env:FLASK_ENV
echo $env:SECRET_KEY
Get-ChildItem env: | Where-Object Name -like "*FLASK*"
```

### Command Prompt
```cmd
echo %FLASK_ENV%
echo %SECRET_KEY%
set | findstr FLASK
```

## Important Notes

- **Never commit `env-local.ps1`, `env-local.bat`, or `.env`** to version control
- These files contain sensitive configuration
- Always use local copies for your specific environment
- The `.gitignore` file is already configured to ignore these files

## Environment Variables Explained

| Variable | Description | Default |
|----------|-------------|---------|
| `SECRET_KEY` | Flask session encryption key | Change in production! |
| `FLASK_ENV` | Flask environment mode | development |
| `FLASK_DEBUG` | Enable debug mode | True |
| `MAX_UPLOAD_SIZE` | Max file upload size (bytes) | 16777216 (16MB) |
| `SESSION_TIMEOUT` | Training session timeout | 3600 (1 hour) |
| `MAX_CONCURRENT_SESSIONS` | Max concurrent training sessions | 10 |
| `LOG_LEVEL` | Logging level | INFO |
| `LOG_FILE` | Log file path | logs/app.log |
| `CORS_ORIGINS` | Allowed frontend origins | localhost:3000,5173 |

## Troubleshooting

### PowerShell Execution Policy
If you get an execution policy error:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Variables Not Persisting
Environment variables set this way only last for the current terminal session. This is by design for security. If you need permanent variables, use Windows Environment Variables through System Properties.

### Python Path Issues
The scripts automatically set `PYTHONPATH` to include the current and parent directories, ensuring the ML module can be imported properly.