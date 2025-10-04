@echo off
echo 🌌 Starting Exoplanet Playground Development Servers
echo.
echo This will start TWO servers:
echo 🔧 Backend API: http://localhost:5000
echo 🎨 Frontend App: http://localhost:3000
echo.
echo Press any key to start both servers...
pause >nul

echo.
echo 🚀 Starting servers...
echo.

start "Backend API Server" cmd /k "cd /d %~dp0 && .venv\Scripts\activate && python start_backend.py"

timeout /t 3 >nul

start "Frontend Dev Server" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo ✅ Both servers starting!
echo.
echo 🌍 Access the app at: http://localhost:3000
echo 📡 API available at: http://localhost:5000
echo.
echo Press any key to close this window...
pause >nul