# 🌌 Exoplanet Playground - Quick Start Guide

## What We've Built

Successfully integrated the React frontend with the Flask backend for the `/` endpoint! 🚀

## ✅ Completed Integration

### Frontend Integration
- ✅ React app builds to `frontend/dist/`
- ✅ Flask serves static files from dist folder
- ✅ Client-side routing works for all React routes
- ✅ Fallback handling if frontend not built
- ✅ API endpoints still available under `/api/`

### Backend Features
- ✅ Serves React app at `/` (all routes)
- ✅ Smart routing: static files + SPA routing
- ✅ Frontend build status detection
- ✅ Helpful build instructions when needed
- ✅ JSON API responses when requested

## 🚀 How to Run

### Method 1: Quick Start
```bash
# Navigate to project
cd "D:\NTU\NASA Hackathon\Exoplanet-Playground"

# Activate venv, build frontend, and start server
.\.venv\Scripts\Activate.ps1; cd frontend; npm run build; cd ..; python run.py
```

### Method 2: Using Build Script
```bash
# Use the automated build script
python build_frontend.py
```

### Method 3: Step by Step
```bash
# 1. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 2. Build frontend (if needed)
cd frontend
npm install
npm run build
cd ..

# 3. Start backend server
python run.py
```

## 📱 Access Points

- **🌐 Main App**: http://localhost:5000/
- **🏠 Homepage**: http://localhost:5000/ (React HomePage)
- **🎯 Select Page**: http://localhost:5000/select
- **📊 Training**: http://localhost:5000/training  
- **🔍 Predict**: http://localhost:5000/predict
- **📈 Results**: http://localhost:5000/result
- **📚 Learn**: http://localhost:5000/learn

## 🔧 API Endpoints

- **📋 Health Check**: http://localhost:5000/api/health
- **🎨 Frontend Status**: http://localhost:5000/api/frontend-status
- **📊 Datasets Info**: http://localhost:5000/api/datasets
- **🤖 Models Info**: http://localhost:5000/api/models

## 🎯 Integration Details

### What Happens at `/`
1. **React Routes**: All frontend routes (`/`, `/select`, `/training`, etc.) serve the React app
2. **Static Files**: CSS, JS, images served from `frontend/dist/`
3. **API Requests**: Requests with `Accept: application/json` return API data
4. **Fallback**: Shows build instructions if frontend not built

### Key Files Modified
- `backend/app.py` - Updated `/` route to serve React app
- `run.py` - Enhanced startup script with frontend status
- `build_frontend.py` - Automated build and start script

### Directory Structure
```
Exoplanet-Playground/
├── backend/
│   ├── app.py          # ✅ Updated Flask app
│   └── config.py       # Configuration
├── frontend/
│   ├── dist/           # ✅ Built React app
│   ├── src/            # React source
│   └── package.json    # Dependencies
├── run.py              # ✅ Enhanced startup
└── build_frontend.py   # ✅ Build automation
```

## 🎉 Success Indicators

When everything works correctly, you should see:

1. **Terminal Output**:
   ```
   🌌 Exoplanet Discovery Playground Backend
   ✅ React frontend built and ready
   🌐 Access full app at: http://localhost:5000/
   📡 Available endpoints: [list of endpoints]
   ```

2. **Browser**: React app loads at http://localhost:5000/
3. **API**: JSON responses at `/api/*` endpoints
4. **Routing**: All React routes work (no 404s)

## 🔄 Development Workflow

1. **Frontend Changes**: 
   ```bash
   cd frontend
   npm run build
   # Server auto-reloads
   ```

2. **Backend Changes**: Flask auto-reloads in debug mode

3. **Full Restart**:
   ```bash
   # Ctrl+C to stop server
   python run.py  # Restart
   ```

## 🎯 Next Steps

The `/` endpoint is fully integrated! Ready for:
- ✅ Frontend serves at `/`
- ✅ All React routes work
- ✅ API endpoints available
- 🔄 Ready to implement other endpoints (`/select`, `/training`, etc.)

---

**Status**: ✅ `/` Endpoint Integration Complete!  
**Server**: 🟢 Running at http://localhost:5000/  
**Frontend**: ✅ React App Integrated  
**API**: ✅ Available under `/api/`