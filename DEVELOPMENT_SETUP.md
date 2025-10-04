# 🚀 Development Setup Guide - Separate Frontend & Backend

## Overview

This setup runs **two separate servers** for development:
- **Frontend (React + Vite)**: `http://localhost:3000` 
- **Backend (Flask API)**: `http://localhost:5000`

The frontend makes API calls to the backend, and both servers support hot reload for development.

## 📋 Prerequisites

1. **Node.js & npm** installed
2. **Python 3.8+** installed
3. **Virtual environment** set up (`.venv` folder exists)

## 🔧 Quick Start (Two Terminals)

### Terminal 1: Backend Server
```bash
# Navigate to project
cd "D:\NTU\NASA Hackathon\Exoplanet-Playground"

# Activate virtual environment and start Flask
.\.venv\Scripts\Activate.ps1; python backend/app.py
```

### Terminal 2: Frontend Server  
```bash
# Navigate to frontend
cd "D:\NTU\NASA Hackathon\Exoplanet-Playground\frontend"

# Install dependencies (if needed)
npm install

# Start Vite dev server
npm run dev
```

## 🌐 Access Points

- **🎨 Frontend App**: http://localhost:3000/
- **📡 Backend API**: http://localhost:5000/
- **🔍 API Health**: http://localhost:5000/api/health

## 📂 What Each Server Does

### Backend (Flask - Port 5000)
- ✅ **API Endpoints Only** - No static file serving
- ✅ **CORS Enabled** for frontend (localhost:3000)
- ✅ **JSON Responses** for all routes
- ✅ **Hot Reload** when Python files change

**Endpoints Available:**
- `GET /` - Homepage data (JSON)
- `GET/POST /select` - Model selection API
- `GET/POST /training` - Training management  
- `GET/POST /predict` - Prediction API
- `GET/POST /result` - Results API
- `GET /api/*` - Various utility APIs

### Frontend (React + Vite - Port 3000)
- ✅ **React Development Server** with hot reload  
- ✅ **API Integration** - calls backend at localhost:5000
- ✅ **Client-side Routing** - all routes handled by React Router
- ✅ **Fast Refresh** when components change

**Routes Available:**
- `/` - Homepage (fetches data from backend API)
- `/select` - Model selection page
- `/training` - Training progress page
- `/predict` - Prediction interface  
- `/result` - Results visualization
- `/learn` - Learning resources

## 🔄 Communication Flow

```
Browser (localhost:3000)
    ↓ User visits /
React App (HomePage)  
    ↓ useEffect calls apiService.getHomeData()
API Service (api.js)
    ↓ fetch('http://localhost:5000/')
Flask Backend 
    ↓ returns JSON data
React displays data
```

## 🛠️ Development Workflow

### Making Changes

**Frontend Changes:**
1. Edit files in `frontend/src/`
2. Vite automatically reloads the browser
3. No build step needed in development

**Backend Changes:**
1. Edit files in `backend/` or `ML/`
2. Flask automatically restarts (debug mode)
3. API changes are immediately available

### Testing API Integration

1. **Check Backend**: Visit http://localhost:5000/api/health
2. **Check Frontend**: Visit http://localhost:3000/ (should load data from backend)
3. **Check CORS**: Network tab should show successful API calls (no CORS errors)

## 📁 Project Structure

```
Exoplanet-Playground/
├── backend/
│   ├── app.py              # ✅ Flask API server (port 5000)
│   └── config.py          
├── frontend/
│   ├── src/
│   │   ├── lib/
│   │   │   └── api.js     # ✅ API service for backend calls
│   │   └── pages/
│   │       └── home/
│   │           └── HomePage.jsx  # ✅ Updated to fetch from API
│   ├── package.json       
│   └── vite.config.js     # ✅ Dev server config (port 3000)
└── ML/                    # ML modules used by backend
```

## 🔧 Configuration Details

### CORS Configuration (Backend)
```python
# In backend/app.py
CORS(app, origins=['http://localhost:3000', 'http://127.0.0.1:3000'])
```

### API Base URL (Frontend)  
```javascript
// In frontend/src/lib/api.js
const API_BASE_URL = 'http://localhost:5000';
```

### Vite Dev Server (Frontend)
```javascript
// In frontend/vite.config.js
export default defineConfig({
  plugins: [react()],
  server: {
    host: 'localhost',
    port: 3000,
  },
})
```

## 🐛 Troubleshooting

### "Failed to connect to backend"
- ✅ Check Flask server is running on port 5000
- ✅ Check CORS configuration allows localhost:3000
- ✅ Check no firewall blocking ports 3000/5000

### "CORS Error"  
- ✅ Verify Flask has `CORS(app, origins=['http://localhost:3000'])`
- ✅ Check API calls use correct base URL (localhost:5000)
- ✅ Ensure both servers are running

### "Module not found" (Backend)
- ✅ Activate virtual environment: `.\.venv\Scripts\Activate.ps1`
- ✅ Install requirements: `pip install -r requirements.txt`

### "Package not found" (Frontend)
- ✅ Install dependencies: `cd frontend && npm install`
- ✅ Check node version compatibility

## 📋 Startup Checklist

- [ ] Virtual environment activated
- [ ] Backend server running on port 5000 
- [ ] Frontend server running on port 3000
- [ ] Both servers show no error messages
- [ ] http://localhost:3000/ loads successfully
- [ ] Homepage displays data from backend API
- [ ] Network tab shows successful API calls

## 🎯 Production Deployment

For production, you would:
1. Build frontend: `npm run build` 
2. Serve built files with a web server (nginx, etc.)
3. Run Flask with production WSGI server (gunicorn)
4. Update CORS to allow production domain
5. Use environment variables for API URLs

---

**Status**: ✅ **Development Setup Complete**  
**Frontend**: 🟢 http://localhost:3000/ (React + Vite)  
**Backend**: 🟢 http://localhost:5000/ (Flask API)  
**Communication**: ✅ API calls working with CORS