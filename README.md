# 🌌 Exoplanet Playground

Explore NASA exoplanet missions, build machine-learning models, and visualize predictions from a single interactive dashboard.

## ✨ Features
- **Interactive workflow** – walk through dataset selection, model training, and prediction inside the web UI.
- **Multiple NASA datasets** – start quickly with curated Kepler, K2, and TESS CSVs stored in `data/`, or upload your own.
- **Configurable models** – try classic ML algorithms (linear regression, SVM, decision tree, random forest, XGBoost, PCA, neural networks) with tunable hyperparameters.
- **Real-time feedback** – track training progress and receive validation through WebSocket updates and flash notifications.
- **Reusable templates** – customize the look and feel of the app through Jinja templates in `templates/`.

## 🚀 Getting Started
### Prerequisites
- Python 3.9+
- (Optional) Node.js if you plan to rebuild the frontend assets in `frontend/`

### 1. Clone the repository
```bash
git clone https://github.com/your-org/Exoplanet-Playground.git
cd NASA-Hackathon-me-test
```

### 2. Create and activate a virtual environment
<details>
<summary>Windows (PowerShell)</summary>

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
</details>

<details>
<summary>macOS / Linux</summary>

```bash
python3 -m venv .venv
source .venv/bin/activate
```
</details>

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Flask app
```bash
python app.py
```
Then open your browser to [`http://localhost:5000`](http://localhost:5000).

## 🗂 Project Structure
```
.
├── app.py               # Flask + Socket.IO backend
├── data/                # Sample NASA exoplanet datasets (CSV)
├── templates/           # Jinja templates for the multi-step UI
├── frontend/            # Frontend assets (Node-managed)
├── visualization/       # Data exploration and visualization utilities
├── requirements.txt     # Python dependencies
└── README.md
```

## 📦 Sample Data
The `data/` folder ships with raw CSVs from the Kepler, K2, and TESS missions. You can upload custom CSVs through the app for bespoke experiments.

## 🧪 Development Tips
- Update or add templates in `templates/` to adjust the UI flow.
- Long-running training jobs are tracked in memory—consider adding persistence or queueing for production use.
- Socket.IO is initialized in `app.py`; expand events there to broadcast richer analytics.

## 📜 License
This project is released under the [MIT License](LICENSE).

## 🙌 Acknowledgements
Built for the NASA Space Apps Hackathon community. Inspired by the scientists, engineers, and dreamers searching the cosmos for new worlds.
