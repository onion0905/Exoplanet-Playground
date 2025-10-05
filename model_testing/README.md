# Random Forest Exoplanet Classifier

## Overview
Clean Random Forest trainer for NASA exoplanet datasets (Kepler, K2, TESS). Automatically filters out identification, archive information, and disposition columns based on NASA Exoplanet Archive API documentation.

## Features
- **Auto-detection**: Automatically detects dataset type (Kepler/K2/TESS)
- **Smart filtering**: Removes inappropriate columns based on NASA API docs
- **Label mapping**: Maps TESS labels (PC, CP, FP, etc.) to standard labels
- **Complete preprocessing**: Handles missing values and categorical encoding
- **Comprehensive results**: Confusion matrix, feature importance, confidence stats

## Usage

### Kepler Dataset
```bash
python model_testing/train_rf_clean.py --data kepler_raw.csv --n-estimators 100 --max-depth 10 --scale
```

### K2 Dataset  
```bash
python model_testing/train_rf_clean.py --data k2_raw.csv --n-estimators 50 --max-depth 8 --test-size 0.25
```

### TESS Dataset
```bash
python model_testing/train_rf_clean.py --data tess_raw.csv --n-estimators 75 --max-depth 12 --test-size 0.3 --scale
```

## Parameters
- `--data`: Dataset filename (required)
- `--n-estimators`: Number of trees (default: 100)
- `--max-depth`: Maximum tree depth (default: None)
- `--min-samples-split`: Min samples to split (default: 2)  
- `--min-samples-leaf`: Min samples in leaf (default: 1)
- `--test-size`: Test split ratio (default: 0.2)
- `--random-state`: Random seed (default: 42)
- `--scale`: Apply StandardScaler to features

## Output Classes
The model predicts one of three classes:
- **CONFIRMED**: Confirmed exoplanet
- **CANDIDATE**: Planet candidate (needs follow-up)
- **FALSE POSITIVE**: Not a planet (stellar activity, etc.)

## Filtered Columns
The script automatically removes columns that should not be used as features:

### Kepler Dataset
- Identification: `kepid`, `kepoi_name`, `kepler_name`
- Disposition: `koi_disposition`, `koi_score`, `koi_fpflag_*`
- Metadata: `rowid`, `koi_quarters`, `koi_datalink_*`

### K2 Dataset  
- Identification: `epic_candname`, `epic_hostname`, `k2_name`
- Disposition: `disposition`, `score`, `fpflag_*` 
- Metadata: `rowid`, `quarters`, `datalink_*`

### TESS Dataset
- Identification: `tid`, `toi`, `tic_*`, `toipfx`
- Disposition: `tfopwg_disp` (used as target), `score`, `fpflag_*`
- Metadata: `rowid`, `datalink_*`

## Results Summary
Recent test results on sample datasets:

| Dataset | Features | Accuracy | F1-Score (Macro) |
|---------|----------|----------|------------------|
| Kepler  | 122      | 84.3%    | 80.0%           |
| K2      | 290      | 97.1%    | 91.0%           |  
| TESS    | 80       | 75.5%    | 60.0%           |

The script provides detailed output including:
- Confusion matrix for all 3 classes
- Classification report with precision/recall
- Top 10 most important features
- Prediction confidence statistics
- Low confidence prediction analysis