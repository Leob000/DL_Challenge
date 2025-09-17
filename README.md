# Electricity Demand Forecasting for France (30‑min, 2022) - Master 1 project

TL;DR: End‑to‑end pipeline to predict electricity consumption every 30 minutes for France, its 12 regions, and 12 metropolitan areas for calendar year 2022. We merge historical load with weather, build time/calendar features, and train both a MLP (scikit‑learn) and an LSTM (PyTorch). The repository outputs a ready‑to‑submit CSV matching the challenge format and saves trained models.

## Why?

Accurate short‑term load forecasting is essential for grid reliability and energy planning. This codebase shows how to combine meteorology + calendar effects + simple deep/tabular models to produce strong baselines that are reproducible and transparent.

## Quick start

> Requires Python 3.12. Install dependencies and run the four scripts in order.

```bash
# 1) Create and activate a virtual environment (recommended)
python3.12 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Put the challenge data in ./data (see “Data layout” below)

# 4) Run the pipeline
python explo_geo.py       # cleans and extends electricity load time series
python explo_meteo.py     # selects/aggregates weather, resamples to 30 min
python feature_eng.py     # joins geo+weather and builds features
python model_MLP.py       # trains and exports predictions (or use model_LSTM.py)
```

Main output: `data/pred.csv` (challenge submission format)

Models saved to: `models/` (timestamped `.joblib` for MLP)

## Data layout (expected files)

Place inputs under `./data/`:

* `train.csv` — historical electricity load (France, 12 regions, 12 metros), 30‑min frequency.
* `pred_template.csv` — 2022 30‑min timestamp template used to extend the index.
* `meteo.parquet` — raw weather observations (≈3‑hourly) from \~40 stations, 2017–2022.
* `villes_loc.csv` — latitude/longitude for target metropolitan areas.

The pipeline produces intermediate artifacts:

* `geo_tweaked.parquet` — cleaned/extended load series.
* `meteo_tweaked.parquet` — curated & resampled weather (France/regions/cities).
* `clean_data.parquet` — final modeling table (joined + engineered features).

> Directories: Keep `data/` for datasets and `models/` for saved models. Both are created/used by the scripts.

## What the pipeline does

### 1) Load cleaning (`explo_geo.py`)

* Uses Europe/Paris timezone consistently; checks index integrity.
* Handles DST fallback hour by dropping duplicated local times (the October clock change).
* Fixes or masks obvious outliers (e.g., Nice known shift, Nancy pre‑2020 anomalies) and minor gaps via time interpolation.
* Extends the index through 2022‑12‑31 23:30 to prepare for forecasting.

### 2) Weather curation (`explo_meteo.py`)

* Selects essential variables: wind speed (`ff`), temperature (`tc`), humidity (`u`), precipitation last hour (`rr1`), station pressure (`pres`).
* Drops problematic coverage (e.g., Var department) and resamples 3‑hourly → 30‑min via time interpolation.
* Builds France (national), per‑region, and per‑city aggregates.

  * Cities mapped to nearest station; Grenoble uses the mean of the 2–3 closest.
* Exports a tidy long format with a `zone` column.

### 3) Feature engineering (`feature_eng.py`)

* Temperature dynamics: EWM smoothed temp (`tc_ewm15`, `tc_ewm06`) + rolling 24h min/max.
* Calendar & cyclic time: sin/cos of minute‑of‑day, day‑of‑week, day‑of‑year; weekend flag; French public holidays; winter/summer time flag.
* Optional drops for August seasonality flags, precipitation and pressure (configurable).
* Standardization per zone for continuous features (optional).

### 4) Modeling

#### MLP baseline (`model_MLP.py`)

* One‑hot encodes `zone_*`, standardizes target `Load` per zone, trains a multi‑layer perceptron (`sklearn.neural_network.MLPRegressor`).
* Default architecture: `(100, 75, 50)` with `activation="logistic"`, `adam` solver, and `random_state=42`.
* Produces per‑timestamp predictions for each zone, rescales back to MW, and aggregates into the challenge columns.
* Saves trained model(s) under `models/` and exports `data/pred.csv` in the required format.

#### LSTM prototype (`model_LSTM.py`)

* Sequence‑to‑one LSTM (PyTorch) with a sliding 96‑step context (2 days at 30‑min) and iterative forecasting over 2022.
* Uses the same engineered exogenous features; target `Load` is standardized per zone and unscaled after prediction.
* Runs on CPU by default and auto‑detects Apple MPS/CUDA if available.

> Evaluation: The challenge uses the sum of per‑zone RMSE (zones with higher average load contribute more). The code mirrors this for validation runs.

## Reproduce a submission

1. Ensure inputs are in `./data/` as described above.
2. In `model_MLP.py`, keep `FULL_TRAIN = True` to train on all data up to 2022 and predict the full year 2022.
3. Run `python model_MLP.py` — resulting file: `data/pred.csv`.

> For validation instead, set `FULL_TRAIN = False` (train ≤2020, predict 2021) and the script will report RMSE by summing over zones.

## Notable implementation details

* Timezone & DST: All series are localized to *Europe/Paris*. Duplicate local times during the October fallback are removed before modeling.
* Data quality fixes: Targeted outlier handling (e.g., Nice offset; Nancy pre‑2020) with transparent, script‑based rules.
* Weather resampling: 3‑hourly station data → 30‑min via time interpolation, preserving department/region/city groupings.
* Config flags: Many scripts expose booleans at the top (e.g., `GRAPHS`, `FULL_TRAIN`, `DROP_PRECIPITATIONS`) for quick experimentation.
* Formatting & lint: `ruff` is configured via `pyproject.toml`.

## Repository layout

```
.
├── explo_geo.py          # Load cleaning, outliers, index extension, export geo_tweaked.parquet
├── explo_meteo.py        # Weather selection/aggregation, resampling, export meteo_tweaked.parquet
├── feature_eng.py        # Join geo+meteo, build features, export clean_data.parquet
├── model_MLP.py          # Tabular MLP training/inference, export data/pred.csv
├── model_LSTM.py         # LSTM training/inference (prototype)
├── utils.py              # Mapping helpers for station/city visualization
├── requirements.txt      # Exact versions for Python 3.12
├── pyproject.toml        # Ruff formatting & lint configuration
└── data/                 # Put input files here; outputs & predictions written here
    └── models/           # Trained models saved here (created automatically)
```
