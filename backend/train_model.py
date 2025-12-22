import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
from pathlib import Path

# Determine base directories and configurable paths via environment variables
BASE_DIR = Path(__file__).resolve().parent
# Default data path: repo-root/data/RF_shuffled_data.xlsx
DEFAULT_DATA_PATH = BASE_DIR.parent / "data" / "RF_shuffled_data.xlsx"
DATA_PATH = Path(os.environ.get("DATA_PATH", str(DEFAULT_DATA_PATH)))
# Default model output path: backend/rf_model.joblib
DEFAULT_MODEL_PATH = BASE_DIR / "rf_model.joblib"
MODEL_PATH = Path(os.environ.get("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
# Default metrics path: backend/metrics.json
DEFAULT_METRICS_PATH = BASE_DIR / "metrics.json"
METRICS_PATH = Path(os.environ.get("METRICS_PATH", str(DEFAULT_METRICS_PATH)))

print(f"Loading data from {DATA_PATH}...")
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Data file not found at {DATA_PATH}. Please set DATA_PATH environment variable or place the dataset at the default location: {DEFAULT_DATA_PATH}")

# 1. Load Data
df = pd.read_excel(DATA_PATH)

# 2. Define Features and Target
# Features identified: ['Model year', 'Engine size (L)', 'Cylinders', 'Fuel type', 'Transmission', 'Vehicle class']
# Target: 'CO2 emissions (g/km)'
features = ['Model year', 'Engine size (L)', 'Cylinders', 'Fuel type', 'Transmission', 'Vehicle class']
target = 'CO2 emissions (g/km)'

X = df[features]
y = df[target]

print("Features:", features)
print("Target:", target)

# 3. Preprocessing Pipeline
# 'Fuel type', 'Transmission', 'Vehicle class' are categorical -> OneHotEncoder
# 'Model year', 'Engine size (L)', 'Cylinders' are numeric -> Passthrough
categorical_features = ['Fuel type', 'Transmission', 'Vehicle class']
numeric_features = ['Model year', 'Engine size (L)', 'Cylinders']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# 4. Create Model Pipeline
# Using RandomForestRegressor as requested
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# 5. Split Data (70% Train, 30% Test)
print("Splitting data 70/30...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 6. Train Model
print("Training model...")
model_pipeline.fit(X_train, y_train)

# 7. Evaluate
print("Evaluating model...")
y_pred = model_pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R2 Score: {r2:.4f}")

# 8. Save Model
# Ensure output directory exists
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model_pipeline, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# 9. Save Metrics (Optional, for frontend/backend usage)
metrics = {
    "mae": round(mae, 2),
    "r2": round(r2, 4)
}
METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f)
print(f"Metrics saved to {METRICS_PATH}")
