import pandas as pd
from pathlib import Path

# ==========================================
# 1. Locate project directory
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

raw_file = BASE_DIR / "Data" / "raw_sensor_data.csv"


# ==========================================
# 2. Load raw sensor data
# ==========================================

raw_data = pd.read_csv(raw_file)

print("\nRaw data loaded successfully!")
print("Shape:", raw_data.shape)


# ==========================================
# 3. Create pm25_change
# ==========================================

raw_data["pm25_change"] = (
    raw_data.groupby("device_id")["raw_pm2_5"].diff()
)


# ==========================================
# 4. Check the result
# ==========================================

print("\nColumns:")
print(raw_data.columns)

print("\nMissing values:")
print(raw_data.isnull().sum())

print("\nFirst 5 rows:")
print(raw_data.head())