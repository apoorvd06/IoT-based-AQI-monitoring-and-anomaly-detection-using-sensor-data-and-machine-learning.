import pandas as pd
import joblib
from pathlib import Path


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

raw_file = BASE_DIR / "Data" / "raw_sensor_data.csv"
model_file = BASE_DIR / "models" / "isolation_forest.pkl"
scaler_file = BASE_DIR / "models" / "scaler.pkl"

output_file = BASE_DIR / "reports" / "raw_data_predictions.csv"


# ============================================================
# 2. LOAD RAW SENSOR DATA
# ============================================================

raw_data = pd.read_csv(raw_file)

print("\nRaw data loaded successfully!")
print("Shape:", raw_data.shape)


# ============================================================
# 3. CREATE PM2.5 CHANGE FEATURE
# ============================================================

raw_data["pm25_change"] = (
    raw_data.groupby("device_id")["raw_pm2_5"].diff()
)

print("\nPM2.5 change feature created successfully!")


# ============================================================
# 4. CHECK RAW DATA
# ============================================================

print("\nColumns:")
print(raw_data.columns)

print("\nMissing values:")
print(raw_data.isnull().sum())

print("\nFirst 5 rows:")
print(raw_data.head())


# ============================================================
# 5. DEFINE ML FEATURES
# ============================================================

features = [
    "raw_pm2_5",
    "temperature",
    "humidity",
    "pm25_change"
]


# ============================================================
# 6. REMOVE ROWS WITH MISSING ML FEATURES
# ============================================================

prediction_data = raw_data.dropna(
    subset=features
).copy()

print("\nData available for prediction:")
print(prediction_data.shape)

print("\nMissing values after cleaning:")
print(prediction_data[features].isnull().sum())


# ============================================================
# 7. LOAD SAVED MODEL AND SCALER
# ============================================================

model = joblib.load(model_file)
scaler = joblib.load(scaler_file)

print("\nModel and scaler loaded successfully!")


# ============================================================
# 8. SELECT ML FEATURES
# ============================================================

X_raw = prediction_data[features]

print("\nFeatures prepared for prediction:")
print(X_raw.head())

print("\nFeature shape:", X_raw.shape)


# ============================================================
# 9. SCALE RAW DATA
# ============================================================

X_raw_scaled = scaler.transform(X_raw)

print("\nRaw data scaled successfully!")
print("Scaled shape:", X_raw_scaled.shape)


# ============================================================
# 10. PREDICT ANOMALIES
# ============================================================

predictions = model.predict(X_raw_scaled)

prediction_data["prediction"] = predictions

prediction_data["predicted_status"] = (
    prediction_data["prediction"]
    .map({
        1: "normal",
        -1: "anomaly"
    })
)

print("\nPrediction completed!")


# ============================================================
# 11. CALCULATE ANOMALY SCORE
# ============================================================

anomaly_scores = model.decision_function(X_raw_scaled)

prediction_data["anomaly_score"] = anomaly_scores


# ============================================================
# 12. NORMAL VS ANOMALY COUNT
# ============================================================

status_counts = prediction_data["predicted_status"].value_counts()

print("\nPrediction summary:")
print(status_counts)

print("\nNumber of normal records:",
      (prediction_data["prediction"] == 1).sum())

print("Number of anomaly records:",
      (prediction_data["prediction"] == -1).sum())


# ============================================================
# 13. EXTRACT ANOMALY RECORDS
# ============================================================

anomalies = prediction_data[
    prediction_data["prediction"] == -1
].copy()

normal = prediction_data[
    prediction_data["prediction"] == 1
].copy()

print("\nNumber of normal records:", len(normal))
print("Number of anomaly records:", len(anomalies))


# ============================================================
# 14. DISPLAY FIRST 10 ANOMALIES
# ============================================================

print("\nFirst 10 anomaly records:")

print(
    anomalies[
        [
            "timestamp",
            "device_id",
            "raw_pm2_5",
            "temperature",
            "humidity",
            "pm25_change",
            "anomaly_score",
            "prediction",
            "predicted_status"
        ]
    ].head(10)
)


# ============================================================
# 15. ANOMALIES BY DEVICE
# ============================================================

anomaly_by_device = (
    anomalies
    .groupby("device_id")
    .size()
    .sort_values(ascending=False)
)

print("\nAnomalies by device:")
print(anomaly_by_device)


# ============================================================
# 16. TOTAL RECORDS BY DEVICE
# ============================================================

device_total = (
    prediction_data
    .groupby("device_id")
    .size()
)

print("\nTotal usable records by device:")
print(device_total)


# ============================================================
# 17. ANOMALY PERCENTAGE BY DEVICE
# ============================================================

device_anomaly_rate = (
    anomaly_by_device
    .div(device_total)
    .mul(100)
    .sort_values(ascending=False)
)

print("\nAnomaly percentage by device:")
print(device_anomaly_rate)


# ============================================================
# 18. CREATE DEVICE SUMMARY TABLE
# ============================================================

device_summary = pd.DataFrame({
    "total_records": device_total,
    "anomaly_records": anomaly_by_device,
    "anomaly_percentage": device_anomaly_rate
})

device_summary["anomaly_records"] = (
    device_summary["anomaly_records"]
    .fillna(0)
    .astype(int)
)

device_summary["anomaly_percentage"] = (
    device_summary["anomaly_percentage"]
    .fillna(0)
)

device_summary = device_summary.sort_values(
    "anomaly_percentage",
    ascending=False
)

print("\nDevice anomaly summary:")
print(device_summary)


# ============================================================
# 19. SAVE PREDICTION RESULTS
# ============================================================

output_file.parent.mkdir(
    parents=True,
    exist_ok=True
)

prediction_data.to_csv(
    output_file,
    index=False
)

print("\nPrediction results saved successfully!")
print("File:", output_file)


# ============================================================
# 20. SAVE DEVICE SUMMARY
# ============================================================

summary_file = (
    BASE_DIR
    / "reports"
    / "device_anomaly_summary.csv"
)

device_summary.to_csv(
    summary_file
)

print("\nDevice anomaly summary saved successfully!")
print("File:", summary_file)


# ============================================================
# 21. FINAL MESSAGE
# ============================================================

print("\n============================================")
print("RAW DATA ANOMALY DETECTION COMPLETED")
print("============================================")