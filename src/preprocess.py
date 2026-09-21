import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import joblib


# ==========================================
# 1. Load datasets
# ==========================================

ground_truth = pd.read_csv(
    "Data/sensor_dataset_with_ground_truth.csv"
)

raw_data = pd.read_csv(
    "Data/raw_sensor_data.csv"
)


# ==========================================
# 2. Sort data by device and timestamp
# ==========================================

ground_truth = ground_truth.sort_values(
    ["device_id", "timestamp"]
).copy()

raw_data = raw_data.sort_values(
    ["device_id", "timestamp"]
).copy()


# ==========================================
# 3. Calculate PM2.5 change
# ==========================================

ground_truth["pm25_change"] = (
    ground_truth.groupby("device_id")["raw_pm2_5"].diff()
)

raw_data["pm25_change"] = (
    raw_data.groupby("device_id")["raw_pm2_5"].diff()
)


# ==========================================
# 4. Identify missing PM2.5 readings
# ==========================================

missing_pm25 = ground_truth[
    ground_truth["raw_pm2_5"].isnull()
].copy()

print("\nMissing PM2.5 Records:")
print(missing_pm25.shape)


# ==========================================
# 5. Select normal data for training
# ==========================================

normal_data = ground_truth[
    ground_truth["fault_type"] == "none"
].copy()


# ==========================================
# 6. Remove rows where PM2.5 change
#    cannot be calculated
# ==========================================

normal_train = normal_data.dropna(
    subset=["pm25_change"]
).copy()


print("\nNormal Training Data:")
print(normal_train.shape)


# ==========================================
# 7. Select ML features
# ==========================================

features = [
    "raw_pm2_5",
    "temperature",
    "humidity",
    "pm25_change"
]

X_train = normal_train[features]


# ==========================================
# 8. Scale features
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)


# ==========================================
# 9. Train Isolation Forest
# ==========================================

model = IsolationForest(
    n_estimators=100,
    contamination="auto",
    random_state=42
)

model.fit(X_train_scaled)

print("\nIsolation Forest trained successfully!")

print("Training samples:", len(X_train))
print("Features:", features)


# ==========================================
# 10. Test on valid ground-truth data
# ==========================================

test_data = ground_truth[
    ground_truth["raw_pm2_5"].notnull()
].dropna(
    subset=["pm25_change"]
).copy()

X_test = test_data[features]

X_test_scaled = scaler.transform(X_test)


# ==========================================
# 11. Predict anomalies
# ==========================================

predictions = model.predict(X_test_scaled)

test_data["prediction"] = predictions

test_data["predicted_status"] = test_data[
    "prediction"
].map({
    1: "normal",
    -1: "anomaly"
})


print("\nPrediction Distribution:")
print(
    test_data["predicted_status"].value_counts()
)


# ==========================================
# 12. Compare with actual fault types
# ==========================================

print("\nResults by Fault Type:")

print(
    pd.crosstab(
        test_data["fault_type"],
        test_data["predicted_status"]
    )
)


# ==========================================
# 13. Missing-data detection
# ==========================================

print(
    "\nMissing-data records:",
    len(missing_pm25)
)


# ==========================================
# 14. Save model and scaler
# ==========================================

joblib.dump(
    model,
    "models/isolation_forest.pkl"
)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

print("\nModel and scaler saved successfully!")