import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

input_file = (
    BASE_DIR
    / "reports"
    / "sensor_health_results.csv"
)

output_file = (
    BASE_DIR
    / "reports"
    / "final_sensor_diagnosis.csv"
)


# ============================================================
# 2. LOAD SENSOR HEALTH RESULTS
# ============================================================

data = pd.read_csv(input_file)

data["timestamp"] = pd.to_datetime(
    data["timestamp"]
)

data = data.sort_values(
    ["device_id", "timestamp"]
).reset_index(drop=True)


print("\nSensor health results loaded!")
print("Records:", len(data))


# ============================================================
# 3. CALCULATE PM2.5 CHANGE
# ============================================================

data["pm25_change"] = (
    data
    .groupby("device_id")["raw_pm2_5"]
    .diff()
)


# ============================================================
# 4. CALCULATE ROLLING STATISTICS
# ============================================================

data["rolling_mean"] = (
    data
    .groupby("device_id")["raw_pm2_5"]
    .transform(
        lambda x: x.rolling(
            window=6,
            min_periods=1
        ).mean()
    )
)


data["rolling_std"] = (
    data
    .groupby("device_id")["raw_pm2_5"]
    .transform(
        lambda x: x.rolling(
            window=6,
            min_periods=2
        ).std()
    )
)


# ============================================================
# 5. INITIALIZE DIAGNOSIS
# ============================================================

data["diagnosed_fault"] = "normal"


# ============================================================
# 6. MISSING DATA
# ============================================================

missing_mask = (
    data["raw_pm2_5"].isna()
)

data.loc[
    missing_mask,
    "diagnosed_fault"
] = "missing_data"


# ============================================================
# 7. BEHAVIORAL ANOMALIES
# ============================================================

anomaly_mask = (
    data["predicted_fault"] == 1
)


# Only diagnose records that have an ML anomaly
# and are not already missing.
valid_anomaly = (
    anomaly_mask
    & ~missing_mask
)


# ============================================================
# 8. SUDDEN SPIKE
# ============================================================

spike_mask = (
    valid_anomaly
    & data["pm25_change"].abs().gt(20)
)

data.loc[
    spike_mask,
    "diagnosed_fault"
] = "sudden_spike"


# ============================================================
# 9. GRADUAL DRIFT
# ============================================================

drift_mask = (
    valid_anomaly
    & ~spike_mask
    & data["pm25_change"].gt(0)
    & data["rolling_mean"].gt(50)
)

data.loc[
    drift_mask,
    "diagnosed_fault"
] = "gradual_drift"


# ============================================================
# 10. PROGRESSIVE DEGRADATION
# ============================================================

degradation_mask = (
    valid_anomaly
    & ~spike_mask
    & ~drift_mask
    & data["rolling_mean"].gt(55)
)

data.loc[
    degradation_mask,
    "diagnosed_fault"
] = "progressive_degradation"


# ============================================================
# 11. OTHER BEHAVIORAL ANOMALY
# ============================================================

remaining_anomaly = (
    valid_anomaly
    & (
        data["diagnosed_fault"]
        == "normal"
    )
)

data.loc[
    remaining_anomaly,
    "diagnosed_fault"
] = "behavioral_anomaly"


# ============================================================
# 12. FINAL SENSOR STATUS
# ============================================================

data["final_sensor_status"] = np.where(
    data["diagnosed_fault"] == "normal",
    "normal",
    "fault_detected"
)


# ============================================================
# 13. PRINT RESULTS
# ============================================================

print("\n============================================")
print("FINAL SENSOR DIAGNOSIS")
print("============================================")

print(
    data["diagnosed_fault"]
    .value_counts()
)


# ============================================================
# 14. DEVICE-WISE DIAGNOSIS
# ============================================================

print("\n============================================")
print("DEVICE-WISE DIAGNOSIS")
print("============================================")

device_summary = (
    data
    .groupby(
        [
            "device_id",
            "diagnosed_fault"
        ]
    )
    .size()
    .unstack(
        fill_value=0
    )
)

print(device_summary)


# ============================================================
# 15. SAVE FINAL RESULTS
# ============================================================

data.to_csv(
    output_file,
    index=False
)

print("\nFinal diagnosis saved successfully!")

print(
    "File:",
    output_file
)


# ============================================================
# 16. FINAL MESSAGE
# ============================================================

print("\n============================================")
print("FAULT DIAGNOSIS COMPLETED")
print("============================================")