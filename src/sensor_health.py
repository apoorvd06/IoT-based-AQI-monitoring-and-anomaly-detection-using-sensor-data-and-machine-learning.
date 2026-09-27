import pandas as pd
from pathlib import Path


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

raw_file = (
    BASE_DIR
    / "Data"
    / "raw_sensor_data.csv"
)

prediction_file = (
    BASE_DIR
    / "reports"
    / "raw_data_predictions.csv"
)

ground_truth_file = (
    BASE_DIR
    / "Data"
    / "sensor_dataset_with_ground_truth.csv"
)

output_file = (
    BASE_DIR
    / "reports"
    / "sensor_health_results.csv"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

raw_data = pd.read_csv(raw_file)

predictions = pd.read_csv(prediction_file)

ground_truth = pd.read_csv(ground_truth_file)


print("\nData loaded successfully!")

print("Raw records:", len(raw_data))
print("ML prediction records:", len(predictions))


# ============================================================
# 3. CREATE SENSOR HEALTH DATAFRAME
# ============================================================

health = raw_data[
    [
        "timestamp",
        "device_id",
        "raw_pm2_5",
        "temperature",
        "humidity"
    ]
].copy()


# ============================================================
# 4. DETECT MISSING PM2.5
# ============================================================

health["missing_pm25"] = (
    health["raw_pm2_5"].isna()
)


print("\nMissing PM2.5 records:")

print(
    health["missing_pm25"].sum()
)


# ============================================================
# 5. ADD ML PREDICTIONS
# ============================================================

health = health.merge(
    predictions[
        [
            "timestamp",
            "device_id",
            "anomaly_score",
            "prediction",
            "predicted_status"
        ]
    ],
    on=[
        "timestamp",
        "device_id"
    ],
    how="left"
)


# ============================================================
# 6. DETERMINE SENSOR STATUS
# ============================================================

health["sensor_status"] = "normal"

health["fault_reason_detected"] = "none"


# ------------------------------------------------------------
# Missing PM2.5
# ------------------------------------------------------------

missing_mask = health["missing_pm25"]

health.loc[
    missing_mask,
    "sensor_status"
] = "fault_detected"

health.loc[
    missing_mask,
    "fault_reason_detected"
] = "missing_data"


# ------------------------------------------------------------
# ML anomaly
# ------------------------------------------------------------

ml_anomaly_mask = (
    health["prediction"] == -1
)

health.loc[
    ml_anomaly_mask,
    "sensor_status"
] = "fault_detected"


# ------------------------------------------------------------
# ML anomaly reason
# ------------------------------------------------------------

health.loc[
    ml_anomaly_mask
    & ~missing_mask,
    "fault_reason_detected"
] = "behavioral_anomaly"


# ============================================================
# 7. FINAL SENSOR STATUS COUNTS
# ============================================================

print("\n============================================")
print("FINAL SENSOR STATUS")
print("============================================")

print(
    health["sensor_status"]
    .value_counts()
)


# ============================================================
# 8. FAULT REASON COUNTS
# ============================================================

print("\n============================================")
print("DETECTED FAULT REASONS")
print("============================================")

print(
    health["fault_reason_detected"]
    .value_counts()
)


# ============================================================
# 9. DEVICE-WISE SENSOR HEALTH
# ============================================================

device_health = (
    health
    .groupby("device_id")
    .agg(
        total_records=(
            "device_id",
            "size"
        ),

        missing_pm25_records=(
            "missing_pm25",
            "sum"
        ),

        ml_anomaly_records=(
            "prediction",
            lambda x: (x == -1).sum()
        ),

        fault_detected_records=(
            "sensor_status",
            lambda x: (
                x == "fault_detected"
            ).sum()
        )
    )
)


device_health[
    "fault_percentage"
] = (
    device_health[
        "fault_detected_records"
    ]
    / device_health[
        "total_records"
    ]
    * 100
)


print("\n============================================")
print("DEVICE-WISE SENSOR HEALTH")
print("============================================")

print(
    device_health
)


# ============================================================
# 10. MERGE GROUND TRUTH
# ============================================================

health = health.merge(
    ground_truth[
        [
            "timestamp",
            "device_id",
            "fault_label",
            "fault_reason"
        ]
    ],
    on=[
        "timestamp",
        "device_id"
    ],
    how="left"
)


# ============================================================
# 11. COMPARE FINAL DETECTOR WITH GROUND TRUTH
# ============================================================

health["predicted_fault"] = (
    health["sensor_status"]
    == "fault_detected"
).astype(int)


health["actual_fault"] = (
    health["fault_label"]
    .fillna(0)
    .astype(int)
)


# ============================================================
# 12. FINAL DETECTION STATISTICS
# ============================================================

tp = (
    (
        (health["actual_fault"] == 1)
        &
        (health["predicted_fault"] == 1)
    )
    .sum()
)

tn = (
    (
        (health["actual_fault"] == 0)
        &
        (health["predicted_fault"] == 0)
    )
    .sum()
)

fp = (
    (
        (health["actual_fault"] == 0)
        &
        (health["predicted_fault"] == 1)
    )
    .sum()
)

fn = (
    (
        (health["actual_fault"] == 1)
        &
        (health["predicted_fault"] == 0)
    )
    .sum()
)


print("\n============================================")
print("FINAL SENSOR FAULT DETECTION")
print("============================================")

print("True Positives :", tp)
print("True Negatives :", tn)
print("False Positives:", fp)
print("False Negatives:", fn)


# ============================================================
# 13. SAVE RESULTS
# ============================================================

health.to_csv(
    output_file,
    index=False
)


print("\nSensor health results saved successfully!")

print(
    "File:",
    output_file
)


# ============================================================
# 14. FINAL MESSAGE
# ============================================================

print("\n============================================")
print("SENSOR HEALTH ANALYSIS COMPLETED")
print("============================================")