import pandas as pd
from pathlib import Path


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

prediction_file = (
    BASE_DIR
    / "reports"
    / "raw_data_predictions.csv"
)


# ============================================================
# 2. LOAD PREDICTION RESULTS
# ============================================================

data = pd.read_csv(prediction_file)

print("\nPrediction results loaded successfully!")
print("Shape:", data.shape)


# ============================================================
# 3. SEPARATE NORMAL AND ANOMALY DATA
# ============================================================

normal = data[
    data["prediction"] == 1
].copy()

anomalies = data[
    data["prediction"] == -1
].copy()

print("\nNormal records:", len(normal))
print("Anomaly records:", len(anomalies))


# ============================================================
# 4. ANOMALY SCORE STATISTICS
# ============================================================

print("\n============================================")
print("ANOMALY SCORE STATISTICS")
print("============================================")

print(
    anomalies["anomaly_score"].describe()
)


# ============================================================
# 5. COMPARE NORMAL VS ANOMALY FEATURES
# ============================================================

features = [
    "raw_pm2_5",
    "temperature",
    "humidity",
    "pm25_change"
]

print("\n============================================")
print("NORMAL DATA STATISTICS")
print("============================================")

print(
    normal[features].describe()
)


print("\n============================================")
print("ANOMALY DATA STATISTICS")
print("============================================")

print(
    anomalies[features].describe()
)


# ============================================================
# 6. FEATURE MEAN COMPARISON
# ============================================================

normal_mean = normal[features].mean()
anomaly_mean = anomalies[features].mean()

comparison = pd.DataFrame({
    "normal_mean": normal_mean,
    "anomaly_mean": anomaly_mean
})

comparison["difference"] = (
    comparison["anomaly_mean"]
    - comparison["normal_mean"]
)

print("\n============================================")
print("NORMAL VS ANOMALY MEAN")
print("============================================")

print(comparison)


# ============================================================
# 7. ANOMALIES BY DEVICE
# ============================================================

anomaly_count = (
    anomalies
    .groupby("device_id")
    .size()
    .rename("anomaly_records")
)

total_count = (
    data
    .groupby("device_id")
    .size()
    .rename("total_records")
)

device_analysis = pd.concat(
    [total_count, anomaly_count],
    axis=1
).fillna(0)

device_analysis["anomaly_records"] = (
    device_analysis["anomaly_records"]
    .astype(int)
)

device_analysis["anomaly_percentage"] = (
    device_analysis["anomaly_records"]
    / device_analysis["total_records"]
    * 100
)

device_analysis = device_analysis.sort_values(
    "anomaly_percentage",
    ascending=False
)

print("\n============================================")
print("DEVICE ANOMALY ANALYSIS")
print("============================================")

print(device_analysis)


# ============================================================
# 8. AVERAGE ANOMALY SCORE BY DEVICE
# ============================================================

score_by_device = (
    anomalies
    .groupby("device_id")["anomaly_score"]
    .mean()
    .sort_values()
)

print("\n============================================")
print("AVERAGE ANOMALY SCORE BY DEVICE")
print("============================================")

print(score_by_device)


# ============================================================
# 9. MOST EXTREME ANOMALIES
# ============================================================

print("\n============================================")
print("10 MOST EXTREME ANOMALIES")
print("============================================")

most_extreme = anomalies.sort_values(
    "anomaly_score"
).head(10)

print(
    most_extreme[
        [
            "timestamp",
            "device_id",
            "raw_pm2_5",
            "temperature",
            "humidity",
            "pm25_change",
            "anomaly_score"
        ]
    ]
)


# ============================================================
# 10. DEVICE FEATURE STATISTICS
# ============================================================

print("\n============================================")
print("DEVICE-WISE PM2.5 STATISTICS")
print("============================================")

device_pm_stats = (
    data
    .groupby("device_id")["raw_pm2_5"]
    .agg([
        "count",
        "mean",
        "std",
        "min",
        "max"
    ])
)

print(device_pm_stats)


# ============================================================
# 11. DEVICE-WISE PM2.5 CHANGE STATISTICS
# ============================================================

print("\n============================================")
print("DEVICE-WISE PM2.5 CHANGE STATISTICS")
print("============================================")

device_change_stats = (
    data
    .groupby("device_id")["pm25_change"]
    .agg([
        "mean",
        "std",
        "min",
        "max"
    ])
)

print(device_change_stats)


# ============================================================
# 12. SAVE ANALYSIS REPORT
# ============================================================

report_file = (
    BASE_DIR
    / "reports"
    / "anomaly_analysis.csv"
)

device_analysis.to_csv(report_file)

print("\nAnalysis report saved successfully!")
print("File:", report_file)


# ============================================================
# 13. FINAL MESSAGE
# ============================================================

print("\n============================================")
print("ANOMALY ANALYSIS COMPLETED")
print("============================================")