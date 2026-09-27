import pandas as pd
from pathlib import Path

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

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

evaluation_file = (
    BASE_DIR
    / "reports"
    / "model_evaluation.csv"
)


# ============================================================
# 2. LOAD PREDICTIONS
# ============================================================

predictions = pd.read_csv(prediction_file)

print("\nPrediction results loaded!")
print("Prediction shape:", predictions.shape)


# ============================================================
# 3. LOAD GROUND TRUTH
# ============================================================

ground_truth = pd.read_csv(ground_truth_file)

print("\nGround-truth dataset loaded!")
print("Ground-truth shape:", ground_truth.shape)


# ============================================================
# 4. CHECK ACTUAL FAULT TYPES
# ============================================================

print("\n============================================")
print("GROUND-TRUTH FAULT TYPES")
print("============================================")

print(
    ground_truth[
        ["device_id", "fault_type"]
    ].drop_duplicates()
    .sort_values("device_id")
)


# ============================================================
# 5. MERGE PREDICTIONS WITH GROUND TRUTH
# ============================================================

evaluation = predictions.merge(
    ground_truth[
        [
            "timestamp",
            "device_id",
            "true_pm2_5",
            "fault_type"
        ]
    ],
    on=["timestamp", "device_id"],
    how="left"
)


print("\nEvaluation dataset created!")
print("Shape:", evaluation.shape)


# ============================================================
# 6. CREATE ACTUAL FAULT LABEL
# ============================================================

# D01 = healthy
# All other fault types = faulty sensor

evaluation["actual_fault"] = (
    evaluation["fault_type"] != "none"
).astype(int)


# Model prediction:
# -1 = anomaly
#  1 = normal

evaluation["predicted_fault"] = (
    evaluation["prediction"] == -1
).astype(int)


# ============================================================
# 7. CHECK LABEL COUNTS
# ============================================================

print("\n============================================")
print("ACTUAL FAULT DISTRIBUTION")
print("============================================")

print(
    evaluation["fault_type"].value_counts()
)


print("\n============================================")
print("PREDICTED DISTRIBUTION")
print("============================================")

print(
    evaluation["predicted_status"].value_counts()
)


# ============================================================
# 8. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    evaluation["actual_fault"],
    evaluation["predicted_fault"]
)

print("\n============================================")
print("CONFUSION MATRIX")
print("============================================")

print(cm)


# ============================================================
# 9. EXTRACT TP / TN / FP / FN
# ============================================================

tn, fp, fn, tp = cm.ravel()

print("\nTrue Negatives :", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives  :", tp)


# ============================================================
# 10. PRECISION / RECALL / F1
# ============================================================

precision = precision_score(
    evaluation["actual_fault"],
    evaluation["predicted_fault"],
    zero_division=0
)

recall = recall_score(
    evaluation["actual_fault"],
    evaluation["predicted_fault"],
    zero_division=0
)

f1 = f1_score(
    evaluation["actual_fault"],
    evaluation["predicted_fault"],
    zero_division=0
)


print("\n============================================")
print("MODEL PERFORMANCE")
print("============================================")

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")


# ============================================================
# 11. FULL CLASSIFICATION REPORT
# ============================================================

print("\n============================================")
print("CLASSIFICATION REPORT")
print("============================================")

print(
    classification_report(
        evaluation["actual_fault"],
        evaluation["predicted_fault"],
        target_names=[
            "normal",
            "faulty"
        ],
        zero_division=0
    )
)


# ============================================================
# 12. PERFORMANCE BY FAULT TYPE
# ============================================================

print("\n============================================")
print("PERFORMANCE BY FAULT TYPE")
print("============================================")

fault_types = [
    "none",
    "drift",
    "spikes",
    "progressive_degradation"
]

for fault in fault_types:

    subset = evaluation[
        evaluation["fault_type"] == fault
    ]

    if len(subset) == 0:
        continue

    anomaly_rate = (
        subset["predicted_fault"].mean() * 100
    )

    print(
        f"{fault:25s} "
        f"records={len(subset):4d} "
        f"predicted_anomaly={anomaly_rate:6.2f}%"
    )


# ============================================================
# 13. DEVICE-WISE PERFORMANCE
# ============================================================

print("\n============================================")
print("DEVICE-WISE PERFORMANCE")
print("============================================")

device_results = []

for device_id, group in evaluation.groupby("device_id"):

    actual_fault_rate = (
        group["actual_fault"].mean() * 100
    )

    predicted_fault_rate = (
        group["predicted_fault"].mean() * 100
    )

    device_results.append({
        "device_id": device_id,
        "fault_type": group["fault_type"].iloc[0],
        "records": len(group),
        "actual_fault_percentage": actual_fault_rate,
        "predicted_anomaly_percentage": predicted_fault_rate
    })


device_results = pd.DataFrame(device_results)

print(device_results)


# ============================================================
# 14. SAVE EVALUATION DATA
# ============================================================

evaluation.to_csv(
    evaluation_file,
    index=False
)

print("\nEvaluation data saved successfully!")
print("File:", evaluation_file)


# ============================================================
# 15. FINAL MESSAGE
# ============================================================

print("\n============================================")
print("MODEL EVALUATION COMPLETED")
print("============================================")