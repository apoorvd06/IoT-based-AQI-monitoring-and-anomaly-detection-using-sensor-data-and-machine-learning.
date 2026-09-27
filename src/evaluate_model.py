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

predictions = pd.read_csv(
    prediction_file
)

print("\nPrediction results loaded!")
print("Prediction shape:", predictions.shape)


# ============================================================
# 3. LOAD RECORD-LEVEL GROUND TRUTH
# ============================================================

ground_truth = pd.read_csv(
    ground_truth_file
)

print("\nGround-truth dataset loaded!")
print("Ground-truth shape:", ground_truth.shape)


# ============================================================
# 4. MERGE PREDICTIONS WITH GROUND TRUTH
# ============================================================

evaluation = predictions.merge(
    ground_truth[
        [
            "timestamp",
            "device_id",
            "true_pm2_5",
            "fault_type",
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


print("\nEvaluation dataset created!")
print("Shape:", evaluation.shape)


# ============================================================
# 5. CHECK FOR MERGE PROBLEMS
# ============================================================

missing_labels = evaluation["fault_label"].isna().sum()

print(
    "\nMissing ground-truth labels after merge:",
    missing_labels
)


# ============================================================
# 6. ACTUAL FAULT LABEL
# ============================================================

# 0 = normal
# 1 = actual sensor fault

evaluation["actual_fault"] = (
    evaluation["fault_label"]
    .astype(int)
)


# ============================================================
# 7. MODEL PREDICTION
# ============================================================

# Isolation Forest:
#  1  = normal
# -1  = anomaly

evaluation["predicted_fault"] = (
    evaluation["prediction"] == -1
).astype(int)


# ============================================================
# 8. BASIC DISTRIBUTION
# ============================================================

print("\n============================================")
print("ACTUAL FAULT DISTRIBUTION")
print("============================================")

print(
    evaluation["actual_fault"]
    .value_counts()
    .sort_index()
)


print("\n============================================")
print("PREDICTED DISTRIBUTION")
print("============================================")

print(
    evaluation["predicted_fault"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 9. CONFUSION MATRIX
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
# 10. TRUE / FALSE POSITIVES AND NEGATIVES
# ============================================================

tn, fp, fn, tp = cm.ravel()


print("\nTrue Negatives :", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives  :", tp)


# ============================================================
# 11. PRECISION
# ============================================================

precision = precision_score(
    evaluation["actual_fault"],
    evaluation["predicted_fault"],
    zero_division=0
)


# ============================================================
# 12. RECALL
# ============================================================

recall = recall_score(
    evaluation["actual_fault"],
    evaluation["predicted_fault"],
    zero_division=0
)


# ============================================================
# 13. F1 SCORE
# ============================================================

f1 = f1_score(
    evaluation["actual_fault"],
    evaluation["predicted_fault"],
    zero_division=0
)


print("\n============================================")
print("MODEL PERFORMANCE")
print("============================================")

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall:    {recall:.4f}"
)

print(
    f"F1-score:  {f1:.4f}"
)


# ============================================================
# 14. CLASSIFICATION REPORT
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
# 15. PERFORMANCE BY FAULT REASON
# ============================================================

print("\n============================================")
print("PERFORMANCE BY FAULT REASON")
print("============================================")


fault_reasons = (
    evaluation["fault_reason"]
    .dropna()
    .unique()
)


for reason in sorted(fault_reasons):

    subset = evaluation[
        evaluation["fault_reason"] == reason
    ]

    actual_faults = subset["actual_fault"].sum()

    predicted_anomalies = (
        subset["predicted_fault"].sum()
    )

    anomaly_rate = (
        predicted_anomalies
        / len(subset)
        * 100
    )

    print(
        f"{reason:40s} "
        f"records={len(subset):4d} "
        f"actual_faults={actual_faults:4d} "
        f"predicted_anomalies={predicted_anomalies:4d} "
        f"anomaly_rate={anomaly_rate:6.2f}%"
    )


# ============================================================
# 16. DEVICE-WISE ACTUAL VS PREDICTED
# ============================================================

print("\n============================================")
print("DEVICE-WISE ACTUAL VS PREDICTED")
print("============================================")


device_results = []


for device_id, group in evaluation.groupby(
    "device_id"
):

    actual_faults = (
        group["actual_fault"].sum()
    )

    predicted_anomalies = (
        group["predicted_fault"].sum()
    )

    device_results.append({

        "device_id": device_id,

        "fault_type": group[
            "fault_type"
        ].iloc[0],

        "usable_records": len(group),

        "actual_fault_records": int(
            actual_faults
        ),

        "predicted_anomaly_records": int(
            predicted_anomalies
        ),

        "actual_fault_percentage": (
            actual_faults
            / len(group)
            * 100
        ),

        "predicted_anomaly_percentage": (
            predicted_anomalies
            / len(group)
            * 100
        )

    })


device_results = pd.DataFrame(
    device_results
)


print(device_results)


# ============================================================
# 17. DETECTION RATE FOR ACTUAL FAULTS
# ============================================================

print("\n============================================")
print("FAULT DETECTION RATE")
print("============================================")


actual_fault_records = (
    evaluation[
        evaluation["actual_fault"] == 1
    ]
)


detected_fault_records = (
    actual_fault_records[
        actual_fault_records["predicted_fault"] == 1
    ]
)


fault_detection_rate = (
    len(detected_fault_records)
    / len(actual_fault_records)
    * 100
)


print(
    f"Actual faulty records: "
    f"{len(actual_fault_records)}"
)

print(
    f"Detected faulty records: "
    f"{len(detected_fault_records)}"
)

print(
    f"Fault detection rate: "
    f"{fault_detection_rate:.2f}%"
)


# ============================================================
# 18. SAVE EVALUATION RESULTS
# ============================================================

evaluation.to_csv(
    evaluation_file,
    index=False
)


print(
    "\nEvaluation data saved successfully!"
)

print(
    "File:",
    evaluation_file
)


# ============================================================
# 19. SAVE DEVICE RESULTS
# ============================================================

device_file = (
    BASE_DIR
    / "reports"
    / "device_evaluation.csv"
)


device_results.to_csv(
    device_file,
    index=False
)


print(
    "\nDevice evaluation saved successfully!"
)

print(
    "File:",
    device_file
)


# ============================================================
# 20. FINAL MESSAGE
# ============================================================

print("\n============================================")
print("CORRECTED MODEL EVALUATION COMPLETED")
print("============================================")