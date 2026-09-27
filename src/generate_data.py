import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

np.random.seed(42)


# ============================================================
# 2. DATASET CONFIGURATION
# ============================================================

NUM_DAYS = 7
INTERVAL_MINUTES = 5
NUM_DEVICES = 5

START_TIME = "2026-01-01 00:00:00"

DEVICE_IDS = [
    "D01",
    "D02",
    "D03",
    "D04",
    "D05"
]

# Threshold at which gradual drift/degradation
# is considered a sensor fault
FAULT_THRESHOLD = 5


# ============================================================
# 3. CREATE TIMESTAMPS
# ============================================================

timestamps = pd.date_range(
    start=START_TIME,
    periods=(NUM_DAYS * 24 * 60) // INTERVAL_MINUTES,
    freq=f"{INTERVAL_MINUTES}min"
)

print(f"Number of timestamps: {len(timestamps)}")


# ============================================================
# 4. GENERATE ENVIRONMENTAL CONDITIONS
# ============================================================

time_index = np.arange(len(timestamps))

samples_per_day = (
    24 * 60 / INTERVAL_MINUTES
)


# ------------------------------------------------------------
# PM2.5
# ------------------------------------------------------------

daily_variation = 10 * np.sin(
    2 * np.pi * time_index / samples_per_day
)

random_variation = np.random.normal(
    0,
    2,
    len(timestamps)
)

true_pm2_5 = (
    40
    + daily_variation
    + random_variation
)


# ------------------------------------------------------------
# Temperature
# ------------------------------------------------------------

temperature = (
    27
    + 5 * np.sin(
        2 * np.pi * (time_index - 360)
        / samples_per_day
    )
    + np.random.normal(
        0,
        0.5,
        len(timestamps)
    )
)


# ------------------------------------------------------------
# Humidity
# ------------------------------------------------------------

humidity = (
    65
    - 10 * np.sin(
        2 * np.pi * (time_index - 360)
        / samples_per_day
    )
    + np.random.normal(
        0,
        1.5,
        len(timestamps)
    )
)


# ============================================================
# 5. KEEP VALUES WITHIN REALISTIC RANGES
# ============================================================

true_pm2_5 = np.clip(
    true_pm2_5,
    10,
    100
)

temperature = np.clip(
    temperature,
    15,
    40
)

humidity = np.clip(
    humidity,
    20,
    95
)


print(
    f"Average PM2.5: "
    f"{true_pm2_5.mean():.2f}"
)

print(
    f"Average temperature: "
    f"{temperature.mean():.2f} °C"
)

print(
    f"Average humidity: "
    f"{humidity.mean():.2f} %"
)


# ============================================================
# 6. CREATE SENSOR DATA
# ============================================================

all_sensor_data = []


for device_id in DEVICE_IDS:

    # --------------------------------------------------------
    # Start with true environmental PM2.5
    # --------------------------------------------------------

    sensor_pm2_5 = true_pm2_5.copy()

    # Every sensor has small measurement noise
    sensor_pm2_5 += np.random.normal(
        0,
        1,
        len(timestamps)
    )


    # --------------------------------------------------------
    # Initialize ground-truth labels
    # --------------------------------------------------------

    fault_label = np.zeros(
        len(timestamps),
        dtype=int
    )

    fault_reason = np.full(
        len(timestamps),
        "normal",
        dtype=object
    )


    # ========================================================
    # D01 - HEALTHY SENSOR
    # ========================================================

    if device_id == "D01":

        fault_type = "none"


    # ========================================================
    # D02 - GRADUAL SENSOR DRIFT
    # ========================================================

    elif device_id == "D02":

        fault_type = "drift"

        # Gradual drift from 0 to +20
        drift = np.linspace(
            0,
            20,
            len(timestamps)
        )

        sensor_pm2_5 += drift

        # Mark rows where drift has become significant
        drift_mask = drift >= FAULT_THRESHOLD

        fault_label[drift_mask] = 1
        fault_reason[drift_mask] = "drift"


    # ========================================================
    # D03 - RANDOM SPIKES
    # ========================================================

    elif device_id == "D03":

        fault_type = "spikes"

        # Select 25 random readings
        spike_indices = np.random.choice(
            len(timestamps),
            size=25,
            replace=False
        )

        # Generate spike magnitude
        spike_values = np.random.uniform(
            50,
            120,
            len(spike_indices)
        )

        sensor_pm2_5[spike_indices] += spike_values

        # Mark exact spike records
        fault_label[spike_indices] = 1
        fault_reason[spike_indices] = "spike"


    # ========================================================
    # D04 - MISSING DATA
    # ========================================================

    elif device_id == "D04":

        fault_type = "missing_data"

        missing_indices = np.random.choice(
            len(timestamps),
            size=100,
            replace=False
        )

        sensor_pm2_5[missing_indices] = np.nan

        # Mark exact missing-data records
        fault_label[missing_indices] = 1
        fault_reason[missing_indices] = "missing_data"


    # ========================================================
    # D05 - PROGRESSIVE DEGRADATION
    # ========================================================

    elif device_id == "D05":

        fault_type = "progressive_degradation"

        # ----------------------------------------------------
        # Gradual degradation
        # ----------------------------------------------------

        degradation = np.linspace(
            0,
            25,
            len(timestamps)
        )

        sensor_pm2_5 += degradation

        degradation_mask = (
            degradation >= FAULT_THRESHOLD
        )

        fault_label[degradation_mask] = 1
        fault_reason[degradation_mask] = (
            "progressive_degradation"
        )


        # ----------------------------------------------------
        # Occasional spikes
        # ----------------------------------------------------

        spike_indices = np.random.choice(
            len(timestamps),
            size=40,
            replace=False
        )

        spike_values = np.random.uniform(
            30,
            80,
            len(spike_indices)
        )

        sensor_pm2_5[spike_indices] += spike_values


        # ----------------------------------------------------
        # Missing readings
        # ----------------------------------------------------

        later_indices = np.arange(
            len(timestamps) // 2,
            len(timestamps)
        )

        missing_indices = np.random.choice(
            later_indices,
            size=80,
            replace=False
        )

        sensor_pm2_5[missing_indices] = np.nan


        # ----------------------------------------------------
        # Update fault labels/reasons
        # ----------------------------------------------------

        for i in range(len(timestamps)):

            degradation_fault = (
                degradation[i]
                >= FAULT_THRESHOLD
            )

            spike_fault = (
                i in spike_indices
            )

            missing_fault = (
                i in missing_indices
            )

            faults = []

            if degradation_fault:
                faults.append(
                    "progressive_degradation"
                )

            if spike_fault:
                faults.append(
                    "spike"
                )

            if missing_fault:
                faults.append(
                    "missing_data"
                )

            if len(faults) > 0:

                fault_label[i] = 1

                fault_reason[i] = "_and_".join(
                    faults
                )


    # ========================================================
    # 7. CREATE DEVICE DATAFRAME
    # ========================================================

    device_data = pd.DataFrame({

        "timestamp": timestamps,

        "device_id": device_id,

        "raw_pm2_5": sensor_pm2_5,

        "temperature": temperature,

        "humidity": humidity,

        "true_pm2_5": true_pm2_5,

        "fault_type": fault_type,

        "fault_label": fault_label,

        "fault_reason": fault_reason

    })

    all_sensor_data.append(
        device_data
    )


# ============================================================
# 8. COMBINE ALL DEVICES
# ============================================================

sensor_data = pd.concat(
    all_sensor_data,
    ignore_index=True
)


# ============================================================
# 9. SORT DATA
# ============================================================

sensor_data = sensor_data.sort_values(
    [
        "timestamp",
        "device_id"
    ]
).reset_index(
    drop=True
)


# ============================================================
# 10. BASIC INFORMATION
# ============================================================

print("\n============================================")
print("DATASET CREATED")
print("============================================")

print(
    f"Total records: "
    f"{len(sensor_data)}"
)

print(
    f"Number of devices: "
    f"{sensor_data['device_id'].nunique()}"
)


# ============================================================
# 11. RECORDS PER DEVICE
# ============================================================

print("\nRecords per device:")

print(
    sensor_data["device_id"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 12. MISSING PM2.5 VALUES
# ============================================================

print("\nMissing PM2.5 values:")

print(
    sensor_data["raw_pm2_5"]
    .isna()
    .sum()
)


# ============================================================
# 13. GROUND-TRUTH FAULT DISTRIBUTION
# ============================================================

print("\n============================================")
print("GROUND-TRUTH FAULT LABEL DISTRIBUTION")
print("============================================")

print(
    sensor_data["fault_label"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 14. FAULT REASON DISTRIBUTION
# ============================================================

print("\n============================================")
print("FAULT REASON DISTRIBUTION")
print("============================================")

print(
    sensor_data["fault_reason"]
    .value_counts()
)


# ============================================================
# 15. FAULT DISTRIBUTION BY DEVICE
# ============================================================

print("\n============================================")
print("FAULT DISTRIBUTION BY DEVICE")
print("============================================")

device_fault_summary = (
    sensor_data
    .groupby("device_id")
    .agg(
        total_records=("device_id", "size"),
        faulty_records=("fault_label", "sum")
    )
)

device_fault_summary["fault_percentage"] = (
    device_fault_summary["faulty_records"]
    / device_fault_summary["total_records"]
    * 100
)

print(
    device_fault_summary
)


# ============================================================
# 16. SAVE COMPLETE GROUND-TRUTH DATASET
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

data_dir = BASE_DIR / "Data"

data_dir.mkdir(
    parents=True,
    exist_ok=True
)


ground_truth_file = (
    data_dir
    / "sensor_dataset_with_ground_truth.csv"
)

sensor_data.to_csv(
    ground_truth_file,
    index=False
)

print(
    "\nComplete ground-truth dataset saved:"
)

print(
    ground_truth_file
)


# ============================================================
# 17. CREATE RAW ML DATASET
# ============================================================

ml_data = sensor_data[
    [
        "timestamp",
        "device_id",
        "raw_pm2_5",
        "temperature",
        "humidity"
    ]
].copy()


raw_file = (
    data_dir
    / "raw_sensor_data.csv"
)

ml_data.to_csv(
    raw_file,
    index=False
)

print(
    "\nRaw ML dataset saved:"
)

print(
    raw_file
)


# ============================================================
# 18. FINAL MESSAGE
# ============================================================

print("\n============================================")
print("DATA GENERATION COMPLETED")
print("============================================")