import numpy as np
import pandas as pd

# Reproducibility
np.random.seed(42)

# Dataset configuration
NUM_DAYS = 7
INTERVAL_MINUTES = 5
NUM_DEVICES = 5

START_TIME = "2026-01-01 00:00:00"

DEVICE_IDS = ["D01", "D02", "D03", "D04", "D05"]

timestamps = pd.date_range(
    start=START_TIME,
    periods=(NUM_DAYS * 24 * 60) // INTERVAL_MINUTES,
    freq=f"{INTERVAL_MINUTES}min"
)

print(f"Number of timestamps: {len(timestamps)}")

# Generate realistic environmental conditions

time_index = np.arange(len(timestamps))

# Daily variation in PM2.5
daily_variation = 10 * np.sin(
    2 * np.pi * time_index / (24 * 60 / INTERVAL_MINUTES)
)

# Random environmental variation
random_variation = np.random.normal(0, 2, len(timestamps))

# True PM2.5 value
true_pm2_5 = 40 + daily_variation + random_variation

# Temperature variation
temperature = (
    27
    + 5 * np.sin(
        2 * np.pi * (time_index - 360)
        / (24 * 60 / INTERVAL_MINUTES)
    )
    + np.random.normal(0, 0.5, len(timestamps))
)

# Humidity variation
humidity = (
    65
    - 10 * np.sin(
        2 * np.pi * (time_index - 360)
        / (24 * 60 / INTERVAL_MINUTES)
    )
    + np.random.normal(0, 1.5, len(timestamps))
)

# Keep values within realistic ranges
true_pm2_5 = np.clip(true_pm2_5, 10, 100)
temperature = np.clip(temperature, 15, 40)
humidity = np.clip(humidity, 20, 95)

print(f"Average PM2.5: {true_pm2_5.mean():.2f}")
print(f"Average temperature: {temperature.mean():.2f} °C")
print(f"Average humidity: {humidity.mean():.2f} %")

# Create sensor data for each device

all_sensor_data = []

for device_id in DEVICE_IDS:

    # Start with the true environmental PM2.5
    sensor_pm2_5 = true_pm2_5.copy()

    # Every real sensor has a small measurement error
    sensor_pm2_5 += np.random.normal(0, 1, len(timestamps))

    # --------------------------------------------------
    # D01 - Healthy sensor
    # --------------------------------------------------
    if device_id == "D01":
        fault_type = "none"

    # --------------------------------------------------
    # D02 - Gradual sensor drift
    # --------------------------------------------------
    elif device_id == "D02":
        fault_type = "drift"

        # Gradually increasing error
        drift = np.linspace(0, 20, len(timestamps))
        sensor_pm2_5 += drift

    # --------------------------------------------------
    # D03 - Random spikes
    # --------------------------------------------------
    elif device_id == "D03":
        fault_type = "spikes"

        # Select random readings
        spike_indices = np.random.choice(
            len(timestamps),
            size=25,
            replace=False
        )

        # Add large errors to those readings
        sensor_pm2_5[spike_indices] += np.random.uniform(
            50, 120, len(spike_indices)
        )

    # --------------------------------------------------
    # D04 - Missing data
    # --------------------------------------------------
    elif device_id == "D04":
        fault_type = "missing_data"

        missing_indices = np.random.choice(
            len(timestamps),
            size=100,
            replace=False
        )

        sensor_pm2_5[missing_indices] = np.nan

    # --------------------------------------------------
    # D05 - Progressive degradation
    # --------------------------------------------------
    elif device_id == "D05":
        fault_type = "progressive_degradation"

        # Start with small drift
        degradation = np.linspace(0, 25, len(timestamps))
        sensor_pm2_5 += degradation

        # Add occasional spikes
        spike_indices = np.random.choice(
            len(timestamps),
            size=40,
            replace=False
        )

        sensor_pm2_5[spike_indices] += np.random.uniform(
            30, 80, len(spike_indices)
        )

        # Add missing readings toward the later part
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

    # Create a dataframe for this device
    device_data = pd.DataFrame({
        "timestamp": timestamps,
        "device_id": device_id,
        "raw_pm2_5": sensor_pm2_5,
        "temperature": temperature,
        "humidity": humidity,
        "true_pm2_5": true_pm2_5,
        "fault_type": fault_type
    })

    all_sensor_data.append(device_data)

    # Combine all sensor data into one dataframe

sensor_data = pd.concat(
    all_sensor_data,
    ignore_index=True
)

# Sort by timestamp and device
sensor_data = sensor_data.sort_values(
    ["timestamp", "device_id"]
).reset_index(drop=True)

# Display basic information
print("\nDataset created successfully!")
print(f"Total records: {len(sensor_data)}")
print(f"Number of devices: {sensor_data['device_id'].nunique()}")

print("\nRecords per device:")
print(sensor_data["device_id"].value_counts())

print("\nMissing PM2.5 values:")
print(sensor_data["raw_pm2_5"].isna().sum())

# Save the complete dataset
sensor_data.to_csv(
    "data/sensor_dataset_with_ground_truth.csv",
    index=False
)

print("\nDataset saved successfully!")

# Create the dataset that our ML system will actually use

ml_data = sensor_data[
    [
        "timestamp",
        "device_id",
        "raw_pm2_5",
        "temperature",
        "humidity"
    ]
].copy()

ml_data.to_csv(
    "data/raw_sensor_data.csv",
    index=False
)

print("ML dataset saved as: data/raw_sensor_data.csv")