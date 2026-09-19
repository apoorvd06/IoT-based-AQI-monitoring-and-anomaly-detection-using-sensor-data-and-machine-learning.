import pandas as pd

ground_truth = pd.read_csv("Data/sensor_dataset_with_ground_truth.csv")
raw_data = pd.read_csv("Data/raw_sensor_data.csv")

print("Ground Truth Data:")
print(ground_truth.head())

print("\nRaw Data:")
print(raw_data.head())

print("\nGround Truth Information:")
print(ground_truth.info())

print("\nRaw Data Information:")
print(raw_data.info())

print("\nMissing Values in Ground Truth:")
print(ground_truth.isnull().sum())

print("\nMissing Values in Raw Data:")
print(raw_data.isnull().sum())

print("\nFault Type Distribution:")
print(ground_truth["fault_type"].value_counts())

print("\nMissing PM2.5 by Fault Type:")
print(
    ground_truth[ground_truth["raw_pm2_5"].isnull()]
    ["fault_type"]
    .value_counts()
)