import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest


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
# Select only normal data for Isolation Forest training
normal_data = ground_truth[ground_truth["fault_type"] == "none"].copy()

print("\nNormal Data for Training:")
print(normal_data.head())

print("\nNormal Data Shape:")
print(normal_data.shape)

print("\nMissing Values in Normal Data:")
print(normal_data.isnull().sum())
# Select features for Isolation Forest
features = ["raw_pm2_5", "temperature", "humidity"]

X_train = normal_data[features]

# Scale the training features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
# Create the Isolation Forest model
model = IsolationForest(
    n_estimators=100,
    contamination="auto",
    random_state=42
)

# Train the model on normal data
model.fit(X_train_scaled)

print("\nIsolation Forest model trained successfully!")

# Prepare all ground-truth data for testing
X_test = ground_truth[features]

# Use the already fitted scaler
X_test_scaled = scaler.transform(X_test)

# Predict anomalies
predictions = model.predict(X_test_scaled)

print("\nPredictions:")
print(predictions[:20])
print("\nScaled Training Data:")
print(X_train_scaled[:5])

print("\nFeatures used for Isolation Forest:")
print(X_train.head())

print("\nTraining Data Shape:")
print(X_train.shape)

# Add predictions to the ground-truth dataset
ground_truth["prediction"] = predictions

# Convert Isolation Forest output to readable labels
ground_truth["predicted_status"] = ground_truth["prediction"].map({
    1: "normal",
    -1: "anomaly"
})

print("\nPrediction Distribution:")
print(ground_truth["predicted_status"].value_counts())

print("\nPredictions by Actual Fault Type:")
print(
    pd.crosstab(
        ground_truth["fault_type"],
        ground_truth["predicted_status"]
    )
)

# Get anomaly scores
scores = model.decision_function(X_test_scaled)

print("\nAnomaly Score Statistics:")
print(pd.Series(scores).describe())

# Analyze anomaly scores by actual fault type
ground_truth["anomaly_score"] = scores

print("\nAnomaly Score by Fault Type:")
print(
    ground_truth.groupby("fault_type")["anomaly_score"]
    .agg(["mean", "median", "min", "max"])
)
print("\nAnomaly Rate by Fault Type:")
print(
    ground_truth.groupby("fault_type")["anomaly_score"]
    .apply(lambda x: (x < 0).mean() * 100)
)