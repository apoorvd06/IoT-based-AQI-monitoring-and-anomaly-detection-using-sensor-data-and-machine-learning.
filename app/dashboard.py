import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Air Quality Monitoring",
    page_icon="🌫️",
    layout="wide"
)


# ============================================================
# 2. PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

data_file = (
    BASE_DIR
    / "reports"
    / "final_sensor_diagnosis.csv"
)


# ============================================================
# 3. LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(data_file)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    return df


data = load_data()


# ============================================================
# 4. TITLE
# ============================================================

st.title(
    "🌫️ Smart Air Quality Monitoring & "
    "Sensor Anomaly Detection"
)

st.markdown(
    """
    **Real-time-style sensor monitoring using
    Isolation Forest and sensor health analysis.**
    """
)


# ============================================================
# 5. SIDEBAR
# ============================================================

st.sidebar.header("Dashboard Controls")

devices = sorted(
    data["device_id"]
    .dropna()
    .unique()
)

selected_device = st.sidebar.selectbox(
    "Select Device",
    ["All Devices"] + list(devices)
)


# ============================================================
# 6. FILTER DATA
# ============================================================

if selected_device == "All Devices":

    filtered = data.copy()

else:

    filtered = data[
        data["device_id"]
        == selected_device
    ].copy()


# ============================================================
# 7. KPI CALCULATIONS
# ============================================================

total_records = len(filtered)

fault_records = (
    filtered["final_sensor_status"]
    == "fault_detected"
).sum()

normal_records = (
    filtered["final_sensor_status"]
    == "normal"
).sum()

missing_records = (
    filtered["diagnosed_fault"]
    == "missing_data"
).sum()

anomaly_records = (
    filtered["predicted_fault"]
    == 1
).sum()


# ============================================================
# 8. KPI CARDS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Total Records",
        f"{total_records:,}"
    )


with col2:

    st.metric(
        "Normal Records",
        f"{normal_records:,}"
    )


with col3:

    st.metric(
        "Fault Detected",
        f"{fault_records:,}"
    )


with col4:

    st.metric(
        "Missing PM2.5",
        f"{missing_records:,}"
    )


# ============================================================
# 9. SENSOR STATUS
# ============================================================

st.subheader("Sensor Health Status")

status_counts = (
    filtered["final_sensor_status"]
    .value_counts()
    .reset_index()
)

status_counts.columns = [
    "status",
    "count"
]


fig_status = px.pie(
    status_counts,
    names="status",
    values="count",
    title="Sensor Health Distribution"
)

st.plotly_chart(
    fig_status,
    use_container_width=True
)


# ============================================================
# 10. PM2.5 TREND
# ============================================================

st.subheader("PM2.5 Monitoring")

pm_data = filtered.sort_values(
    "timestamp"
)


fig_pm = px.line(
    pm_data,
    x="timestamp",
    y="raw_pm2_5",
    color="device_id",
    title="PM2.5 Sensor Readings"
)

st.plotly_chart(
    fig_pm,
    use_container_width=True
)


# ============================================================
# 11. FAULT DISTRIBUTION
# ============================================================

st.subheader("Detected Fault Types")

fault_counts = (
    filtered["diagnosed_fault"]
    .value_counts()
    .reset_index()
)

fault_counts.columns = [
    "fault_type",
    "count"
]


fig_faults = px.bar(
    fault_counts,
    x="fault_type",
    y="count",
    title="Fault Type Distribution",
    text="count"
)

fig_faults.update_layout(
    xaxis_title="Fault Type",
    yaxis_title="Number of Records"
)

st.plotly_chart(
    fig_faults,
    use_container_width=True
)


# ============================================================
# 12. DEVICE-WISE FAULTS
# ============================================================

st.subheader("Device-wise Fault Detection")

device_faults = (
    data[
        data["final_sensor_status"]
        == "fault_detected"
    ]
    .groupby("device_id")
    .size()
    .reset_index(
        name="fault_records"
    )
)


fig_device = px.bar(
    device_faults,
    x="device_id",
    y="fault_records",
    title="Faults Detected by Device",
    text="fault_records"
)

st.plotly_chart(
    fig_device,
    use_container_width=True
)


# ============================================================
# 13. ANOMALY SCORE
# ============================================================

st.subheader("Anomaly Score")

score_data = filtered.sort_values(
    "timestamp"
)

fig_score = px.scatter(
    score_data,
    x="timestamp",
    y="anomaly_score",
    color="predicted_status",
    title="Isolation Forest Anomaly Scores"
)

st.plotly_chart(
    fig_score,
    use_container_width=True
)


# ============================================================
# 14. RECENT FAULTS
# ============================================================

st.subheader("Recent Fault Events")

recent_faults = (
    filtered[
        filtered["final_sensor_status"]
        == "fault_detected"
    ]
    .sort_values(
        "timestamp",
        ascending=False
    )
    .head(20)
)


display_columns = [
    "timestamp",
    "device_id",
    "raw_pm2_5",
    "temperature",
    "humidity",
    "diagnosed_fault",
    "anomaly_score"
]


st.dataframe(
    recent_faults[
        display_columns
    ],
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 15. PROJECT INFORMATION
# ============================================================

st.divider()

st.subheader("Detection Architecture")

st.markdown(
    """
    **1. Sensor Data**

    Raw PM2.5, temperature and humidity readings.

    **2. Data Quality Detection**

    Missing PM2.5 readings are identified directly.

    **3. Feature Engineering**

    PM2.5 change is calculated for anomaly detection.

    **4. Isolation Forest**

    Detects abnormal sensor behaviour.

    **5. Sensor Health Layer**

    Combines missing-data detection and ML anomalies.

    **6. Fault Diagnosis**

    Classifies detected problems into:

    - Missing Data
    - Sudden Spike
    - Gradual Drift
    - Progressive Degradation
    - Behavioral Anomaly
    """
)


# ============================================================
# 16. FOOTER
# ============================================================

st.divider()

st.caption(
    "Smart Air Quality Monitoring and "
    "Sensor Anomaly Detection System"
)