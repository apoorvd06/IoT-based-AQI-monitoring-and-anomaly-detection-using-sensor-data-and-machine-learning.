# IoT-Based AQI Monitoring and Anomaly Detection Using Sensor Data and Machine Learning

# IoT-Based AQI Monitoring and Anomaly Detection Using Sensor Data and Machine Learning

## 🚀 Live Demo

👉 **[Open the Live Dashboard](https://sensorhealthmonitoringsystem.streamlit.app/)**

## 📌 Project Overview

Air quality monitoring systems depend on sensors to continuously measure parameters such as PM2.5, temperature, and humidity. However, sensors can develop faults such as missing readings, sudden spikes, gradual drift, or progressive degradation.

This project presents an **IoT-based air quality monitoring and sensor anomaly detection system** that combines sensor data processing, Machine Learning, and rule-based fault diagnosis.

The system uses an **Isolation Forest** model to detect abnormal sensor behavior and a separate sensor-health layer to identify missing data. A rule-based diagnosis layer then categorizes detected faults into different fault types.

A **Streamlit dashboard** provides real-time-style visualization of sensor health, PM2.5 trends, anomalies, and detected fault types.

---

## 🎯 Objectives

The main objectives of the project are:

- Monitor PM2.5, temperature, and humidity sensor data.
- Detect abnormal sensor behavior automatically.
- Identify missing PM2.5 readings.
- Detect different types of sensor faults.
- Use Machine Learning for behavioral anomaly detection.
- Provide sensor-wise health information.
- Visualize sensor data and detected faults through a dashboard.
- Evaluate the performance of the anomaly detection model using ground-truth data.

---

## 🏗️ System Architecture

```text
                Sensor Data
                     │
                     ▼
        ┌─────────────────────────┐
        │     Data Generation     │
        │  / Raw Sensor Dataset   │
        └────────────┬────────────┘
                     │
                     ▼
        ┌─────────────────────────┐
        │   Feature Engineering   │
        │                         │
        │  • PM2.5                │
        │  • Temperature          │
        │  • Humidity             │
        │  • PM2.5 Change         │
        └────────────┬────────────┘
                     │
                     ▼
        ┌─────────────────────────┐
        │    Isolation Forest     │
        │   Anomaly Detection     │
        └────────────┬────────────┘
                     │
                     ▼
        ┌─────────────────────────┐
        │    Sensor Health Layer  │
        │                         │
        │  • Missing Data         │
        │  • Behavioral Anomaly   │
        └────────────┬────────────┘
                     │
                     ▼
        ┌─────────────────────────┐
        │    Fault Diagnosis      │
        │                         │
        │  • Sudden Spike         │
        │  • Gradual Drift        │
        │  • Progressive Degrad.  │
        │  • Missing Data         │
        └────────────┬────────────┘
                     │
                     ▼
        ┌─────────────────────────┐
        │   Streamlit Dashboard   │
        │                         │
        │  Sensor Health          │
        │  PM2.5 Trends           │
        │  Fault Analysis         │
        │  Anomaly Scores         │
        └─────────────────────────┘

        server deployed on 29-09-2026