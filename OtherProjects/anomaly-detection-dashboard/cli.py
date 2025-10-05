import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Load sample CSV
df = pd.read_csv("dataset.csv")  # columns: timestamp, value

# Fit Isolation Forest
model = IsolationForest(contamination=0.05, random_state=42)
df['anomaly'] = model.fit_predict(df[['value']])

# -1 = anomaly, 1 = normal
anomalies = df[df['anomaly'] == -1]

# Plot
plt.figure(figsize=(12,6))
plt.plot(df['timestamp'], df['value'], label='Value')
plt.scatter(anomalies['timestamp'], anomalies['value'], color='red', label='Anomaly')
plt.xlabel("Time")
plt.ylabel("Sensor Value")
plt.legend()
plt.show()
