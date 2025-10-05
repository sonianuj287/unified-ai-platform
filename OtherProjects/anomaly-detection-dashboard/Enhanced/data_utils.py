# data_utils.py
import numpy as np
import pandas as pd

def generate_sine_data(length=10000, seq_len=50, noise=0.05, anomaly_ratio=0.01, random_seed=42):
    """
    Generate sine wave with occasional anomalies (spikes).
    Returns a DataFrame with columns ['timestamp', 'value'].
    """
    np.random.seed(random_seed)
    t = np.arange(length)
    values = np.sin(0.02 * t) + 0.5 * np.sin(0.005 * t)  # mix of frequencies
    values += np.random.normal(scale=noise, size=length)

    # inject anomalies (spikes)
    n_anom = int(length * anomaly_ratio)
    anom_indices = np.random.choice(length, n_anom, replace=False)
    values[anom_indices] += np.random.uniform(3, 6, size=n_anom)  # big spikes

    df = pd.DataFrame({"timestamp": t, "value": values})
    df["is_anomaly"] = 0
    df.loc[anom_indices, "is_anomaly"] = 1
    return df
