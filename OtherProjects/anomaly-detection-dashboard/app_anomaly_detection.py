import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px

st.set_page_config(page_title="🔍 Anomaly Detection Dashboard", layout="wide")
st.title("🚨 Anomaly Detection & Root Cause Analysis Dashboard")

st.markdown("""
Upload a CSV file with numeric columns (e.g., sensor readings, metrics, etc.).
The app will:
1. Detect anomalies using **Isolation Forest**  
2. Highlight them visually  
3. Perform basic **Root Cause Analysis**
""")

uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Preview of Uploaded Data")
    st.write(df.head())

    # Select feature columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        st.warning("No numeric columns found in the uploaded file.")
    else:
        selected_features = st.multiselect("Select features for anomaly detection:", numeric_cols, default=numeric_cols[:2])

        contamination = st.slider("Select contamination (expected anomaly %):", 0.01, 0.2, 0.05)

        if st.button("🚀 Run Anomaly Detection"):
            model = IsolationForest(contamination=contamination, random_state=42)
            df["anomaly"] = model.fit_predict(df[selected_features])

            # -1 = anomaly
            df["anomaly_flag"] = df["anomaly"].apply(lambda x: "Anomaly" if x == -1 else "Normal")

            st.success("✅ Anomaly detection complete!")

            # Visualization (use first feature for simplicity)
            feature = selected_features[0]
            fig = px.scatter(
                df,
                x=df.index,
                y=feature,
                color="anomaly_flag",
                title=f"Anomaly Detection on {feature}",
                color_discrete_map={"Normal": "blue", "Anomaly": "red"}
            )
            st.plotly_chart(fig, use_container_width=True)

            # Root Cause Analysis
            st.subheader("🧩 Root Cause Analysis (Feature Importance)")
            try:
                importance = np.mean(np.abs(model.decision_function(df[selected_features].values.reshape(-1, len(selected_features)))), axis=0)
            except:
                importance = np.abs(model.decision_function(df[selected_features]))[:len(selected_features)]
            
            feature_importance = pd.DataFrame({
                "Feature": selected_features,
                "Importance": np.abs(model.decision_function(df[selected_features]))[:len(selected_features)]
            }).sort_values("Importance", ascending=False)

            st.write(feature_importance)
