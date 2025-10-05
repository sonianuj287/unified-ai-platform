import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
import plotly.express as px

st.set_page_config(page_title="🚨 Anomaly Detection Dashboard", layout="wide")
st.title("🔍 Enhanced Anomaly Detection & RCA Dashboard")

st.markdown("""
Upload a CSV file and select features for anomaly detection.
The app supports:
- **Isolation Forest**
- **Local Outlier Factor (LOF)**
- **One-Class SVM**
""")

uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Preview of Uploaded Data")
    st.write(df.head())

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        st.warning("No numeric columns found in your data.")
    else:
        with st.sidebar:
            st.header("⚙️ Model Configuration")
            algo = st.selectbox("Select Anomaly Detection Algorithm", 
                                ["Isolation Forest", "Local Outlier Factor", "One-Class SVM"])
            selected_features = st.multiselect("Select Features", numeric_cols, default=numeric_cols[:2])
            contamination = st.slider("Contamination (expected anomaly %)", 0.01, 0.2, 0.05)

        if st.button("🚀 Run Detection"):
            X = df[selected_features].values

            if algo == "Isolation Forest":
                model = IsolationForest(contamination=contamination, random_state=42)
                df["anomaly"] = model.fit_predict(X)
            elif algo == "Local Outlier Factor":
                model = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
                df["anomaly"] = model.fit_predict(X)
            else:
                model = OneClassSVM(nu=contamination, kernel="rbf", gamma="scale")
                df["anomaly"] = model.fit_predict(X)

            df["anomaly_flag"] = df["anomaly"].apply(lambda x: "Anomaly" if x == -1 else "Normal")

            # Summary
            num_anomalies = (df["anomaly_flag"] == "Anomaly").sum()
            st.success(f"✅ {num_anomalies} anomalies detected out of {len(df)} records "
                       f"({(num_anomalies/len(df))*100:.2f}% of data)")

            # Visualization
            feature = selected_features[0]
            if "time" in df.columns or "timestamp" in df.columns:
                time_col = "time" if "time" in df.columns else "timestamp"
                fig = px.line(df, x=time_col, y=feature, color="anomaly_flag",
                              title=f"Anomaly Detection over Time ({algo})",
                              color_discrete_map={"Normal": "blue", "Anomaly": "red"})
            else:
                fig = px.scatter(df, x=df.index, y=feature, color="anomaly_flag",
                                 title=f"Anomaly Detection on {feature} ({algo})",
                                 color_discrete_map={"Normal": "blue", "Anomaly": "red"})
            st.plotly_chart(fig, use_container_width=True)

            # Root Cause Analysis (only for Isolation Forest)
            if algo == "Isolation Forest":
                st.subheader("🧩 Root Cause Analysis (Feature Importance)")
                try:
                    importance = model.feature_importances_
                    feature_importance = pd.DataFrame({
                        "Feature": selected_features,
                        "Importance": importance
                    }).sort_values("Importance", ascending=False)
                    st.bar_chart(feature_importance.set_index("Feature"))
                except Exception as e:
                    st.info("Feature importance not available for this model.")
