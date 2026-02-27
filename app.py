from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import json
import matplotlib.pyplot as plt

from preprocess import basic_clean

st.set_page_config(page_title="Fraud Risk Scoring", layout="wide")


@st.cache_resource
def load_artifacts(path: str = "model.joblib"):
    return joblib.load(path)

@st.cache_data
def load_metrics(path: str = "metrics.json") -> dict:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

metrics = load_metrics()

with st.expander("Model card (training metrics)", expanded=True):
    roc = metrics.get("roc_auc", None)
    fraud_rate = metrics.get("fraud_rate", None)
    rec_thr = metrics.get("recommended_threshold", None)

    cols = st.columns(3)
    cols[0].metric("ROC-AUC", f"{roc:.3f}" if roc is not None else "N/A")
    cols[1].metric("Fraud rate (train)", f"{fraud_rate*100:.2f}%" if fraud_rate is not None else "N/A")
    cols[2].metric("Recommended threshold", f"{rec_thr:.2f}" if rec_thr is not None else "N/A")

    st.caption(
        "Because fraud is rare, threshold tuning is used to trade off precision vs recall. "
        "Use the slider in the sidebar to adjust how aggressively records are flagged."
    )


def risk_histogram(scores: pd.Series, bins: int = 20) -> pd.DataFrame:
    # Creates a simple histogram table that st.bar_chart can display
    counts, edges = np.histogram(scores.dropna().values, bins=bins, range=(0.0, 1.0))
    labels = [f"{edges[i]:.2f}–{edges[i+1]:.2f}" for i in range(len(counts))]
    return pd.DataFrame({"bin": labels, "count": counts}).set_index("bin")


st.title("Fraud Risk Scoring App")
st.markdown(
"""
**Project Summary:**  
This model was trained on ~460,000 electricity customer records with a fraud rate of 1.23%.  
Due to severe class imbalance, threshold tuning is used to optimize fraud recall while maintaining precision.
"""
)
st.caption("Upload a CSV, score records, and rank the highest-risk accounts. Built for imbalanced fraud detection.")

artifacts = load_artifacts()
pipe = artifacts["pipeline"]
columns_seen = artifacts["columns_seen"]

st.subheader("Top drivers of fraud risk (feature importance)")

fi = artifacts.get("feature_importances", None)
fn = artifacts.get("feature_names", None)

if fi is None or fn is None:
    st.info("Feature importance not available. Retrain to save importances and feature names.")
else:
    # Build a top-N importance table
    imp_df = pd.DataFrame({"feature": fn, "importance": fi}).sort_values("importance", ascending=False)
    top_n = st.slider("Top features to display", 5, 30, 15)

    st.dataframe(imp_df.head(top_n), use_container_width=True)

    # Simple horizontal bar chart
    plot_df = imp_df.head(top_n).iloc[::-1]  # reverse for nicer bars
    fig, ax = plt.subplots()
    ax.barh(plot_df["feature"], plot_df["importance"])
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    st.pyplot(fig, clear_figure=True)

# --- Sidebar controls ---
st.sidebar.header("Controls")
threshold = st.sidebar.slider("Fraud Risk Threshold", 0.05, 0.90, 0.30, 0.05)
top_k = st.sidebar.slider("Show top K high-risk", 10, 200, 50)
show_schema_debug = st.sidebar.checkbox("Show column alignment details", value=False)

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV to begin. Tip: use the same feature columns the model was trained on.")
    st.stop()

df = pd.read_csv(uploaded, low_memory=False)

st.subheader("Preview")
st.dataframe(df.head(20), use_container_width=True)

# Prepare features (assume inference CSV does NOT include the target)
X = basic_clean(df.copy())

# Align to training columns
missing = [c for c in columns_seen if c not in X.columns]
extra = [c for c in X.columns if c not in columns_seen]

# Reindex to the exact training column order; missing columns become NaN (imputer in pipeline handles this)
X_aligned = X.reindex(columns=columns_seen)

if not hasattr(pipe, "predict_proba"):
    st.error("Model does not support predict_proba; retrain with a probabilistic classifier.")
    st.stop()

proba = pipe.predict_proba(X_aligned)[:, 1]
preds = (proba >= threshold).astype(int)

out = df.copy()
out["fraud_risk_score"] = proba
out["predicted_fraud"] = preds

# --- KPI row ---
flagged = int(out["predicted_fraud"].sum())
total = len(out)
pct_flagged = (flagged / total) * 100 if total else 0.0

c1, c2, c3 = st.columns(3)
c1.metric("Total records", f"{total:,}")
c2.metric("Flagged high-risk", f"{flagged:,}")
c3.metric("% flagged", f"{pct_flagged:.2f}%")

# Optional debug info (keeps demo clean)
if show_schema_debug:
    st.markdown("### Column alignment details")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Missing columns (expected from training):**")
        st.write(missing[:100] if missing else "None")
    with col2:
        st.write("**Extra columns (ignored):**")
        st.write(extra[:100] if extra else "None")

# --- Charts + table ---
left, right = st.columns([1, 1])

with left:
    st.subheader("Risk score distribution")
    hist_df = risk_histogram(pd.Series(out["fraud_risk_score"]), bins=20)
    st.bar_chart(hist_df, use_container_width=True)

with right:
    st.subheader("Top high-risk records")
    st.dataframe(
        out.sort_values("fraud_risk_score", ascending=False).head(top_k),
        use_container_width=True,
    )

st.download_button(
    label="Download scored CSV",
    data=out.to_csv(index=False).encode("utf-8"),
    file_name="scored.csv",
    mime="text/csv",
)