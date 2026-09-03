"""
AI-Driven Anomaly Detection in Component Burn-In & Screening
Interactive QA Inspector Dashboard (Streamlit Web App)
ISRO Problem Statement ID: 26170
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from config import DATA_DIR, ROBUST_Z_SCORE_THRESHOLD, SAFETY_SLOPE_MARGIN_RATIO
from data.synthetic_generator import generate_synthetic_burnin_data
from pipeline import run_full_screening_pipeline
from engine.audit_logger import get_audit_history

# Page Configuration
st.set_page_config(
    page_title="ISRO Burn-In Screening AI System",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Custom CSS Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .status-pass {
        color: #16A34A;
        font-weight: bold;
        background-color: #DCFCE7;
        padding: 4px 10px;
        border-radius: 6px;
    }
    .status-review {
        color: #D97706;
        font-weight: bold;
        background-color: #FEF3C7;
        padding: 4px 10px;
        border-radius: 6px;
    }
    .status-flag {
        color: #DC2626;
        font-weight: bold;
        background-color: #FEE2E2;
        padding: 4px 10px;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown('<div class="main-header">🛰️ AI Component Burn-In Anomaly Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">ISRO Problem Statement ID: 26170 | Dynamic Peer-Relative Screening & 168h Time-Series Drift Prediction</div>', unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.image("https://img.icons8.com/color/96/satellite.png", width=64)
st.sidebar.title("Screening Settings")
st.sidebar.markdown("---")

z_thresh = st.sidebar.slider("Robust Z-Score Threshold (|Z|)", 1.5, 5.0, float(ROBUST_Z_SCORE_THRESHOLD), 0.1)
safety_ratio = st.sidebar.slider("Safety Margin Ratio (% of Limit)", 50, 95, int(SAFETY_SLOPE_MARGIN_RATIO * 100), 5) / 100.0

st.sidebar.markdown("---")
st.sidebar.info(
    "**System Status**: Model loaded & active.\n\n"
    "**Primary Goal**: Minimize false negatives (unnoticed defective component escapes)."
)

# Load / Select Dataset
@st.cache_data
def load_default_sample():
    csv_path = DATA_DIR / "sample_burnin_data.csv"
    if not csv_path.exists():
        df = generate_synthetic_burnin_data(save_path=str(csv_path))
    else:
        df = pd.read_csv(csv_path)
    return df

# Main Data Source Selection
data_source_opt = st.radio(
    "Select Test Dataset Source:",
    ["Use ISRO Benchmark Sample Dataset", "Upload Custom CSV Dataset"],
    horizontal=True
)

if data_source_opt == "Upload Custom CSV Dataset":
    uploaded_file = st.file_uploader("Upload Component Test CSV Data", type=["csv"])
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        dataset_name = uploaded_file.name
    else:
        st.warning("Please upload a CSV file or switch to the ISRO Benchmark Sample Dataset.")
        st.stop()
else:
    raw_df = load_default_sample()
    dataset_name = "ISRO Benchmark Sample Dataset"

# Execute Pipeline
with st.spinner("Processing component data through Dynamic Screening Pipeline..."):
    processed_df, summary = run_full_screening_pipeline(raw_df, dataset_name=dataset_name)

# Display Top KPI Metric Cards
col1, col2, col3, col4, col5 = st.columns(5)

total_count = summary["total_processed"]
counts = summary["screening_counts"]
perf = summary.get("performance_metrics", {})

col1.metric("Total Components", total_count)
col2.metric("PASS (Safe)", counts["PASS"], delta=f"{counts['PASS']/total_count:.1%}", delta_color="normal")
col3.metric("REVIEW (Attention)", counts["REVIEW"], delta=f"{counts['REVIEW']/total_count:.1%}", delta_color="off")
col4.metric("FLAG (High Risk)", counts["FLAG"], delta=f"-{counts['FLAG']/total_count:.1%}", delta_color="inverse")
if perf.get("has_ground_truth") and perf.get("mae") is not None:
    col5.metric("168h Predict MAE", f"{perf['mae']} µA")
else:
    col5.metric("168h Predict Engine", "Active")

st.markdown("---")

# Dashboard Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Component Screening Table",
    "🔍 Detail Inspector & Rationale",
    "📊 Peer Lot Distribution (Module A)",
    "📈 Trajectory & Drift (Module B)",
    "🎯 Model Evaluation & Audit Trail"
])

# TAB 1: Component Screening Table
with tab1:
    st.subheader("Component Screening Results")
    
    # Filter Controls
    f_col1, f_col2, f_col3 = st.columns([2, 2, 4])
    selected_status = f_col1.multiselect("Filter Status:", ["PASS", "REVIEW", "FLAG (HIGH RISK)"], default=["PASS", "REVIEW", "FLAG (HIGH RISK)"])
    selected_lot = f_col2.multiselect("Filter Lot ID:", processed_df["Lot_ID"].unique().tolist(), default=processed_df["Lot_ID"].unique().tolist())
    search_comp = f_col3.text_input("Search Component ID:", "")
    
    filtered_df = processed_df[
        processed_df["Screening_Status"].isin(selected_status) &
        processed_df["Lot_ID"].isin(selected_lot)
    ]
    if search_comp:
        filtered_df = filtered_df[filtered_df["Component_ID"].str.contains(search_comp, case=False)]
        
    display_cols = [
        "Component_ID", "Lot_ID", "Screening_Status", "Risk_Score",
        "Value_0h", "Value_24h", "Value_24h_Robust_Z",
        "Predicted_Value_168h", "Datasheet_Limit", "Explanation_Summary"
    ]
    
    st.dataframe(
        filtered_df[display_cols].sort_values(by="Risk_Score", ascending=False),
        use_container_width=True,
        hide_index=True
    )

# TAB 2: Detail Inspector & Rationale
with tab2:
    st.subheader("QA Component Detail & Auditable Rationale")
    
    comp_list = processed_df["Component_ID"].tolist()
    selected_cid = st.selectbox("Select Component to Inspect:", comp_list)
    
    comp_row = processed_df[processed_df["Component_ID"] == selected_cid].iloc[0]
    
    c_status = comp_row["Screening_Status"]
    c_score = comp_row["Risk_Score"]
    
    # Status Banner
    if c_status == "FLAG (HIGH RISK)":
        st.error(f"**STATUS: {c_status}** (Risk Score: {c_score}/100)")
    elif c_status == "REVIEW":
        st.warning(f"**STATUS: {c_status}** (Risk Score: {c_score}/100)")
    else:
        st.success(f"**STATUS: {c_status}** (Risk Score: {c_score}/100)")
        
    m_col1, m_col2 = st.columns(2)
    
    with m_col1:
        st.markdown("### 📝 Component Metadata")
        st.write(f"**Component ID**: {comp_row['Component_ID']}")
        st.write(f"**Manufacturing Lot**: {comp_row['Lot_ID']}")
        st.write(f"**Measured Parameter**: {comp_row['Parameter']}")
        st.write(f"**Datasheet Limit**: {comp_row['Datasheet_Limit']} µA")
        st.write(f"**Safety Threshold (80%)**: {comp_row['Safety_Threshold_168h']} µA")
        st.write(f"**24h Robust Z-Score**: {comp_row['Value_24h_Robust_Z']:+.2f}")
        st.write(f"**Predicted 168h Value**: {comp_row['Predicted_Value_168h']} µA")

    with m_col2:
        st.markdown("### 🕵️ Auditable QA Explanation Cards")
        explanations = comp_row["Explanation_List"]
        for idx, exp in enumerate(explanations, 1):
            st.info(f"**Evidence #{idx}**: {exp}")

# TAB 3: Peer Lot Distribution (Module A)
with tab3:
    st.subheader("Module A — Dynamic Peer Population Analysis")
    st.markdown("Identifies statistically anomalous components relative to their manufacturing lot baseline.")
    
    fig_box = px.box(
        processed_df,
        x="Lot_ID",
        y="Value_24h",
        color="Screening_Status",
        points="all",
        hover_data=["Component_ID", "Value_24h_Robust_Z"],
        title="24-Hour Parameter Distribution by Lot",
        labels={"Value_24h": "24h Measurement (µA)", "Lot_ID": "Lot Identifier"},
        color_discrete_map={"PASS": "#16A34A", "REVIEW": "#D97706", "FLAG (HIGH RISK)": "#DC2626"}
    )
    st.plotly_chart(fig_box, use_container_width=True)

    fig_z = px.scatter(
        processed_df,
        x="Value_24h",
        y="Value_24h_Robust_Z",
        color="Screening_Status",
        hover_data=["Component_ID", "Lot_ID"],
        title="24h Robust Z-Score vs. Absolute Value (Peer Relative)",
        color_discrete_map={"PASS": "#16A34A", "REVIEW": "#D97706", "FLAG (HIGH RISK)": "#DC2626"}
    )
    fig_z.add_hline(y=z_thresh, line_dash="dash", line_color="red", annotation_text=f"+{z_thresh} Z-Threshold")
    fig_z.add_hline(y=-z_thresh, line_dash="dash", line_color="red", annotation_text=f"-{z_thresh} Z-Threshold")
    st.plotly_chart(fig_z, use_container_width=True)

# TAB 4: Trajectory & Drift (Module B)
with tab4:
    st.subheader("Module B — Time-Series Drift & 168h Prediction")
    
    # Plot Trajectories
    selected_comp_traj = st.multiselect(
        "Select Components to Plot Time-Series Trajectory:",
        processed_df["Component_ID"].tolist(),
        default=processed_df[processed_df["Screening_Status"] != "PASS"]["Component_ID"].head(5).tolist()
    )
    
    if selected_comp_traj:
        fig_traj = go.Figure()
        
        for cid in selected_comp_traj:
            row = processed_df[processed_df["Component_ID"] == cid].iloc[0]
            
            # Measured time points
            x_meas = [0, 24]
            y_meas = [row["Value_0h"], row["Value_24h"]]
            
            if "Value_96h" in row and pd.notnull(row["Value_96h"]):
                x_meas.append(96)
                y_meas.append(row["Value_96h"])
            if "Value_168h" in row and pd.notnull(row["Value_168h"]):
                x_meas.append(168)
                y_meas.append(row["Value_168h"])
                
            fig_traj.add_trace(go.Scatter(
                x=x_meas, y=y_meas,
                mode="lines+markers",
                name=f"{cid} (Actual)",
                line=dict(width=2)
            ))
            
            # Predicted 168h point
            fig_traj.add_trace(go.Scatter(
                x=[24, 168],
                y=[row["Value_24h"], row["Predicted_Value_168h"]],
                mode="lines+markers",
                name=f"{cid} (Predicted 168h)",
                line=dict(dash="dash", width=2)
            ))
            
        fig_traj.add_hline(y=processed_df["Datasheet_Limit"].iloc[0], line_color="darkred", annotation_text="Datasheet Limit (50 µA)")
        fig_traj.add_hline(y=processed_df["Datasheet_Limit"].iloc[0] * safety_ratio, line_color="orange", annotation_text=f"Safety Margin ({int(safety_ratio*100)}%)")
        
        fig_traj.update_layout(
            title="Component Parameter Trajectory & 168h Prediction",
            xaxis_title="Burn-In Hours",
            yaxis_title="Parameter Value (µA)",
            hovermode="x unified"
        )
        st.plotly_chart(fig_traj, use_container_width=True)

# TAB 5: Model Evaluation & Audit Trail
with tab5:
    st.subheader("Model Evaluation & Audit Compliance")
    
    eval_c1, eval_c2 = st.columns(2)
    
    with eval_c1:
        st.markdown("### 📊 Model Prediction Performance")
        if perf.get("has_ground_truth"):
            st.write(f"**Mean Absolute Error (MAE)**: {perf['mae']} µA")
            st.write(f"**Root Mean Squared Error (RMSE)**: {perf['rmse']} µA")
            st.write(f"**R² Score**: {perf['r2']}")
            st.write(f"**Evaluated Components**: {perf['total_samples_evaluated']}")
        else:
            st.info("Ground truth 168h measurements not provided in dataset. Performance metrics will update when complete burn-in data is uploaded.")

    with eval_c2:
        st.markdown("### 📜 Audit Log History")
        history = get_audit_history(limit=5)
        if history:
            for entry in reversed(history):
                st.code(
                    f"Time: {entry['timestamp']}\n"
                    f"Dataset: {entry['dataset']}\n"
                    f"Screening: PASS={entry['summary']['PASS']} | REVIEW={entry['summary']['REVIEW']} | FLAG={entry['summary']['FLAG']}\n"
                    f"Thresholds: Z={entry['configured_thresholds']['robust_z_score_threshold']}, Safety={entry['configured_thresholds']['safety_margin_ratio']}"
                )
        else:
            st.write("No audit entries recorded yet.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #94A3B8; font-size: 0.85rem;'>"
    "AI-Driven Anomaly Detection System for Aerospace Component Screening | Powered by Antigravity IDE"
    "</div>",
    unsafe_allow_html=True
)
