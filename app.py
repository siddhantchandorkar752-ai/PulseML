import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from src.data_loader import prepare_data
from src.model import train_model, evaluate_model, save_model, load_model
from src.monitor import run_drift_report, run_performance_report, load_data
from src.alerts import check_drift_alert, check_performance_alert

st.set_page_config(page_title="PulseML", page_icon="🔬", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #0d1117 50%, #0a0f1e 100%);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.2);
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #8b5cf6, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }

    .hero-sub {
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 2rem;
        letter-spacing: 0.05em;
    }

    .metric-card {
        background: linear-gradient(135deg, rgba(99,102,241,0.08), rgba(139,92,246,0.05));
        border: 1px solid rgba(99,102,241,0.25);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        border-color: rgba(99,102,241,0.6);
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(99,102,241,0.15);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #a78bfa;
    }

    .metric-label {
        font-size: 0.75rem;
        color: #64748b;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-top: 0.25rem;
    }

    .alert-danger {
        background: linear-gradient(135deg, rgba(239,68,68,0.1), rgba(220,38,38,0.05));
        border: 1px solid rgba(239,68,68,0.4);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        color: #fca5a5;
        font-weight: 500;
    }

    .alert-success {
        background: linear-gradient(135deg, rgba(16,185,129,0.1), rgba(5,150,105,0.05));
        border: 1px solid rgba(16,185,129,0.4);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        color: #6ee7b7;
        font-weight: 500;
    }

    .section-header {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #6366f1;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(99,102,241,0.2);
    }

    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.2rem !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
        margin-bottom: 0.5rem !important;
    }

    .stButton > button:hover {
        opacity: 0.85 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 20px rgba(99,102,241,0.4) !important;
    }

    div[data-testid="metric-container"] {
        background: rgba(99,102,241,0.05);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 12px;
        padding: 1rem;
    }

    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
    }

    .badge-live {
        background: rgba(16,185,129,0.15);
        color: #10b981;
        border: 1px solid rgba(16,185,129,0.3);
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 1rem 0;'>
        <div style='font-size:1.4rem; font-weight:700; background: linear-gradient(135deg, #6366f1, #8b5cf6);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>⚡ PulseML</div>
        <div style='color:#475569; font-size:0.75rem; margin-top:0.25rem;'>ML Monitoring Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Pipeline</div>", unsafe_allow_html=True)
    run_prepare = st.button("📦  Prepare Data")
    run_train   = st.button("🧠  Train Model")
    run_monitor = st.button("📊  Run Monitor")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Info</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='color:#64748b; font-size:0.8rem; line-height:1.8;'>
        <b style='color:#94a3b8;'>Dataset:</b> IEEE-CIS Fraud<br>
        <b style='color:#94a3b8;'>Model:</b> XGBoost<br>
        <b style='color:#94a3b8;'>Monitoring:</b> Evidently AI<br>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='color:#475569; font-size:0.72rem; line-height:1.8;'>
        ① Prepare Data<br>② Train Model<br>③ Run Monitor
    </div>
    """, unsafe_allow_html=True)

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom:2rem;'>
    <div class='hero-title'>PulseML</div>
    <div class='hero-sub'>⚡ Real-time ML Monitoring · Data Drift Detection · Model Performance · IEEE-CIS Fraud Detection
        <span class='status-badge badge-live' style='margin-left:0.75rem;'>● LIVE</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Prepare Data ─────────────────────────────────────────────────────────────
if run_prepare:
    with st.spinner("Loading & preprocessing IEEE-CIS dataset..."):
        try:
            reference, production = prepare_data()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""<div class='metric-card'>
                    <div class='metric-value'>{len(reference):,}</div>
                    <div class='metric-label'>Reference Rows</div>
                </div>""", unsafe_allow_html=True)
            with col2:
                st.markdown(f"""<div class='metric-card'>
                    <div class='metric-value'>{len(production):,}</div>
                    <div class='metric-label'>Production Rows</div>
                </div>""", unsafe_allow_html=True)
            with col3:
                st.markdown(f"""<div class='metric-card'>
                    <div class='metric-value'>{reference.shape[1]}</div>
                    <div class='metric-label'>Features</div>
                </div>""", unsafe_allow_html=True)
            st.success("✅ Data prepared successfully!")
        except Exception as e:
            st.error(f"❌ {e}")

# ── Train Model ───────────────────────────────────────────────────────────────
if run_train:
    with st.spinner("Training XGBoost on reference data..."):
        try:
            reference, _ = load_data()
            model = train_model(reference)
            metrics = evaluate_model(model, reference)
            save_model(model)

            st.markdown("<div class='section-header'>Model Performance</div>", unsafe_allow_html=True)
            cols = st.columns(5)
            labels = ["Accuracy", "F1 Score", "Precision", "Recall", "ROC AUC"]
            keys   = ["accuracy", "f1_score", "precision", "recall", "roc_auc"]
            for col, label, key in zip(cols, labels, keys):
                with col:
                    st.markdown(f"""<div class='metric-card'>
                        <div class='metric-value'>{metrics[key]}</div>
                        <div class='metric-label'>{label}</div>
                    </div>""", unsafe_allow_html=True)
            st.success("✅ Model trained & saved!")
        except Exception as e:
            st.error(f"❌ {e}")

# ── Run Monitor ───────────────────────────────────────────────────────────────
if run_monitor:
    with st.spinner("Running drift & performance analysis..."):
        try:
            reference, production = load_data()
            model = load_model()
            drift_results = run_drift_report(reference, production)
            perf_results  = run_performance_report(reference, production, model)
            drift_alert   = check_drift_alert(drift_results)
            perf_alert    = check_performance_alert(perf_results)

            # Alerts
            st.markdown("<div class='section-header'>System Alerts</div>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                cls = "alert-danger" if drift_alert["alert"] else "alert-success"
                icon = "🔴" if drift_alert["alert"] else "🟢"
                st.markdown(f"<div class='{cls}'>{icon} <b>Drift:</b> {drift_alert['message']}</div>", unsafe_allow_html=True)
            with col2:
                cls = "alert-danger" if perf_alert["alert"] else "alert-success"
                icon = "🔴" if perf_alert["alert"] else "🟢"
                st.markdown(f"<div class='{cls}'>{icon} <b>Performance:</b> {perf_alert['message']}</div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Drift Metrics
            st.markdown("<div class='section-header'>Drift Analysis</div>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""<div class='metric-card'>
                    <div class='metric-value' style='color:{"#ef4444" if drift_results["dataset_drifted"] else "#10b981"};'>
                        {drift_results["drift_share"]}</div>
                    <div class='metric-label'>Drift Share</div>
                </div>""", unsafe_allow_html=True)
            with col2:
                st.markdown(f"""<div class='metric-card'>
                    <div class='metric-value' style='color:{"#ef4444" if drift_results["dataset_drifted"] else "#10b981"};'>
                        {"YES" if drift_results["dataset_drifted"] else "NO"}</div>
                    <div class='metric-label'>Dataset Drifted</div>
                </div>""", unsafe_allow_html=True)
            with col3:
                st.markdown(f"""<div class='metric-card'>
                    <div class='metric-value' style='color:#a78bfa;'>{drift_alert["threshold"]}</div>
                    <div class='metric-label'>Drift Threshold</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Performance Chart
            st.markdown("<div class='section-header'>Performance Comparison</div>", unsafe_allow_html=True)
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"""<div class='metric-card' style='margin-bottom:1rem;'>
                    <div class='metric-value' style='color:#10b981;'>{perf_results["reference_f1"]}</div>
                    <div class='metric-label'>Reference F1</div>
                </div>""", unsafe_allow_html=True)
                st.markdown(f"""<div class='metric-card' style='margin-bottom:1rem;'>
                    <div class='metric-value' style='color:#f59e0b;'>{perf_results["current_f1"]}</div>
                    <div class='metric-label'>Current F1</div>
                </div>""", unsafe_allow_html=True)
                st.markdown(f"""<div class='metric-card'>
                    <div class='metric-value' style='color:{"#ef4444" if perf_results["f1_drop"] > 0.05 else "#10b981"};'>
                        -{perf_results["f1_drop"]}</div>
                    <div class='metric-label'>F1 Drop</div>
                </div>""", unsafe_allow_html=True)
            with col2:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name="Reference", x=["F1 Score"], y=[perf_results["reference_f1"]],
                    marker=dict(color="#10b981", opacity=0.85),
                    width=0.3
                ))
                fig.add_trace(go.Bar(
                    name="Production", x=["F1 Score"], y=[perf_results["current_f1"]],
                    marker=dict(color="#f59e0b", opacity=0.85),
                    width=0.3
                ))
                fig.update_layout(
                    barmode="group",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#94a3b8", family="Inter"),
                    legend=dict(bgcolor="rgba(0,0,0,0)"),
                    margin=dict(l=20, r=20, t=30, b=20),
                    yaxis=dict(gridcolor="rgba(99,102,241,0.1)", range=[0, 1]),
                    xaxis=dict(gridcolor="rgba(99,102,241,0.1)"),
                    height=280,
                )
                st.plotly_chart(fig, use_container_width=True)

            # Drift gauge
            st.markdown("<div class='section-header'>Drift Gauge</div>", unsafe_allow_html=True)
            fig2 = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=drift_results["drift_share"] * 100,
                delta={"reference": 15, "suffix": "%"},
                title={"text": "Drift Share %", "font": {"color": "#94a3b8", "family": "Inter"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#475569"},
                    "bar": {"color": "#6366f1"},
                    "bgcolor": "rgba(0,0,0,0)",
                    "steps": [
                        {"range": [0, 15], "color": "rgba(16,185,129,0.15)"},
                        {"range": [15, 50], "color": "rgba(245,158,11,0.15)"},
                        {"range": [50, 100], "color": "rgba(239,68,68,0.15)"},
                    ],
                    "threshold": {"line": {"color": "#ef4444", "width": 2}, "value": 15},
                },
                number={"suffix": "%", "font": {"color": "#a78bfa", "family": "Inter"}},
            ))
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8"),
                height=300,
                margin=dict(l=40, r=40, t=40, b=20),
            )
            st.plotly_chart(fig2, use_container_width=True)

            st.markdown("<div class='section-header'>Download Reports</div>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            col1.markdown(f"[📥 Drift Report]({drift_results['report_path']})")
            col2.markdown(f"[📥 Performance Report]({perf_results['report_path']})")

        except Exception as e:
            st.error(f"❌ {e}")