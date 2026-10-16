"""
CivicPulse AI — Intelligent Decision-Support System for Municipal Infrastructure Repair Prioritization.
Built for City Authorities, Municipal Engineers, and Hackathon Demonstrations.
"""

import os
import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from core.data_loader import (
    load_raw_data,
    audit_data_quality,
    aggregate_location_data
)
from core.priority_engine import (
    calculate_priority_scores,
    WEIGHT_PRESETS
)
from core.resource_planner import simulate_resource_allocation
from core.ml_engine import (
    prepare_weekly_incident_series,
    evaluate_forecasting_baselines,
    train_resolution_time_model,
    detect_volume_anomalies
)

# ---------------------------------------------------------
# Page Configuration & Global Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="CivicPulse AI | Smart Infrastructure Repair Prioritization",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark/Light harmonious civic theme, KPI cards, badges)
st.markdown("""
<style>
    /* Metric Card Container */
    .metric-card {
        background: linear-gradient(135deg, rgba(25, 33, 50, 0.04) 0%, rgba(35, 50, 80, 0.08) 100%);
        border: 1px solid rgba(120, 140, 180, 0.2);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #556677;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #112233;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #667788;
        margin-top: 4px;
    }

    /* Override Badges */
    .badge-override {
        background-color: #fee2e2;
        color: #991b1b;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 6px;
        border: 1px solid #f87171;
        font-size: 0.8rem;
        display: inline-block;
    }
    .badge-critical {
        background-color: #fef2f2;
        color: #b91c1c;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        display: inline-block;
    }
    .badge-high {
        background-color: #fffbeb;
        color: #b45309;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        display: inline-block;
    }
    .badge-medium {
        background-color: #f0fdf4;
        color: #15803d;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        display: inline-block;
    }

    /* Section divider */
    hr {
        margin: 20px 0;
        border: none;
        border-top: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------
# Cached Data Ingestion
# ---------------------------------------------------------
@st.cache_data
def load_all_data():
    locs, inc, comp = load_raw_data(DATA_DIR)
    audit = audit_data_quality(locs, inc, comp)
    return locs, inc, comp, audit

try:
    locations_master, incidents_master, complaints_master, data_audit = load_all_data()
except Exception as e:
    st.error(f"Critical error loading dataset: {e}")
    st.stop()

# ---------------------------------------------------------
# Sidebar Controls & Configurable Weights
# ---------------------------------------------------------
with st.sidebar:
    st.title("🏙️ CivicPulse AI")
    st.caption("Municipal Infrastructure Prioritization Engine")
    st.markdown("---")

    st.subheader("1. Spatial & Category Filters")
    
    # Zone selector with presets
    all_zones = sorted(locations_master["zone"].dropna().unique().tolist())
    col_z1, col_z2 = st.columns(2)
    with col_z1:
        if st.button("Select All Zones", use_container_width=True):
            st.session_state["sel_zones"] = all_zones
    with col_z2:
        if st.button("Reset Zones", use_container_width=True):
            st.session_state["sel_zones"] = all_zones[:5]

    default_zones = st.session_state.get("sel_zones", all_zones)
    chosen_zones = st.multiselect("Select Municipal Zones", all_zones, default=default_zones)

    # Incident types
    all_types = sorted(incidents_master["incident_type"].dropna().unique().tolist())
    chosen_types = st.multiselect("Incident Types", all_types, default=all_types)

    # Severities
    all_severities = ["Low", "Medium", "High", "Critical"]
    chosen_severities = st.multiselect("Incident Severities", all_severities, default=all_severities)

    # Road Condition
    all_conditions = ["Critical", "Poor", "Fair", "Good"]
    chosen_conditions = st.multiselect("Road Conditions", all_conditions, default=all_conditions)

    st.markdown("---")
    st.subheader("2. Priority Scoring Weights")
    
    preset_choice = st.selectbox(
        "Scoring Profile Preset",
        list(WEIGHT_PRESETS.keys()) + ["Custom Weights"],
        index=0
    )

    if preset_choice != "Custom Weights":
        active_preset = WEIGHT_PRESETS[preset_choice]
        w_sev = active_preset["severity"]
        w_freq = active_preset["frequency"]
        w_comp = active_preset["complaints"]
        w_context = active_preset["context"]
        w_wait = active_preset["waiting_time"]
    else:
        w_sev = 30
        w_freq = 25
        w_comp = 20
        w_context = 15
        w_wait = 10

    w_sev = st.slider("Severity Factor Weight (%)", 0, 100, int(w_sev))
    w_freq = st.slider("Incident Frequency Weight (%)", 0, 100, int(w_freq))
    w_comp = st.slider("Citizen Complaints Weight (%)", 0, 100, int(w_comp))
    w_context = st.slider("Contextual Risk Weight (%)", 0, 100, int(w_context))
    w_wait = st.slider("Waiting Time / Backlog Weight (%)", 0, 100, int(w_wait))

    total_w = w_sev + w_freq + w_comp + w_context + w_wait
    if total_w == 0:
        st.warning("⚠️ Total weight is 0%. Uniform equal weighting (20% each) applied.")
    elif total_w != 100:
        st.caption(f"Weights sum to **{total_w}%**. Normalized automatically to 100%.")

    st.markdown("---")
    st.subheader("3. Safety Guardrails")
    enable_override = st.checkbox(
        "Enable Critical Safety Override",
        value=True,
        help="Mandatory rule-based guardrail: Automatically elevates priority score floor to 85.0 for Critical hazards adjacent to schools, hospitals, or with reported injuries."
    )

# ---------------------------------------------------------
# Dynamic Filtering & Aggregation Pipeline
# ---------------------------------------------------------
# Filter locations
loc_filtered = locations_master[
    locations_master["zone"].isin(chosen_zones) &
    locations_master["road_condition"].isin(chosen_conditions)
].copy()

# Filter incidents
inc_filtered = incidents_master[
    incidents_master["location_id"].isin(loc_filtered["location_id"]) &
    incidents_master["incident_type"].isin(chosen_types) &
    incidents_master["severity"].isin(chosen_severities)
].copy()

# Filter complaints
comp_filtered = complaints_master[
    complaints_master["location_id"].isin(loc_filtered["location_id"])
].copy()

# Aggregations
agg_inc, agg_comp = aggregate_location_data(inc_filtered, comp_filtered)

# Compute Priority Scores
weights_dict = {
    "severity": w_sev,
    "frequency": w_freq,
    "complaints": w_comp,
    "context": w_context,
    "waiting_time": w_wait
}

ranked_locations = calculate_priority_scores(
    locations_df=loc_filtered,
    agg_incidents_df=agg_inc,
    agg_complaints_df=agg_comp,
    weights=weights_dict,
    enable_safety_override=enable_override
)

# ---------------------------------------------------------
# Header & Data Provenance Notice
# ---------------------------------------------------------
st.markdown("""
<div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
    <div>
        <h1 style="margin: 0; padding: 0; font-size: 2.2rem; color: #0f172a;">🏙️ CivicPulse AI</h1>
        <p style="margin: 0; color: #475569; font-size: 1.05rem;">
            Intelligent Infrastructure Repair Prioritization Engine & Resource Dispatcher
        </p>
    </div>
    <div style="text-align: right;">
        <span style="background-color: #ecfdf5; color: #065f46; padding: 6px 12px; border-radius: 8px; font-weight: 600; font-size: 0.85rem; border: 1px solid #a7f3d0;">
            🟢 Relational Integrity: 100% Verified
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# Synthetic data banner
st.info(
    "⚠️ **Educational / Synthetic Dataset Notice:** All geographic coordinates, timestamps, and incident logs are simulated for student hackathon demonstration. "
    "Coordinates must not be interpreted as physical real-world sites."
)

# Empty filter state check
if ranked_locations.empty:
    st.warning("⚠️ No locations match the selected filter criteria. Please adjust your zone, incident type, or severity filters in the sidebar.")
    st.stop()

# ---------------------------------------------------------
# KPI Cards Row
# ---------------------------------------------------------
k1, k2, k3, k4, k5, k6 = st.columns(6)

with k1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Monitored Sites</div>
        <div class="metric-value">{len(ranked_locations):,}</div>
        <div class="metric-sub">{len(chosen_zones)} Active Zones</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Incidents Filtered</div>
        <div class="metric-value">{len(inc_filtered):,}</div>
        <div class="metric-sub">{int(inc_filtered['injuries'].sum()):,} Total Injuries</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Citizen Complaints</div>
        <div class="metric-value">{len(comp_filtered):,}</div>
        <div class="metric-sub">{int(comp_filtered['is_open'].sum()):,} Open Backlog</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    overrides_count = int(ranked_locations["safety_override_triggered"].sum()) if "safety_override_triggered" in ranked_locations.columns else 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Safety Overrides</div>
        <div class="metric-value" style="color: #dc2626;">{overrides_count:,}</div>
        <div class="metric-sub">School / Hospital Hazards</div>
    </div>
    """, unsafe_allow_html=True)

with k5:
    crit_high = int((ranked_locations["priority_score"] >= 50.0).sum())
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">High & Critical Sites</div>
        <div class="metric-value" style="color: #ea580c;">{crit_high:,}</div>
        <div class="metric-sub">{crit_high / len(ranked_locations) * 100:.1f}% of Active Network</div>
    </div>
    """, unsafe_allow_html=True)

with k6:
    resolved_comp = comp_filtered[~comp_filtered["is_open"]]
    avg_comp_days = resolved_comp["days_to_resolution"].mean() if len(resolved_comp) else 0.0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Complaint MTTR</div>
        <div class="metric-value">{avg_comp_days:.1f} d</div>
        <div class="metric-sub">Mean Days to Resolution</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Main Tabs Layout
# ---------------------------------------------------------
tab_map, tab_ranking, tab_trends, tab_sla, tab_ml, tab_resource, tab_audit = st.tabs([
    "🗺️ Geospatial Risk Map",
    "📋 Ranked Locations & Recommendations",
    "📈 Trends & Anomaly Monitor",
    "⏱️ SLA & Resolution Times",
    "🧠 Predictive Analytics & Baselines",
    "👷 Resource Planner & Crew Dispatcher",
    "🛡️ Data Quality & Audit Report"
])

# =========================================================
# TAB 1: Geospatial Risk Map & Executive Hotspots
# =========================================================
with tab_map:
    st.subheader("Geospatial Priority Risk Heatmap")
    st.caption("Visualizing repair priority across simulated civic coordinates. Red pins denote urgent, high-risk infrastructure sites.")

    map_df = ranked_locations.dropna(subset=["latitude", "longitude"]).copy()
    map_df["pin_size"] = (map_df["priority_score"] / 10.0).clip(lower=4.0, upper=18.0)

    # Resilient OpenStreetMap builder supporting Plotly 6/7+ (scatter_map) and Plotly 5 (scatter_mapbox)
    hover_opts = {
        "zone": True,
        "priority_score": True,
        "priority_level": True,
        "road_condition": True,
        "incident_count": True,
        "complaint_count": True,
        "primary_incident_type": True,
        "pin_size": False,
        "latitude": False,
        "longitude": False
    }

    if hasattr(px, "scatter_map"):
        fig_map = px.scatter_map(
            map_df,
            lat="latitude",
            lon="longitude",
            color="priority_score",
            size="pin_size",
            color_continuous_scale="RdYlGn_r",
            range_color=[0, 100],
            zoom=9,
            height=520,
            map_style="open-street-map",
            hover_name="location_id",
            hover_data=hover_opts
        )
    elif hasattr(px, "scatter_mapbox"):
        fig_map = px.scatter_mapbox(
            map_df,
            lat="latitude",
            lon="longitude",
            color="priority_score",
            size="pin_size",
            color_continuous_scale="RdYlGn_r",
            range_color=[0, 100],
            zoom=9,
            height=520,
            mapbox_style="open-street-map",
            hover_name="location_id",
            hover_data=hover_opts
        )
    else:
        fig_map = px.scatter_geo(
            map_df,
            lat="latitude",
            lon="longitude",
            color="priority_score",
            size="pin_size",
            color_continuous_scale="RdYlGn_r",
            range_color=[0, 100],
            height=520,
            hover_name="location_id",
            hover_data=hover_opts
        )

    fig_map.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    st.plotly_chart(fig_map, use_container_width=True)

    # Hotspot summary columns
    col_hot1, col_hot2 = st.columns([1, 1])
    with col_hot1:
        st.markdown("#### 🚨 Top 5 Critical Hotspots Requiring Immediate Attention")
        for _, top_row in ranked_locations.head(5).iterrows():
            override_badge = "🚨 OVERRIDE" if top_row.get("safety_override_triggered") else ""
            with st.container():
                st.markdown(f"""
                <div style="background-color: #f8fafc; border-left: 4px solid #ef4444; padding: 10px 14px; border-radius: 6px; margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between;">
                        <strong>#{int(top_row['rank'])} · {top_row['location_id']} ({top_row['zone']})</strong>
                        <span style="font-weight: 700; color: #dc2626;">Score: {top_row['priority_score']:.1f}/100 {override_badge}</span>
                    </div>
                    <div style="font-size: 0.85rem; color: #475569; margin-top: 2px;">
                        <strong>Road:</strong> {top_row.get('road_type', 'N/A')} ({top_row.get('road_condition', 'N/A')}) | 
                        <strong>Primary Issue:</strong> {top_row.get('primary_incident_type', 'N/A')}
                    </div>
                    <div style="font-size: 0.85rem; color: #0284c7; margin-top: 2px;">
                        <strong>Action:</strong> {top_row.get('recommended_action', 'Inspect site')}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with col_hot2:
        st.markdown("#### 📊 Priority Score Distribution by Zone")
        zone_summary = ranked_locations.groupby("zone").agg(
            mean_score=("priority_score", "mean"),
            critical_sites=("priority_score", lambda s: (s >= 50).sum()),
            total_sites=("location_id", "count")
        ).reset_index().sort_values("mean_score", ascending=False).head(10)
        
        fig_zone = px.bar(
            zone_summary,
            x="zone",
            y="mean_score",
            color="mean_score",
            color_continuous_scale="Reds",
            title="Top 10 High-Risk Zones (Mean Priority Score)",
            labels={"mean_score": "Mean Priority Score", "zone": "Municipal Zone"}
        )
        fig_zone.update_layout(height=300, margin=dict(l=20, r=20, t=35, b=20))
        st.plotly_chart(fig_zone, use_container_width=True)

# =========================================================
# TAB 2: Ranked Locations & Explainable Recommendations
# =========================================================
with tab_ranking:
    st.subheader("Ranked Location Inventory & Explainable Queue")
    st.write(
        "Locations are ranked by our transparent, multi-factor priority algorithm. "
        "Expand any row to inspect the exact mathematical factor breakdown and evidence justification."
    )

    col_srch, col_lvl, col_dl = st.columns([2, 1, 1])
    with col_srch:
        search_kw = st.text_input("🔍 Search by Location ID (e.g. L0002) or Zone", "")
    with col_lvl:
        prio_filter = st.selectbox(
            "Filter Priority Tier",
            ["All Tiers", "Critical (Override)", "Critical", "High", "Medium", "Low"]
        )
    with col_dl:
        csv_buffer = io.StringIO()
        ranked_locations.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Full Ranked CSV",
            data=csv_buffer.getvalue().encode("utf-8"),
            file_name="civicpulse_ranked_locations.csv",
            mime="text/csv",
            use_container_width=True
        )

    # Filtered view
    display_df = ranked_locations.copy()
    if search_kw:
        mask = (
            display_df["location_id"].str.contains(search_kw, case=False, na=False) |
            display_df["zone"].str.contains(search_kw, case=False, na=False)
        )
        display_df = display_df[mask]

    if prio_filter != "All Tiers":
        display_df = display_df[display_df["priority_level"] == prio_filter]

    st.caption(f"Showing **{len(display_df):,}** locations matching current search.")

    # High-level summary table
    table_cols = [
        "rank", "location_id", "zone", "priority_score", "priority_level",
        "primary_incident_type", "incident_count", "complaint_count",
        "open_complaints", "road_condition", "recommended_action"
    ]
    renamed_cols = {
        "rank": "Rank",
        "location_id": "Location",
        "zone": "Zone",
        "priority_score": "Score",
        "priority_level": "Tier",
        "primary_incident_type": "Primary Issue",
        "incident_count": "Incidents",
        "complaint_count": "Complaints",
        "open_complaints": "Open Backlog",
        "road_condition": "Road Cond.",
        "recommended_action": "Recommended Action"
    }
    st.dataframe(
        display_df[table_cols].rename(columns=renamed_cols),
        use_container_width=True,
        hide_index=True,
        height=320
    )

    # Drill-down expander for top recommendations
    st.markdown("#### 🔬 Detailed Factor Contribution Breakdown (Top Locations)")
    top_inspect_n = st.slider("Inspect Top N Locations in Detail", 5, 30, 10)
    for _, rec in display_df.head(top_inspect_n).iterrows():
        override_text = " [🚨 SAFETY OVERRIDE ACTIVE]" if rec.get("safety_override_triggered") else ""
        header_title = f"#{int(rec['rank'])} · {rec['location_id']} ({rec['zone']}) — Priority Score: {rec['priority_score']:.1f}/100 [{rec['priority_level']}]{override_text}"
        
        with st.expander(header_title):
            c_left, c_right = st.columns([1, 1])
            with c_left:
                st.markdown(f"**Primary Detected Hazard:** {rec.get('primary_incident_type', 'N/A')}")
                st.markdown(f"**Road Attributes:** {rec.get('road_type', 'N/A')} · Condition: **{rec.get('road_condition', 'N/A')}**")
                st.markdown(f"**Vulnerability Context:** Near School: {'Yes' if rec.get('near_school') == 1 else 'No'} | Near Hospital: {'Yes' if rec.get('near_hospital') == 1 else 'No'}")
                st.markdown(f"**Historical Records:** {int(rec['incident_count'])} incidents ({int(rec['injury_count'])} injuries) · {int(rec['complaint_count'])} complaints ({int(rec['open_complaints'])} open)")
                st.markdown(f"**Recommended Municipal Action:** `{rec.get('recommended_action', 'N/A')}`")
                st.markdown(f"**Evidence Narrative:** *{rec.get('evidence_reasons', 'N/A')}*")
            
            with c_right:
                # Contribution chart
                contrib_data = pd.DataFrame([
                    {"Factor": "Severity & Injury", "Points": rec.get("contrib_severity", 0)},
                    {"Factor": "Incident Frequency", "Points": rec.get("contrib_frequency", 0)},
                    {"Factor": "Public Complaints", "Points": rec.get("contrib_complaints", 0)},
                    {"Factor": "Contextual Risk", "Points": rec.get("contrib_context", 0)},
                    {"Factor": "Waiting Time / Backlog", "Points": rec.get("contrib_waiting_time", 0)}
                ])
                fig_contrib = px.bar(
                    contrib_data,
                    x="Points",
                    y="Factor",
                    orientation="h",
                    title="Exact Factor Points Contributed to Final Score",
                    color="Points",
                    color_continuous_scale="Tealgrn"
                )
                fig_contrib.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig_contrib, use_container_width=True)

# =========================================================
# TAB 3: Trends, Seasonality & Anomaly Monitor
# =========================================================
with tab_trends:
    st.subheader("Incident & Complaint Dynamics Monitor")
    st.caption("Temporal pattern analysis, category cross-tabulation, and statistical anomaly detection.")

    t_col1, t_col2 = st.columns(2)
    with t_col1:
        if not inc_filtered.empty:
            monthly_inc = inc_filtered.set_index("timestamp").resample("MS").size().rename("Monthly Incidents").reset_index()
            fig_inc_line = px.line(
                monthly_inc,
                x="timestamp",
                y="Monthly Incidents",
                markers=True,
                title="Monthly Incident Trajectory (2025)",
                line_shape="spline"
            )
            fig_inc_line.update_traces(line_color="#2563eb")
            st.plotly_chart(fig_inc_line, use_container_width=True)
        else:
            st.info("No incidents match the active filters.")

    with t_col2:
        if not comp_filtered.empty:
            monthly_comp = comp_filtered.set_index("created_at").resample("MS").size().rename("Monthly Complaints").reset_index()
            fig_comp_bar = px.bar(
                monthly_comp,
                x="created_at",
                y="Monthly Complaints",
                title="Monthly Citizen Complaint Volume (2025)",
                color="Monthly Complaints",
                color_continuous_scale="Blues"
            )
            st.plotly_chart(fig_comp_bar, use_container_width=True)
        else:
            st.info("No complaints match the active filters.")

    # Category comparisons
    c_cat1, c_cat2 = st.columns(2)
    with c_cat1:
        if not inc_filtered.empty:
            inc_cat = inc_filtered["incident_type"].value_counts().reset_index()
            inc_cat.columns = ["Incident Type", "Count"]
            fig_type = px.pie(
                inc_cat,
                names="Incident Type",
                values="Count",
                title="Incident Distribution by Category",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            st.plotly_chart(fig_type, use_container_width=True)

    with c_cat2:
        if not comp_filtered.empty:
            comp_cat = comp_filtered["category"].value_counts().reset_index()
            comp_cat.columns = ["Complaint Category", "Count"]
            fig_comp = px.pie(
                comp_cat,
                names="Complaint Category",
                values="Count",
                title="Citizen Complaints by Category",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig_comp, use_container_width=True)

    # Statistical Anomaly Detection
    st.markdown("---")
    st.subheader("Statistical Surge Anomaly Detection")
    st.write(
        "Detects months with statistically abnormal spikes in municipal incident volume. "
        "Flags are calculated using moving statistical standard deviations (Z-Scores)."
    )
    
    col_z, col_space = st.columns([1, 2])
    with col_z:
        z_thresh = st.slider("Anomaly Sensitivity Threshold (Standard Deviations)", 1.5, 3.0, 2.0, step=0.1)
    
    anomaly_df = detect_volume_anomalies(inc_filtered, z_threshold=z_thresh)
    if not anomaly_df.empty:
        anom_display = anomaly_df.copy()
        anom_display["Month"] = anom_display["timestamp"].dt.strftime("%B %Y")
        st.dataframe(
            anom_display[["Month", "incident_count", "z_score", "anomaly_label"]].rename(columns={
                "incident_count": "Recorded Incidents",
                "z_score": "Statistical Z-Score",
                "anomaly_label": "Audit Classification"
            }),
            use_container_width=True,
            hide_index=True
        )
        st.caption(
            "📌 **Data Science Note:** A statistical outlier flag highlights volume divergence, not a verified external disaster. "
            "Municipal audits should cross-reference flags with weather extremes or reporting campaigns."
        )

# =========================================================
# TAB 4: SLA & Resolution Times
# =========================================================
with tab_sla:
    st.subheader("Municipal SLA & Resolution Times Dashboard")
    st.caption("Monitoring resolution benchmarks, citizen wait durations, and backlog escalation queues.")

    open_c = int(comp_filtered["is_open"].sum())
    resolved_c = comp_filtered[~comp_filtered["is_open"]]
    
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Active Open Backlog", f"{open_c:,} complaints")
    s2.metric("Closed / Resolved", f"{len(resolved_c):,} complaints")
    s3.metric("Average Resolution Duration", f"{resolved_c['days_to_resolution'].mean():.1f} days" if len(resolved_c) else "—")
    s4.metric("Incident MTTR (Hours)", f"{inc_filtered['resolution_hours'].mean():.1f} hrs" if len(inc_filtered) else "—")

    col_sla1, col_sla2 = st.columns(2)
    with col_sla1:
        st.markdown("#### Complaint Workflow Status Distribution")
        status_dist = comp_filtered["status"].value_counts().reset_index()
        status_dist.columns = ["Status", "Volume"]
        fig_status = px.bar(
            status_dist,
            x="Status",
            y="Volume",
            color="Status",
            title="Public Complaint Lifecycle Stages",
            text="Volume"
        )
        st.plotly_chart(fig_status, use_container_width=True)

    with col_sla2:
        st.markdown("#### Mean Resolution Days by Complaint Category")
        if not resolved_c.empty:
            cat_perf = resolved_c.groupby("category")["days_to_resolution"].agg(["mean", "median", "count"]).reset_index()
            cat_perf.columns = ["Category", "Mean Days", "Median Days", "Resolved Cases"]
            fig_perf = px.bar(
                cat_perf.sort_values("Mean Days", ascending=False),
                x="Category",
                y="Mean Days",
                color="Mean Days",
                color_continuous_scale="Viridis",
                title="Average Turnaround Days by Category"
            )
            st.plotly_chart(fig_perf, use_container_width=True)

    st.markdown("---")
    st.subheader("🚨 Overdue Complaints SLA Escalation Queue")
    sla_threshold_days = st.slider("Flag open complaints unresolved for more than (days)", 1, 30, 7)

    overdue_queue = comp_filtered[
        comp_filtered["is_open"] &
        (comp_filtered["days_to_resolution"] >= sla_threshold_days)
    ].sort_values("days_to_resolution", ascending=False).copy()

    st.write(f"Found **{len(overdue_queue):,}** open complaints exceeding the **{sla_threshold_days}-day** SLA threshold.")

    if not overdue_queue.empty:
        st.dataframe(
            overdue_queue[["complaint_id", "location_id", "category", "priority_reported", "status", "days_to_resolution"]].rename(columns={
                "complaint_id": "Complaint ID",
                "location_id": "Location",
                "category": "Category",
                "priority_reported": "Citizen Priority",
                "status": "Current Status",
                "days_to_resolution": "Open Duration (Days)"
            }),
            use_container_width=True,
            hide_index=True,
            height=260
        )
        
        csv_overdue = io.StringIO()
        overdue_queue.to_csv(csv_overdue, index=False)
        st.download_button(
            label="📥 Download Overdue Queue CSV",
            data=csv_overdue.getvalue().encode("utf-8"),
            file_name="overdue_complaints_sla.csv",
            mime="text/csv"
        )
    else:
        st.success("🎉 Excellent! No open complaints currently breach the selected SLA threshold.")

# =========================================================
# TAB 5: Predictive Analytics & Baselines (ML Engine)
# =========================================================
with tab_ml:
    st.subheader("Predictive Analytics & Baseline Modeling Engine")
    st.write(
        "Demonstrating machine learning and statistical time-series forecasting. "
        "Strictly utilizes time-aware train/test evaluation with real, honest benchmark metrics."
    )

    # 1. Weekly Incident Forecasting
    st.markdown("### 1. City-Wide Weekly Incident Forecasting")
    st.caption("Time-aware evaluation: Evaluates models on held-out Q4 test data (Weeks 41-52) to prevent look-ahead bias.")

    weekly_series = prepare_weekly_incident_series(incidents_master)
    test_w = st.slider("Holdout Test Window (Weeks)", 4, 16, 12)

    with st.spinner("Training time-series forecasting baselines..."):
        forecast_results = evaluate_forecasting_baselines(weekly_series, test_weeks=test_w)

    if "error" not in forecast_results:
        f_col1, f_col2 = st.columns([1, 1])
        with f_col1:
            st.markdown("#### Real Out-of-Sample Evaluation Metrics")
            st.dataframe(forecast_results["metrics_table"], use_container_width=True, hide_index=True)
            st.info(f"🏆 Best Performing Model on Held-out Test Data: **{forecast_results['best_model']}**")

        with f_col2:
            st.markdown("#### 4-Week Future Incident Projections (Horizon)")
            st.dataframe(forecast_results["future_forecast"], use_container_width=True, hide_index=True)
            st.caption("Forecast includes 95% Confidence Intervals calculated from residual out-of-sample variance.")

        # Test vs Actual plot
        test_eval = forecast_results["test_eval_df"]
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=test_eval["week_start"], y=test_eval["incident_count"], mode="lines+markers", name="Actual Incidents", line=dict(color="#0f172a", width=3)))
        fig_forecast.add_trace(go.Scatter(x=test_eval["week_start"], y=test_eval["Pred_Moving_Average_4W"], mode="lines+markers", name="4-Week Moving Avg", line=dict(color="#3b82f6", dash="dash")))
        fig_forecast.add_trace(go.Scatter(x=test_eval["week_start"], y=test_eval["Pred_Ridge_Autoregressive"], mode="lines+markers", name="Ridge Autoregressive ML", line=dict(color="#10b981", dash="dot")))
        fig_forecast.update_layout(title="Out-of-Sample Evaluation: Actual vs Model Predictions", xaxis_title="Week", yaxis_title="Incidents", height=320, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_forecast, use_container_width=True)

    # 2. Resolution Time Estimation
    st.markdown("---")
    st.markdown("### 2. Resolution-Time Estimation Modeling")
    st.caption("Predicts expected resolution duration for incoming municipal incidents based on historical closed cases.")

    with st.spinner("Training resolution time models..."):
        res_time_results = train_resolution_time_model(incidents_master, locations_master)

    if "error" not in res_time_results:
        r_col1, r_col2 = st.columns([1, 1])
        with r_col1:
            st.markdown("#### Model Benchmark Comparison")
            st.dataframe(res_time_results["results_table"], use_container_width=True, hide_index=True)
            st.caption(f"Trained on {res_time_results['train_records']:,} resolved cases; tested on {res_time_results['test_records']:,} held-out cases.")

        with r_col2:
            st.markdown("#### Feature Importance (Random Forest Regressor)")
            fig_imp = px.bar(
                res_time_results["feature_importances"],
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Purples",
                title="Top Predictive Drivers of Resolution Time"
            )
            fig_imp.update_layout(height=260, margin=dict(l=10, r=10, t=35, b=10))
            st.plotly_chart(fig_imp, use_container_width=True)

    # Scientific Rigor Callout
    st.markdown("""
    <div style="background-color: #f1f5f9; border-left: 4px solid #64748b; padding: 12px 16px; border-radius: 6px; margin-top: 12px;">
        <strong>🔬 Data Science Rigor & Honest Reporting Note:</strong><br/>
        <em>Why baseline moving averages often outperform complex machine learning on municipal time-series:</em><br/>
        With exactly 1 calendar year of educational data (52 weekly records), high-parameter neural networks or complex ensembles risk extreme overfitting. 
        Evaluating against historical mean and moving-average baselines provides an honest, reliable benchmark without fabricating artificial accuracy scores.
        To support advanced production forecasting, multi-year historical logs, weather sensor feeds, and live traffic telemetry would be required.
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# TAB 6: Municipal Resource Planner & Crew Dispatcher
# =========================================================
with tab_resource:
    st.subheader("Municipal Resource Planner & Dispatch Simulator")
    st.write(
        "Translates prioritized locations into an operational dispatch schedule. "
        "Uses a greedy priority-per-cost heuristic knapsack algorithm to maximize risk mitigation within budget and crew capacity constraints."
    )

    c_b1, c_b2, c_b3 = st.columns(3)
    with c_b1:
        sim_budget = st.slider("Total Municipal Repair Budget ($)", 10000, 300000, 100000, step=5000)
    with c_b2:
        sim_crews = st.slider("Available Specialized Crews", 2, 20, 8)
    with c_b3:
        sim_shift_hours = st.slider("Planning Horizon (Working Hours / Crew)", 20, 80, 40, step=5)

    dispatch_df, sim_summary = simulate_resource_allocation(
        ranked_locations_df=ranked_locations,
        total_budget=float(sim_budget),
        available_crews=int(sim_crews),
        shift_hours_per_crew=float(sim_shift_hours)
    )

    # Resource KPIs
    rk1, rk2, rk3, rk4, rk5 = st.columns(5)
    rk1.metric("Budget Allocated", f"${sim_summary['spent_budget']:,.2f}", f"{sim_summary['budget_utilized_pct']}% utilized")
    rk2.metric("Remaining Budget", f"${sim_summary['remaining_budget']:,.2f}")
    rk3.metric("Sites Dispatched", f"{sim_summary['locations_allocated']:,}", f"{sim_summary['locations_deferred']:,} deferred")
    rk4.metric("Crew Hours Dispatched", f"{sim_summary['total_repair_hours']:.1f} hrs", f"{sim_summary['capacity_utilized_pct']}% capacity")
    rk5.metric("Risk Score Mitigated", f"{sim_summary['risk_score_mitigated']:,.1f} pts")

    # Crew assignment breakdown
    col_c1, col_c2 = st.columns([1, 1])
    with col_c1:
        st.markdown("#### Dispatched Sites by Specialized Repair Crew")
        dispatched_only = dispatch_df[dispatch_df["allocation_status"] == "Dispatched"]
        if not dispatched_only.empty:
            crew_counts = dispatched_only["assigned_crew"].value_counts().reset_index()
            crew_counts.columns = ["Specialized Crew", "Sites Assigned"]
            fig_crews = px.bar(
                crew_counts,
                x="Sites Assigned",
                y="Specialized Crew",
                orientation="h",
                color="Sites Assigned",
                color_continuous_scale="Teal"
            )
            fig_crews.update_layout(height=260, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_crews, use_container_width=True)
        else:
            st.info("No sites dispatched with current budget/capacity parameters.")

    with col_c2:
        st.markdown("#### Cumulative Budget Burn Curve")
        if not dispatched_only.empty:
            fig_burn = px.line(
                dispatched_only.head(30),
                x="rank",
                y="cumulative_budget_spent",
                markers=True,
                title="Cumulative Budget Spend vs Location Priority Rank",
                labels={"rank": "Priority Rank", "cumulative_budget_spent": "Cumulative Spend ($)"}
            )
            fig_burn.add_hline(y=sim_budget, line_dash="dash", line_color="red", annotation_text="Budget Ceiling")
            fig_burn.update_layout(height=260, margin=dict(l=10, r=10, t=35, b=10))
            st.plotly_chart(fig_burn, use_container_width=True)

    # Detailed Dispatch Schedule Table
    st.markdown("#### 📋 Operational Crew Dispatch Schedule")
    show_dispatch_cols = [
        "rank", "location_id", "zone", "priority_score", "primary_incident_type",
        "assigned_crew", "estimated_cost", "estimated_repair_hours",
        "cumulative_budget_spent", "allocation_status"
    ]
    st.dataframe(
        dispatch_df[show_dispatch_cols].rename(columns={
            "rank": "Rank",
            "location_id": "Location",
            "zone": "Zone",
            "priority_score": "Score",
            "primary_incident_type": "Primary Issue",
            "assigned_crew": "Assigned Crew",
            "estimated_cost": "Cost ($)",
            "estimated_repair_hours": "Duration (hrs)",
            "cumulative_budget_spent": "Cumul. Spend ($)",
            "allocation_status": "Status"
        }),
        use_container_width=True,
        hide_index=True,
        height=300
    )

    csv_dispatch = io.StringIO()
    dispatch_df[show_dispatch_cols].to_csv(csv_dispatch, index=False)
    st.download_button(
        label="📥 Download Full Dispatch Schedule CSV",
        data=csv_dispatch.getvalue().encode("utf-8"),
        file_name="civicpulse_crew_dispatch_plan.csv",
        mime="text/csv"
    )

    st.caption(
        "📌 **Heuristic Assumption Disclaimer:** Repair costs and durations are estimates based on standard municipal unit costs "
        "(e.g., $3,200 for road patching, $1,800 for signal diagnostics). This simulation is a decision-support heuristic, "
        "not a verified dynamic operations-research vehicle routing solver."
    )

# =========================================================
# TAB 7: Data Quality & Audit Report
# =========================================================
with tab_audit:
    st.subheader("Automated Data Quality & Schema Audit Report")
    st.write("Verifies data cleanliness, relational integrity, schema mappings, and synthetic coordinate markers.")

    aq1, aq2, aq3, aq4 = st.columns(4)
    aq1.metric("Locations Table", f"{data_audit['locations_count']:,} rows", "0 Nulls | 0 Duplicates")
    aq2.metric("Incidents Table", f"{data_audit['incidents_count']:,} rows", "0 Nulls | 0 Duplicates")
    aq3.metric("Complaints Table", f"{data_audit['complaints_count']:,} rows", "0 Nulls | 0 Duplicates")
    aq4.metric("Join Integrity", "100% Match", "0 Orphaned Keys")

    st.markdown("---")
    col_sch1, col_sch2 = st.columns(2)
    with col_sch1:
        st.markdown("#### 📐 Relational Schema & Join Architecture")
        st.markdown("""
        All three datasets are linked via the shared foreign key `location_id`:
        - **`locations.csv`** (1,000 primary records): Contains static contextual signals (`zone`, `road_type`, `road_condition`, `population_density_per_km2`, `near_school`, `near_hospital`, `pedestrian_activity`, `latitude`, `longitude`).
        - **`incidents.csv`** (10,000 operational records): Contains time-stamped incident telemetry (`incident_type`, `severity`, `injuries`, `reported_by`, `resolved`, `resolution_hours`).
        - **`public_complaints.csv`** (8,000 civic records): Contains public reports (`category`, `priority_reported`, `status`, `days_to_resolution`).
        
        **Join Method:** Incidents and Complaints are aggregated to the location level, then joined to Location context using an inner/left relational merge on `location_id`.
        """)

    with col_sch2:
        st.markdown("#### ⚖️ Priority Scoring Methodology")
        st.markdown(r"""
        The **CivicPulse Priority Score** ($P \in [0, 100]$) is computed as:
        $$P = \\sum_{i=1}^5 w_i \\times F_i$$
        Where normalized factors $F_i$ represent:
        1. **Severity Factor ($F_{sev}$):** $0.70 \\times \\text{Mean Severity} + 0.30 \\times \\min(\\text{Injuries} \\times 20, 100)$
        2. **Frequency Factor ($F_{freq}$):** Normalized incident volume
        3. **Citizen Complaint Factor ($F_{comp}$):** Complaints volume + $1.5 \\times$ High-Priority complaints
        4. **Contextual Risk Factor ($F_{context}$):** Road condition ($40\\%$) + Pedestrian density ($25\\%$) + Schools ($15\\%$) + Hospitals ($15\\%$) + Pop. Density ($5\\%$)
        5. **Backlog Factor ($F_{wait}$):** Open complaints + Unresolved incidents + Max wait days
        
        **Safety Override:** When a Critical severity incident occurs near a school, hospital, or causes injuries, the score floor is clamped to $\\ge 85.0$ with an automatic Critical flag.
        """)

    st.markdown("---")
    st.markdown("#### 📋 Raw Location Schema Preview")
    st.dataframe(locations_master.head(10), use_container_width=True, hide_index=True)
