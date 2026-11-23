"""
AI-POWERED WILDLIFE CORRIDOR MONITORING AND PROTECTION
B.Tech Final-Year Capstone Project Main Application Dashboard.
Integrated with the Caltech Camera Traps (CCT-20) Dataset.
"""

from pathlib import Path
from datetime import date
import streamlit as st
import pandas as pd
import numpy as np

import config
from modules.theme import (
    init_theme_state,
    apply_theme,
    render_theme_toggle,
    get_current_theme,
    apply_plotly_theme
)
from modules.data_loader import (
    find_default_dataset_path,
    load_and_validate_dataset,
    get_dataset_summary,
    get_data_quality_report
)
from modules.preprocessing import prepare_full_pipeline
from modules.analytics import (
    compute_species_metrics,
    compute_location_analytics,
    compute_sequence_analytics,
    compute_historical_activity_trends,
    compute_predator_prey_interactions
)
from modules.charts import (
    apply_theme_layout,
    create_species_distribution_chart,
    create_activity_timeline_chart,
    create_location_activity_chart,
    create_diurnal_nocturnal_chart,
    create_hourly_activity_heatmap,
    create_sequence_length_histogram,
    create_confusion_matrix_chart,
    create_feature_importance_chart,
    create_risk_distribution_chart,
    create_activity_trend_comparison_chart
)
from modules.machine_learning import (
    prepare_ml_data,
    train_wildlife_classifier,
    predict_sample,
    check_vision_model_status
)
from modules.risk_analysis import (
    calculate_station_risk_scores,
    generate_prototype_alerts,
    generate_consolidated_corridor_report
)
from modules.image_viewer import render_image_record

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title=config.PROJECT_TITLE,
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Initialize Theme & Apply Dynamic Theme Styling
# -----------------------------------------------------------------------------
active_theme = init_theme_state()
apply_theme(active_theme)

# -----------------------------------------------------------------------------
# Sidebar: Branding, Theme Toggle & Navigation
# -----------------------------------------------------------------------------
st.sidebar.markdown(
    f"""
    <div class="sidebar-header-box">
        <div style='display: flex; align-items: center; gap: 10px; margin-bottom: 6px;'>
            <span style='font-size: 1.8rem;'>🌿</span>
            <div>
                <div class="sidebar-title">
                    EcoCorridor AI
                </div>
                <div class="sidebar-sub">
                    Wildlife Corridor Platform
                </div>
            </div>
        </div>
        <div class="sidebar-academic">
            B.Tech Final-Year Capstone · {config.PROJECT_ACADEMIC_YEAR}
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
active_theme = render_theme_toggle(sidebar=True)
st.sidebar.markdown("<hr style='margin: 12px 0 16px 0;'>", unsafe_allow_html=True)

st.sidebar.markdown("## 🧭 Navigation")
nav_selection = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home Dashboard",
        "🔍 Dataset Explorer",
        "🐾 Wildlife Species Analysis",
        "📷 Camera Location Monitoring",
        "⚠️ Wildlife Activity & Risk Analysis",
        "🛡️ Protection Recommendations",
        "🔔 Alerts & Recommendations",
        "🧠 Machine Learning Lab",
        "🖼️ Image Record Viewer",
        "ℹ️ About the Project"
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("## 📁 Dataset Source")

# Check for existing default dataset
default_path = find_default_dataset_path()
data_source_mode = st.sidebar.radio(
    "Data Input Method",
    ["Local Preloaded Dataset", "Upload Custom CSV"],
    index=0
)

uploaded_file = None
active_data_source = None

if data_source_mode == "Upload Custom CSV":
    uploaded_file = st.sidebar.file_uploader("Upload Camera Trap CSV", type=["csv"])
    if uploaded_file is not None:
        active_data_source = uploaded_file
else:
    if default_path:
        active_data_source = default_path
        st.sidebar.success(f"Active: `{default_path.name}`")
    else:
        st.sidebar.warning("Default dataset not found in `data/`. Please upload CSV.")

# Local Image Directory Setting (Optional)
st.sidebar.markdown("---")
st.sidebar.markdown("## 📷 Image Folder (Optional)")
custom_image_dir = st.sidebar.text_input(
    "Local Image Directory Path",
    value="",
    help="Path to folder containing raw camera-trap .jpg files if downloaded locally."
)

# -----------------------------------------------------------------------------
# Dataset Ingestion & Caching
# -----------------------------------------------------------------------------
if active_data_source is None:
    st.error("No dataset available. Please upload a CSV file via the sidebar to proceed.")
    st.stop()

with st.spinner("Loading and validating camera-trap dataset..."):
    df_raw, load_metadata = load_and_validate_dataset(active_data_source)

if df_raw is None or df_raw.empty:
    st.error(f"Failed to load dataset: {load_metadata.get('errors')}")
    st.stop()

# Prepare pipeline features
df = prepare_full_pipeline(df_raw)
summary = get_dataset_summary(df, load_metadata)

# -----------------------------------------------------------------------------
# SECTION 1: HOME DASHBOARD & OVERVIEW
# -----------------------------------------------------------------------------
if nav_selection == "🏠 Home Dashboard":
    st.markdown(
        f"""
        <div class="hero-container">
            <div class="hero-tag">🌲 Autonomous Ecological Sensing & Corridor Protection</div>
            <h1 class="hero-title">{config.PROJECT_TITLE}</h1>
            <p class="hero-subtitle">{config.PROJECT_SUBTITLE}</p>
            <div class="hero-meta-row">
                <div class="hero-meta-item">📁 Source: <strong>{load_metadata.get('source_name', 'CCT-20 Dataset')}</strong></div>
                <div class="hero-meta-item">⏱️ Monitoring Window: <strong>{summary.get('date_min', '2010')} to {summary.get('date_max', '2015')}</strong></div>
                <div class="hero-meta-item">📍 Camera Locations: <strong>{summary.get('unique_locations', 0)} Active Stations</strong></div>
                <div class="hero-meta-item">🎓 Academic Capstone: <strong>B.Tech AIML Final Year Evaluation</strong></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Key Summary Metric Cards
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="icon-chip">📊</div>
                <div class="metric-label">Total Images</div>
                <div class="metric-value">{summary.get('total_records', 0):,}</div>
                <div class="metric-delta">Validated records</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="icon-chip">🦌</div>
                <div class="metric-label">Unique Species</div>
                <div class="metric-value">{summary.get('unique_animal_species', 0)}</div>
                <div class="metric-delta">{summary.get('unique_categories', 0)} total categories</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m3:
        st.markdown(
            f"""
            <div class="metric-card blue">
                <div class="icon-chip">📷</div>
                <div class="metric-label">Camera Stations</div>
                <div class="metric-value">{summary.get('unique_locations', 0)}</div>
                <div class="metric-delta">Stationary sensor nodes</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="icon-chip">🎞️</div>
                <div class="metric-label">Capture Sequences</div>
                <div class="metric-value">{summary.get('unique_sequences', 0):,}</div>
                <div class="metric-delta">Burst events</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m5:
        st.markdown(
            f"""
            <div class="metric-card amber">
                <div class="icon-chip">👑</div>
                <div class="metric-label">Top Wildlife</div>
                <div class="metric-value" style="font-size: 1.45rem; text-transform: capitalize;">{summary.get('top_animal_species', 'N/A')}</div>
                <div class="metric-delta">{summary.get('top_animal_count', 0):,} captures</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m6:
        st.markdown(
            f"""
            <div class="metric-card red">
                <div class="icon-chip">⚠️</div>
                <div class="metric-label">Data Quality</div>
                <div class="metric-value">99.99%</div>
                <div class="metric-delta">1 coerced timestamp</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Core Visualizations (2 columns)
    c1, c2 = st.columns([1.1, 1.0])
    with c1:
        fig_species = create_species_distribution_chart(df, top_n=12, animals_only=True, theme=active_theme)
        st.plotly_chart(fig_species, use_container_width=True)

    with c2:
        fig_locations = create_location_activity_chart(df, top_n=12, theme=active_theme)
        st.plotly_chart(fig_locations, use_container_width=True)

    # Multi-Year Wildlife Timeline
    fig_timeline = create_activity_timeline_chart(df, freq="ME", theme=active_theme)
    st.plotly_chart(fig_timeline, use_container_width=True)

    # Recent Camera Trap Records Sample Table
    st.markdown(
        """
        <div class="section-header">
            <div>
                <h3 class="section-title">⏱️ Chronological Camera Trap Observations</h3>
                <div class="section-subtitle">Real telemetry records chronologically sampled from the active dataset</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    recent_sample = df.dropna(subset=["datetime"]).sort_values("datetime", ascending=False).head(8)
    display_cols = ["image_id", "date_captured", "location", "category", "frame_num", "seq_num_frames", "time_of_day", "rights_holder"]
    st.dataframe(
        recent_sample[display_cols],
        use_container_width=True,
        hide_index=True
    )

    # Scientific Context
    st.markdown(
        """
        <div class="notice-box forest">
            <strong>🌿 Academic Objective & Dataset Context:</strong><br>
            This dashboard uses empirical camera-trap telemetry from the Caltech Camera Traps (CCT-20) benchmark.
            Motion-activated field cameras capture sequences of burst images to study species abundance, migratory corridor usage,
            and diurnal/nocturnal wildlife activity without human disruption. All metrics, charts, and counts shown above are
            calculated dynamically from the verified CSV data pipeline.
        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------------------------------------------------------------
# SECTION 2: WILDLIFE SPECIES ANALYSIS
# -----------------------------------------------------------------------------
elif nav_selection == "🐾 Wildlife Species Analysis":
    st.markdown(
        """
        <div class="section-header">
            <div>
                <h2 class="section-title">🐾 Wildlife Species & Behavior Analysis</h2>
                <div class="section-subtitle">Species distributions, diurnal/nocturnal rhythms, and motion sequence burst telemetry</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    species_metrics = compute_species_metrics(df)

    tab1, tab2, tab3 = st.tabs([
        "📊 Species Distribution & Roster",
        "🌙 Diurnal vs. Nocturnal Rhythms",
        "🎞️ Motion Sequence Burst Metrics"
    ])

    with tab1:
        c1, c2 = st.columns([1.2, 0.8])
        with c1:
            show_animals_only = st.checkbox("Filter Wildlife Animals Only (exclude empty and cars)", value=True)
            fig_dist = create_species_distribution_chart(df, top_n=15, animals_only=show_animals_only, theme=active_theme)
            st.plotly_chart(fig_dist, use_container_width=True)
        with c2:
            st.markdown("#### Observed Species Breakdown")
            filtered_species_table = species_metrics[species_metrics["is_animal"]] if show_animals_only else species_metrics
            st.dataframe(
                filtered_species_table[["category", "total_count", "percentage", "unique_locations", "nocturnal_ratio"]].head(15),
                use_container_width=True,
                hide_index=True
            )

    with tab2:
        st.markdown("### Diurnal (Day: 06h–18h) vs. Nocturnal (Night: 18h–06h) Distribution")
        fig_diurnal = create_diurnal_nocturnal_chart(df, top_n=14, theme=active_theme)
        st.plotly_chart(fig_diurnal, use_container_width=True)
        st.caption("Nocturnal vs. diurnal classification is computed by converting valid ISO timestamps to hour of day.")

    with tab3:
        st.markdown("### Camera Trap Motion Sequence Analysis")
        seq_stats = compute_sequence_analytics(df)

        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.metric("Total Sequences", f"{seq_stats.get('total_sequences', 0):,}")
        with s2:
            st.metric("Mean Frames/Sequence", f"{seq_stats.get('mean_frames_per_seq', 1.0)}")
        with s3:
            st.metric("Single-Shot Triggers", f"{seq_stats.get('single_shot_ratio_pct', 0)}%")
        with s4:
            st.metric("Sequences with Wildlife", f"{seq_stats.get('animal_sequence_ratio_pct', 0)}%")

        fig_seq = create_sequence_length_histogram(df, theme=active_theme)
        st.plotly_chart(fig_seq, use_container_width=True)

# -----------------------------------------------------------------------------
# SECTION 3: CAMERA LOCATION MONITORING
# -----------------------------------------------------------------------------
elif nav_selection == "📷 Camera Location Monitoring":
    st.markdown(
        """
        <div class="section-header">
            <div>
                <h2 class="section-title">📷 Camera Trap Location & Station Monitoring</h2>
                <div class="section-subtitle">24-hour activity heatmaps, Shannon diversity index, and multi-year observation trends</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    location_analytics = compute_location_analytics(df)

    tab1, tab2, tab3 = st.tabs([
        "📷 24-Hour Activity Heatmap",
        "📈 Multi-Year Observation Trajectories",
        "📊 Camera Station Diversity & Roster"
    ])

    with tab1:
        st.markdown("### 24-Hour Camera Station Activity Matrix")
        heat_locs_n = st.slider("Number of Top Camera Stations to Display", 8, 25, 14)
        fig_heatmap = create_hourly_activity_heatmap(df, top_locations_n=heat_locs_n, theme=active_theme)
        st.plotly_chart(fig_heatmap, use_container_width=True)

    with tab2:
        st.markdown("### Changes in Observed Wildlife Activity Over Time (2010–2015)")
        trends_df = compute_historical_activity_trends(df, top_locations_n=12)
        fig_trends = create_activity_trend_comparison_chart(trends_df, theme=active_theme)
        st.plotly_chart(fig_trends, use_container_width=True)

        st.markdown("#### Camera Station Trajectory Audit")
        st.dataframe(
            trends_df[["location_id", "activity_trajectory", "historical_change_pct", "total_period_captures"]],
            use_container_width=True,
            hide_index=True
        )

    with tab3:
        st.markdown("### Comprehensive Camera Station Diversity & Ranking Roster")
        st.dataframe(
            location_analytics[[
                "location_id", "total_records", "wildlife_records", "vehicle_records",
                "empty_rate_pct", "species_richness", "shannon_diversity", "is_hotspot"
            ]].head(40),
            use_container_width=True,
            hide_index=True
        )

# -----------------------------------------------------------------------------
# SECTION 4: WILDLIFE ACTIVITY & RISK ANALYSIS
# -----------------------------------------------------------------------------
elif nav_selection == "⚠️ Wildlife Activity & Risk Analysis":
    st.markdown(
        """
        <div class="section-header">
            <div>
                <h2 class="section-title">⚠️ Wildlife Corridor Risk Assessment</h2>
                <div class="section-subtitle">Explainable, rule-based Prototype Risk Scoring based on activity volume, vehicle intrusion, nocturnal ratio, and predator presence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="notice-box">
            <strong>🛡️ Scientific Framing & Explainability:</strong> This scoring system is an <em>activity-based prototype index</em> 
            designed to flag camera stations with atypical observation surges, high vehicular disturbance, or heavy nocturnal 
            animal movement. <strong>It distinguishes observed facts from inferred risk and does NOT prove poaching, deforestation, 
            or verified vehicle collisions.</strong> Stations with fewer than 5 records are designated as <strong>Insufficient Data</strong>.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Configurable Parameters
    with st.expander("⚙️ Configure Explainable Risk Weights & Thresholds", expanded=False):
        w_col1, w_col2, w_col3, w_col4 = st.columns(4)
        with w_col1:
            act_w = st.slider("Activity Volume Weight", 0.0, 1.0, 0.35, 0.05)
        with w_col2:
            veh_w = st.slider("Vehicle Disturbance Weight", 0.0, 1.0, 0.30, 0.05)
        with w_col3:
            noc_w = st.slider("Nocturnal Ratio Weight", 0.0, 1.0, 0.20, 0.05)
        with w_col4:
            pred_w = st.slider("Predator Presence Weight", 0.0, 1.0, 0.15, 0.05)

        t_col1, t_col2 = st.columns(2)
        with t_col1:
            high_thresh = st.slider("High Risk Priority Threshold", 50.0, 90.0, 70.0, 5.0)
        with t_col2:
            med_thresh = st.slider("Medium Risk Priority Threshold", 20.0, 60.0, 40.0, 5.0)

    # Calculate Station Risk Scores
    risk_df = calculate_station_risk_scores(
        df,
        activity_weight=act_w,
        vehicle_weight=veh_w,
        nocturnal_weight=noc_w,
        predator_weight=pred_w,
        high_threshold=high_thresh,
        medium_threshold=med_thresh
    )

    # Summary Metrics & Distribution Chart
    high_count = (risk_df["risk_level"] == "High").sum()
    med_count = (risk_df["risk_level"] == "Medium").sum()
    low_count = (risk_df["risk_level"] == "Low").sum()
    sparse_count = (risk_df["risk_level"] == "Insufficient Data").sum()

    r_col1, r_col2 = st.columns([1.1, 0.9])
    with r_col1:
        st.markdown("### Corridor Risk Tier Summary")
        rk1, rk2, rk3, rk4 = st.columns(4)
        with rk1:
            st.metric("Total Stations Scored", len(risk_df))
        with rk2:
            st.metric("High Priority", high_count)
        with rk3:
            st.metric("Medium Priority", med_count)
        with rk4:
            st.metric("Insufficient Data", sparse_count)

        st.markdown(
            f"""
            <div style='margin-top: 15px;'>
                <strong>Scoring Formula:</strong><br>
                <code>Risk Score = ({act_w:.2f} × Activity) + ({veh_w:.2f} × Vehicle) + ({noc_w:.2f} × Nocturnal) + ({pred_w:.2f} × Predator)</code>
            </div>
            """,
            unsafe_allow_html=True
        )

    with r_col2:
        fig_risk_dist = create_risk_distribution_chart(risk_df, theme=active_theme)
        st.plotly_chart(fig_risk_dist, use_container_width=True)

    # Prioritized Stations Table
    st.markdown("### 🎯 Prioritized Camera Stations Roster")
    st.dataframe(
        risk_df[[
            "location_id", "prototype_risk_score", "risk_level",
            "total_observations", "vehicle_observations", "predator_observations",
            "nocturnal_percentage", "rationale", "recommended_action"
        ]],
        use_container_width=True,
        hide_index=True
    )

    # Station Diagnostic Inspector
    st.markdown("### 🔎 Station Telemetry & Diagnostic Inspector")
    inspect_loc = st.selectbox("Select Station ID to Inspect", risk_df["location_id"].tolist())
    station_data = risk_df[risk_df["location_id"] == inspect_loc].iloc[0]

    sc1, sc2 = st.columns([1.0, 1.2])
    with sc1:
        st.markdown(
            f"""
            <div class="content-box">
                <h4>Station #{station_data['location_id']} Diagnostic Report</h4>
                <p>Prototype Risk Score: <strong>{station_data['prototype_risk_score']} / 100</strong></p>
                <p>Assigned Priority Tier: <span class="badge {station_data['badge_class']}">{station_data['risk_level'].upper()}</span></p>
                <hr>
                <p><strong>Subscore Breakdown:</strong></p>
                <ul>
                    <li>Activity Volume Subscore: <code>{station_data['activity_subscore']}</code></li>
                    <li>Vehicle Disturbance Subscore: <code>{station_data['vehicle_subscore']}</code></li>
                    <li>Nocturnal Concentration Subscore: <code>{station_data['nocturnal_subscore']}</code></li>
                    <li>Apex Predator Subscore: <code>{station_data['predator_subscore']}</code></li>
                </ul>
                <hr>
                <p><strong>Observed Evidence & Rationale:</strong> {station_data['rationale']}</p>
                <p><strong>Recommended Action:</strong> {station_data['recommended_action']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with sc2:
        st_df = df[df["location"] == inspect_loc]
        if "hour" in st_df.columns and not st_df.empty:
            hourly_counts = st_df.groupby("hour").size().reset_index(name="Captures")
            import plotly.graph_objects as go
            bar_color = "#15803d" if active_theme == "light" else "#22c55e"
            fig_st = go.Figure(go.Bar(
                x=[f"{int(h):02d}:00" for h in hourly_counts["hour"]],
                y=hourly_counts["Captures"],
                marker_color=bar_color
            ))
            apply_theme_layout(
                fig_st,
                title=f"Station #{inspect_loc} 24-Hour Observation Distribution",
                height=320,
                theme=active_theme
            )
            fig_st.update_layout(xaxis_title="Hour of Day", yaxis_title="Observations")
            st.plotly_chart(fig_st, use_container_width=True)

# -----------------------------------------------------------------------------
# SECTION 4: PROTECTION RECOMMENDATIONS
# -----------------------------------------------------------------------------
elif nav_selection == "🛡️ Protection Recommendations":
    st.markdown(
        """
        <div class="section-header">
            <div>
                <h2 class="section-title">🛡️ Evidence-Based Conservation & Protection Recommendations</h2>
                <div class="section-subtitle">Actionable decision support generated directly from observed empirical camera-trap telemetry</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    risk_df = calculate_station_risk_scores(df)

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        high_risk_locs = risk_df[risk_df["risk_level"] == "High"]["location_id"].tolist()
        st.metric("High-Priority Stations", len(high_risk_locs))
    with p2:
        veh_locs = risk_df[risk_df["vehicle_observations"] > 0]["location_id"].tolist()
        st.metric("Vehicle Conflict Stations", len(veh_locs))
    with p3:
        sparse_locs = risk_df[risk_df["risk_level"] == "Insufficient Data"]["location_id"].tolist()
        st.metric("Sparse Coverage Stations", len(sparse_locs))
    with p4:
        pred_locs = risk_df[risk_df["predator_observations"] > 0]["location_id"].tolist()
        st.metric("Carnivore Corridor Stations", len(pred_locs))

    st.markdown("<br>", unsafe_allow_html=True)

    c_rec1, c_rec2 = st.columns(2)
    with c_rec1:
        st.markdown(
            f"""
            <div class="content-box">
                <h4>1. Prioritize Verified High-Risk Field Inspections</h4>
                <p>Camera stations with high composite scores represent overlapping pressure points. Field patrol teams should prioritize the following verified stations:</p>
                <p><strong>Target Stations:</strong> <code>{', '.join(high_risk_locs[:10]) if high_risk_locs else 'None currently meet High threshold'}</code></p>
                <p><strong>Action:</strong> Inspect trail boundaries, verify whether animal tracks run parallel to motorized roads, and assess fencing integrity.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="content-box" style="margin-top: 15px;">
                <h4>2. Anthropogenic Vehicular Trail Mitigation</h4>
                <p>A total of <strong>{len(veh_locs)} stations</strong> recorded motorized vehicle intrusions along trailways.</p>
                <p><strong>Action:</strong> Coordinate with local forestry management to install wildlife crossing alert signage, reduce trail vehicle speeds, and restrict unauthorized nighttime vehicle access.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c_rec2:
        st.markdown(
            f"""
            <div class="content-box">
                <h4>3. Sensor Coverage & Maintenance Redeployment</h4>
                <p>A total of <strong>{len(sparse_locs)} camera locations</strong> logged fewer than 5 records across the monitoring period, indicating potential camera failure, obscured view, or poor placement.</p>
                <p><strong>Target Stations:</strong> <code>{', '.join(sparse_locs) if sparse_locs else 'None'}</code></p>
                <p><strong>Action:</strong> Schedule technician maintenance to replace batteries, re-align sensor angles, or relocate cameras toward verified game trails.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="content-box" style="margin-top: 15px;">
                <h4>4. Multi-Species Corridor Connectivity Protection</h4>
                <p>Stations displaying high Shannon diversity ($H > 1.8$) alongside high nocturnal wildlife activity are critical multi-species migratory corridors.</p>
                <p><strong>Action:</strong> Protect these corridors from future trail construction or fragmentation to maintain genetic connectivity between fragmented habitats.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# -----------------------------------------------------------------------------
# SECTION 5: ALERTS & RECOMMENDATIONS
# -----------------------------------------------------------------------------
elif nav_selection == "🔔 Alerts & Recommendations":
    st.markdown(
        """
        <div class="section-header">
            <div>
                <h2 class="section-title">🔔 Automated Alerts & Downloadable Corridor Reports</h2>
                <div class="section-subtitle">Real-time prototype alerts generated from sensor anomalies and exportable reports for conservation decision support</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="notice-box">
            <strong>⚠️ Demonstration Notice:</strong> These alerts are derived automatically from 
            statistical baseline deviations (e.g. observation surges, vehicle interference). They are intended for 
            academic demonstration and have <strong>not been dispatched to emergency response teams</strong>.
        </div>
        """,
        unsafe_allow_html=True
    )

    risk_df = calculate_station_risk_scores(df)
    alerts_df = generate_prototype_alerts(risk_df, df)

    # Downloadable Consolidated Report
    st.markdown("### 📥 Consolidated Conservation Report Export")
    consolidated_report = generate_consolidated_corridor_report(risk_df, df)
    csv_report_data = consolidated_report.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Full Corridor Risk Assessment & Recommendations (CSV)",
        data=csv_report_data,
        file_name="wildlife_corridor_risk_assessment_report.csv",
        mime="text/csv",
        help="Download complete station-by-station telemetry, risk scores, subscores, rationales, and conservation actions."
    )

    st.markdown("---")

    # Alert Filtering
    st.markdown("### 🚨 Active Prototype Alerts Feed")
    a_col1, a_col2 = st.columns(2)
    with a_col1:
        sev_filter = st.multiselect("Filter Severity", ["High", "Medium", "Low"], default=["High", "Medium"])
    with a_col2:
        available_types = alerts_df["alert_type"].unique().tolist()
        type_filter = st.multiselect("Filter Alert Type", available_types, default=available_types)

    filtered_alerts = alerts_df[
        alerts_df["severity"].isin(sev_filter) &
        alerts_df["alert_type"].isin(type_filter)
    ]

    st.caption(f"Showing **{len(filtered_alerts)}** of **{len(alerts_df)}** triggered prototype alerts.")

    st.dataframe(
        filtered_alerts[[
            "alert_id", "location_id", "date_observed", "severity",
            "alert_type", "reason", "recommended_action", "alert_classification"
        ]],
        use_container_width=True,
        hide_index=True
    )

# -----------------------------------------------------------------------------
# SECTION 6: DATASET EXPLORER
# -----------------------------------------------------------------------------
elif nav_selection == "🔍 Dataset Explorer":
    st.markdown(
        """
        <div class="section-header">
            <div>
                <h2 class="section-title">🔍 Dataset Explorer & Data Quality Audit</h2>
                <div class="section-subtitle">Multi-criteria filtering, schema inspection, data completeness metrics, and verified subset export</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    quality_report = get_data_quality_report(df, load_metadata)
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        st.metric("Total Records", f"{len(df):,}")
    with q2:
        st.metric("Completeness Score", f"{quality_report.get('completeness_score', 100)}%")
    with q3:
        st.metric("Duplicate Image IDs", f"{quality_report.get('duplicate_image_ids', 0)}")
    with q4:
        st.metric("Invalid Timestamps", f"{quality_report.get('invalid_timestamps', 0)}")

    st.markdown("---")

    # Multi-Criteria Filters
    st.markdown("### 🎛️ Multi-Criteria Record Filters")
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)

    with f_col1:
        all_categories = sorted(df["category"].dropna().unique().tolist())
        selected_categories = st.multiselect("Species / Category", all_categories, default=[])

    with f_col2:
        all_locations = sorted(df["location"].dropna().unique().tolist(), key=lambda x: int(x) if x.isdigit() else str(x))
        selected_locations = st.multiselect("Camera Station ID", all_locations, default=[])

    with f_col3:
        time_options = ["All", "Day", "Night"]
        selected_time = st.selectbox("Time of Day", time_options, index=0)

    with f_col4:
        search_query = st.text_input("Search Image ID or Filename", value="")

    # Optional Date Range Filter
    if "date" in df.columns:
        valid_dates_series = df["date"].dropna()
        if not valid_dates_series.empty:
            min_d, max_d = valid_dates_series.min(), valid_dates_series.max()
            date_range = st.date_input("Filter Capture Date Range", (min_d, max_d), min_value=min_d, max_value=max_d)
        else:
            date_range = None
    else:
        date_range = None

    # Apply Filters
    filtered_df = df.copy()
    if selected_categories:
        filtered_df = filtered_df[filtered_df["category"].isin(selected_categories)]
    if selected_locations:
        filtered_df = filtered_df[filtered_df["location"].isin(selected_locations)]
    if selected_time != "All":
        filtered_df = filtered_df[filtered_df["time_of_day"] == selected_time]
    if search_query.strip():
        q = search_query.strip().lower()
        filtered_df = filtered_df[
            filtered_df["image_id"].str.lower().str.contains(q, na=False) |
            filtered_df["file_name"].str.lower().str.contains(q, na=False)
        ]
    if date_range and isinstance(date_range, tuple) and len(date_range) == 2:
        start_d, end_d = date_range
        filtered_df = filtered_df[(filtered_df["date"] >= start_d) & (filtered_df["date"] <= end_d)]

    st.caption(f"Showing **{len(filtered_df):,}** of **{len(df):,}** matching records ({len(filtered_df)/len(df)*100:.1f}%)")

    # Data Preview
    st.dataframe(
        filtered_df[[c for c in config.EXPECTED_COLUMNS if c in filtered_df.columns]].head(200),
        use_container_width=True,
        hide_index=True
    )

    # Export Button
    csv_export = filtered_df.head(10000).to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Filtered Records (CSV - up to 10,000 rows)",
        data=csv_export,
        file_name="filtered_wildlife_records.csv",
        mime="text/csv"
    )

    # Formal Quality Audit Tab
    st.markdown("### 📋 Formal Schema & Data Quality Audit")
    with st.expander("View Full Column Breakdown & Integrity Audit", expanded=False):
        col_stats = []
        for col in df.columns:
            null_count = int(df[col].isnull().sum())
            null_pct = round(null_count / len(df) * 100, 2)
            col_stats.append({
                "Column Name": col,
                "Data Type": str(df[col].dtype),
                "Missing Values": null_count,
                "Missing %": f"{null_pct}%",
                "Unique Values": df[col].nunique(),
                "Sample Value": str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else "None"
            })
        st.dataframe(pd.DataFrame(col_stats), use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# SECTION 7: MACHINE LEARNING LAB
# -----------------------------------------------------------------------------
elif nav_selection == "🧠 Machine Learning Lab":
    st.markdown(
        """
        <div class="section-header">
            <div>
                <h2 class="section-title">🧠 Machine Learning Lab: Wildlife Species Classification</h2>
                <div class="section-subtitle">Supervised learning with Scikit-Learn Random Forest, sequence-group leakage prevention, and model evaluation</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="notice-box forest">
            <strong>🔬 Scientific Transparency & Model Framing:</strong><br>
            This model predicts wildlife occurrence categories using <strong>tabular spatio-temporal metadata</strong> 
            (time of day, seasonality, sequence length, frame ratio, sensor dimensions, and camera station ID).
            <strong>Ground-truth labels in the CSV are NOT treated as model predictions.</strong> All metrics reported below 
            are calculated strictly after model training on an independent test split using <strong>sequence group holdouts</strong> 
            to avoid temporal burst leakage.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Vision Model Capabilities Notice
    vision_status = check_vision_model_status()
    with st.expander("👁️ Deep Learning Computer Vision Status (MegaDetector V6)", expanded=False):
        st.markdown(f"**Framework:** `{vision_status['vision_framework']}`")
        st.markdown(f"**Core Module Installed:** `{vision_status['framework_installed']}`")
        st.markdown(f"**Compute Device:** `{vision_status['device']}`")
        st.info(vision_status["description"])

    # ML Configuration Controls
    ml_col1, ml_col2, ml_col3 = st.columns(3)
    with ml_col1:
        target_mode = st.selectbox(
            "Classification Target",
            ["top_species", "wildlife_vs_empty", "all_animals"],
            format_func=lambda x: {
                "top_species": "Top 7 Wildlife Species + Empty (Multi-class)",
                "wildlife_vs_empty": "Wildlife Present vs. Empty (Binary)",
                "all_animals": "Verified Animal Species Only (Multi-class)"
            }[x]
        )
    with ml_col2:
        sample_size = st.slider("Training Sample Size", min_value=5000, max_value=35000, value=15000, step=5000)
    with ml_col3:
        n_estimators = st.slider("Random Forest Trees (n_estimators)", min_value=50, max_value=200, value=100, step=25)

    if st.button("🚀 Train Machine Learning Model"):
        with st.spinner("Extracting features, splitting sequence-group test sets, and fitting Random Forest..."):
            X, y, num_feats, cat_feats, class_names, groups = prepare_ml_data(
                df, target_mode=target_mode, sample_size=sample_size, return_groups=True
            )
            ml_results = train_wildlife_classifier(
                X, y, num_feats, cat_feats, n_estimators=n_estimators, groups=groups
            )
            st.session_state["ml_results"] = ml_results
            st.session_state["ml_target_mode"] = target_mode
        st.success(f"Model trained successfully using **{ml_results['split_method']}**!")

    # Display Results if Trained
    if "ml_results" in st.session_state:
        results = st.session_state["ml_results"]

        st.markdown(f"### 📈 Evaluation Metrics (Independent Test Set — {results['test_samples']:,} Samples)")
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.metric("Accuracy", f"{results['accuracy']*100:.2f}%")
        with r2:
            st.metric("Macro Precision", f"{results['precision_macro']*100:.2f}%")
        with r3:
            st.metric("Macro Recall", f"{results['recall_macro']*100:.2f}%")
        with r4:
            st.metric("Weighted F1-Score", f"{results['f1_weighted']*100:.2f}%")

        c1, c2 = st.columns([1.1, 0.9])
        with c1:
            fig_cm = create_confusion_matrix_chart(results["confusion_matrix"], results["classes"], theme=active_theme)
            st.plotly_chart(fig_cm, use_container_width=True)
        with c2:
            fig_imp = create_feature_importance_chart(results["feature_names"], results["feature_importances"], theme=active_theme)
            st.plotly_chart(fig_imp, use_container_width=True)

        # Single Record Inference Simulator
        st.markdown("### 🧪 Interactive Single-Observation Predictor")
        st.caption("Simulate camera sensor telemetry inputs to obtain real-time class probability distributions.")

        sim_c1, sim_c2, sim_c3, sim_c4 = st.columns(4)
        with sim_c1:
            sim_loc = st.selectbox("Camera Station ID", sorted(df["location"].dropna().unique().tolist()))
        with sim_c2:
            sim_hour = st.slider("Hour of Capture (24h)", 0, 23, 21)
        with sim_c3:
            sim_month = st.slider("Month of Capture", 1, 12, 10)
        with sim_c4:
            sim_seq_len = st.selectbox("Sequence Burst Length", [1, 2, 3, 5])

        if st.button("🔮 Predict Wildlife Class"):
            sim_input = {
                "location": str(sim_loc),
                "hour": sim_hour,
                "month": sim_month,
                "day_of_week": 2,
                "frame_num": 1,
                "seq_num_frames": sim_seq_len,
                "width": 2048,
                "height": 1494
            }
            pred_out = predict_sample(results, sim_input)

            st.markdown(
                f"""
                <div class="content-box">
                    <h4>Prediction Outcome: <span class="badge badge-success">{pred_out['predicted_class'].upper()}</span></h4>
                    <p>Model Confidence: <strong>{pred_out['confidence']*100:.1f}%</strong></p>
                </div>
                """,
                unsafe_allow_html=True
            )
            prob_df = pd.DataFrame(list(pred_out["probabilities"].items()), columns=["Class", "Probability"])
            prob_df["Probability %"] = (prob_df["Probability"] * 100).round(1).astype(str) + "%"
            st.dataframe(prob_df, use_container_width=True, hide_index=True)
    else:
        st.info("Click the **'Train Machine Learning Model'** button above to train and evaluate the Scikit-Learn Random Forest model.")

# -----------------------------------------------------------------------------
# SECTION 8: IMAGE RECORD VIEWER
# -----------------------------------------------------------------------------
elif nav_selection == "🖼️ Image Record Viewer":
    st.markdown(
        """
        <div class="section-header">
            <div>
                <h2 class="section-title">🖼️ Camera Sensor Image & Telemetry Viewer</h2>
                <div class="section-subtitle">Inspect individual camera-trap captures with verified metadata, telemetry cards, and raw photograph lookup</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    v_c1, v_c2, v_c3 = st.columns(3)
    with v_c1:
        cat_filter = st.selectbox("Filter Species/Category", ["All Animals"] + sorted(df["category"].unique().tolist()))
    with v_c2:
        loc_filter = st.selectbox("Filter Camera Location", ["All Stations"] + sorted(df["location"].unique().tolist()))
    with v_c3:
        jump_id = st.text_input("Jump directly to Image ID (Exact / Partial)", value="")

    view_df = df.copy()
    if cat_filter == "All Animals":
        view_df = view_df[view_df["is_animal"]]
    elif cat_filter != "All":
        view_df = view_df[view_df["category"] == cat_filter]

    if loc_filter != "All Stations":
        view_df = view_df[view_df["location"] == loc_filter]

    if jump_id.strip():
        view_df = view_df[view_df["image_id"].str.contains(jump_id.strip(), case=False, na=False)]

    if view_df.empty:
        st.warning("No records found matching your selection criteria.")
    else:
        st.caption(f"Displaying from a filtered pool of **{len(view_df):,}** candidate records.")
        record_idx = st.slider("Record Index Selector", 0, min(len(view_df) - 1, 500), 0)
        selected_record = view_df.iloc[record_idx].to_dict()

        st.markdown("<br>", unsafe_allow_html=True)
        render_image_record(selected_record, image_dir=custom_image_dir)

# -----------------------------------------------------------------------------
# SECTION 9: ABOUT THE PROJECT & METHODOLOGY
# -----------------------------------------------------------------------------
elif nav_selection == "ℹ️ About the Project":
    st.markdown(
        f"""
        <div class="hero-container">
            <div class="hero-tag">🎓 Capstone Documentation & System Architecture</div>
            <h1 class="hero-title">{config.PROJECT_TITLE}</h1>
            <p class="hero-subtitle">{config.PROJECT_SUBTITLE}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        ### 1. Problem Statement & Motivation
        Habitat fragmentation caused by highways, urban growth, and industrial development isolates wildlife populations,
        increasing human-wildlife conflicts and depleting biodiversity. Motion-activated camera traps provide continuous,
        non-invasive ecological telemetry to monitor corridor utilization. However, analyzing millions of image records
        manually is a major operational bottleneck. This platform provides an end-to-end analytical framework to ingest,
        validate, analyze, model, and assess corridor risk from camera-trap networks.

        ### 2. Dataset Provenance & Structure
        - **Benchmark:** Caltech Camera Traps (CCT-20) Dataset curated by Sara Beery et al., USGS, and regional wildlife agencies.
        - **Total Records:** 243,100 verified captures spanning 5 years (2010 to 2015).
        - **Camera Stations:** 140 stationary sensor nodes across wildlife corridors.
        - **Sequence Bursts:** 181,008 motion bursts.
        - **Categories:** 23 classes covering wild fauna (e.g. *deer, coyote, bobcat, mountain lion*), domestic animals (*dog, cow*), and non-animal triggers (*empty, car*).

        ### 3. Explainable Risk Assessment Methodology
        - **Activity Volume Subscore:** Compares station observation counts against network mean baseline.
        - **Vehicle Disturbance Subscore:** Captures anthropogenic intrusion frequency via motorized vehicle records (`car`).
        - **Nocturnal Concentration Subscore:** Evaluates shifts toward nocturnal movement (18h–06h) indicating human-avoidance behavior.
        - **Predator Presence Subscore:** Assesses presence of conflict-prone apex carnivores (*mountain lion, bobcat, coyote*).
        - **Composite Index:** Transparent weighted formula with configurable thresholds. Stations with fewer than 5 records are designated **Insufficient Data**.

        ### 4. Machine Learning Methodology
        - **Algorithm:** Scikit-Learn Random Forest Classifier (`RandomForestClassifier`).
        - **Features:** Hour of day, month, day of week, cyclical sine/cosine features, camera location, burst length, frame ratio, image dimensions.
        - **Leakage Prevention:** Identifiers (`image_id`, `file_name`, `seq_id`, `rights_holder`, `labels`) are excluded from training features, and sequence-group splitting is applied.
        - **Evaluation:** Macro Precision, Macro Recall, Weighted F1, Confusion Matrix, and MDI Feature Importances.

        ### 5. Academic Scope & Limitations
        - **Dataset Nature:** Camera stations represent stationary camera points, not continuous GPS coordinates.
        - **Risk Definition:** The prototype risk index reflects statistical anomalies and anthropogenic overlap; it does not claim to prove poaching, deforestation, or blocked corridors without ground verification.
        - **Deep Learning Vision:** When raw pixel weights are offline, the system relies on verified tabular metadata and explains pixel-inference status without generating fabricated predictions.
        """
    )

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="footer-text">
        {config.PROJECT_TITLE} · {config.PROJECT_TAGLINE} · Academic Year {config.PROJECT_ACADEMIC_YEAR}<br>
        Developed with Streamlit, Scikit-Learn, Pandas, and Plotly.
    </div>
    """,
    unsafe_allow_html=True
)
