"""
Visualization and Plotly Chart Utilities.
Generates styled interactive charts adhering to centralized Dark Forest and Light Day themes.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import config
from modules.theme import apply_plotly_theme, get_current_theme, get_theme_config


def apply_theme_layout(
    fig: go.Figure,
    title: str = "",
    height: int = 400,
    theme: Optional[str] = None
) -> go.Figure:
    """
    Apply consistent styling to any Plotly figure according to the active theme.
    Shared layout helper supporting both Dark Forest and Light Day modes.
    """
    return apply_plotly_theme(fig, theme=theme, title=title, height=height)


def create_species_distribution_chart(
    df: pd.DataFrame,
    top_n: int = 15,
    animals_only: bool = True,
    theme: Optional[str] = None
) -> go.Figure:
    """Generate horizontal bar chart of species / category counts with theme-aware styling."""
    active_theme = theme or get_current_theme()
    target_df = df
    if animals_only and "is_animal" in df.columns:
        target_df = df[df["is_animal"]]

    counts = target_df["category"].value_counts().head(top_n).reset_index()
    counts.columns = ["Species / Category", "Observation Count"]
    counts = counts.sort_values("Observation Count", ascending=True)

    if active_theme == "light":
        colorscale = [
            [0.0, "#059669"],
            [0.5, "#16a34a"],
            [1.0, "#4ade80"]
        ]
        line_color = "#15803d"
    else:
        colorscale = [
            [0.0, "#10b981"],
            [0.5, "#22c55e"],
            [1.0, "#86efac"]
        ]
        line_color = "#2e593e"

    fig = go.Figure(go.Bar(
        x=counts["Observation Count"],
        y=counts["Species / Category"],
        orientation="h",
        marker=dict(
            color=counts["Observation Count"],
            colorscale=colorscale,
            line=dict(color=line_color, width=1)
        ),
        hovertemplate="<b>%{y}</b><br>Observations: %{x:,}<extra></extra>"
    ))

    title = f"Top {top_n} {'Animal Species' if animals_only else 'Categories'} Recorded"
    apply_theme_layout(fig, title=title, height=420, theme=active_theme)
    fig.update_layout(xaxis_title="Total Image Observations", yaxis_title="")
    return fig


def create_activity_timeline_chart(
    df: pd.DataFrame,
    freq: str = "ME",
    theme: Optional[str] = None
) -> go.Figure:
    """Generate time series area chart of wildlife observations over time."""
    active_theme = theme or get_current_theme()
    if "datetime" not in df.columns or df["datetime"].dropna().empty:
        # Fallback empty chart
        fig = go.Figure()
        apply_theme_layout(fig, title="Observation Timeline (No valid dates)", height=350, theme=active_theme)
        return fig

    # Filter valid dates and resample
    valid_df = df.dropna(subset=["datetime"]).copy()
    valid_df = valid_df.set_index("datetime")

    # Resample all vs animal-only with version compatibility
    try:
        total_series = valid_df.resample(freq).size().rename("Total Captures")
    except ValueError:
        freq = "M" if freq == "ME" else "ME"
        total_series = valid_df.resample(freq).size().rename("Total Captures")

    if "is_animal" in valid_df.columns:
        animal_series = valid_df[valid_df["is_animal"]].resample(freq).size().rename("Wildlife Captures")
        combined = pd.concat([total_series, animal_series], axis=1).fillna(0).reset_index()
    else:
        combined = total_series.reset_index()

    fig = go.Figure()

    all_line_color = "#94a3b8" if active_theme == "light" else "#475569"
    wildlife_color = "#15803d" if active_theme == "light" else "#22c55e"
    fill_color = "rgba(21, 128, 61, 0.15)" if active_theme == "light" else "rgba(34, 197, 94, 0.2)"

    if "Total Captures" in combined.columns:
        fig.add_trace(go.Scatter(
            x=combined["datetime"],
            y=combined["Total Captures"],
            name="All Records (incl. Empty)",
            mode="lines",
            line=dict(color=all_line_color, width=1.5, dash="dot"),
            hovertemplate="%{x|%b %Y}: %{y:,} total<extra></extra>"
        ))

    if "Wildlife Captures" in combined.columns:
        fig.add_trace(go.Scatter(
            x=combined["datetime"],
            y=combined["Wildlife Captures"],
            name="Wildlife Captures",
            mode="lines",
            fill="tozeroy",
            fillcolor=fill_color,
            line=dict(color=wildlife_color, width=2.5),
            hovertemplate="%{x|%b %Y}: %{y:,} wildlife<extra></extra>"
        ))

    apply_theme_layout(
        fig,
        title="Wildlife Activity Timeline Across Multi-Year Corridor Monitoring",
        height=380,
        theme=active_theme
    )
    fig.update_layout(
        xaxis_title="Monitoring Period",
        yaxis_title="Monthly Image Captures",
        hovermode="x unified"
    )
    return fig


def create_location_activity_chart(
    df: pd.DataFrame,
    top_n: int = 15,
    theme: Optional[str] = None
) -> go.Figure:
    """Generate bar chart showing the highest activity camera trap locations."""
    active_theme = theme or get_current_theme()
    if "location" not in df.columns:
        return go.Figure()

    loc_counts = df["location"].value_counts().head(top_n).reset_index()
    loc_counts.columns = ["Camera Location ID", "Observations"]
    loc_counts["Camera Location ID"] = "Loc #" + loc_counts["Camera Location ID"].astype(str)

    if active_theme == "light":
        colorscale = [
            [0.0, "#0369a1"],
            [0.5, "#0284c7"],
            [1.0, "#38bdf8"]
        ]
        line_color = "#0284c7"
    else:
        colorscale = [
            [0.0, "#0284c7"],
            [0.5, "#38bdf8"],
            [1.0, "#7dd3fc"]
        ]
        line_color = "#0369a1"

    fig = go.Figure(go.Bar(
        x=loc_counts["Camera Location ID"],
        y=loc_counts["Observations"],
        marker=dict(
            color=loc_counts["Observations"],
            colorscale=colorscale,
            line=dict(color=line_color, width=1)
        ),
        hovertemplate="<b>%{x}</b><br>Recorded Images: %{y:,}<extra></extra>"
    ))

    apply_theme_layout(fig, title=f"Top {top_n} Camera Trap Stations by Recording Volume", height=380, theme=active_theme)
    fig.update_layout(xaxis_title="Camera Location Identifier", yaxis_title="Recorded Images")
    return fig


def create_diurnal_nocturnal_chart(
    df: pd.DataFrame,
    top_n: int = 10,
    theme: Optional[str] = None
) -> go.Figure:
    """Compare Day vs Night observations per animal species."""
    active_theme = theme or get_current_theme()
    if "is_animal" not in df.columns or "time_of_day" not in df.columns:
        return go.Figure()

    animal_df = df[df["is_animal"] & df["time_of_day"].isin(["Day", "Night"])]
    top_species = animal_df["category"].value_counts().head(top_n).index

    grouped = animal_df[animal_df["category"].isin(top_species)].groupby(
        ["category", "time_of_day"]
    ).size().unstack(fill_value=0).reset_index()

    day_color = "#d97706" if active_theme == "light" else "#f59e0b"
    night_color = "#4f46e5" if active_theme == "light" else "#6366f1"

    fig = go.Figure()
    if "Day" in grouped.columns:
        fig.add_trace(go.Bar(
            name="Diurnal (Day: 06-18h)",
            x=grouped["category"],
            y=grouped["Day"],
            marker_color=day_color,
            hovertemplate="Day Observations: %{y:,}<extra></extra>"
        ))
    if "Night" in grouped.columns:
        fig.add_trace(go.Bar(
            name="Nocturnal (Night: 18-06h)",
            x=grouped["category"],
            y=grouped["Night"],
            marker_color=night_color,
            hovertemplate="Night Observations: %{y:,}<extra></extra>"
        ))

    apply_theme_layout(fig, title="Diurnal vs. Nocturnal Activity Distribution by Species", height=400, theme=active_theme)
    fig.update_layout(barmode="group", xaxis_title="Wildlife Species", yaxis_title="Observations")
    return fig


def create_hourly_activity_heatmap(
    df: pd.DataFrame,
    top_locations_n: int = 12,
    theme: Optional[str] = None
) -> go.Figure:
    """Generate heatmap of camera locations vs. hour of day."""
    active_theme = theme or get_current_theme()
    if "hour" not in df.columns or "location" not in df.columns:
        return go.Figure()

    valid_df = df.dropna(subset=["hour"]).copy()
    top_locs = valid_df["location"].value_counts().head(top_locations_n).index

    filtered = valid_df[valid_df["location"].isin(top_locs)]
    cross = pd.crosstab(filtered["location"], filtered["hour"])
    cross.index = ["Loc #" + str(idx) for idx in cross.index]

    if active_theme == "light":
        colorscale = [
            [0.0, "#f0fdf4"],
            [0.2, "#86efac"],
            [0.6, "#16a34a"],
            [1.0, "#d97706"]
        ]
    else:
        colorscale = [
            [0.0, "#09120c"],
            [0.2, "#133e25"],
            [0.6, "#22c55e"],
            [1.0, "#f59e0b"]
        ]

    fig = go.Figure(data=go.Heatmap(
        z=cross.values,
        x=[f"{int(h):02d}:00" for h in cross.columns],
        y=cross.index,
        colorscale=colorscale,
        hoverongaps=False,
        hovertemplate="<b>%{y}</b> at %{x}<br>Activity: %{z:,} images<extra></extra>"
    ))

    apply_theme_layout(fig, title="24-Hour Activity Matrix across Top Camera Stations", height=420, theme=active_theme)
    fig.update_layout(xaxis_title="Hour of Day (24-Hour Cycle)", yaxis_title="Camera Station")
    return fig


def create_sequence_length_histogram(
    df: pd.DataFrame,
    theme: Optional[str] = None
) -> go.Figure:
    """Distribution of camera trap burst lengths (seq_num_frames)."""
    active_theme = theme or get_current_theme()
    if "seq_num_frames" not in df.columns:
        return go.Figure()

    # Drop duplicates by seq_id to get true sequence length distribution
    seq_df = df.drop_duplicates(subset=["seq_id"]) if "seq_id" in df.columns else df
    counts = seq_df["seq_num_frames"].value_counts().head(10).reset_index()
    counts.columns = ["Frames in Sequence", "Sequence Count"]
    counts = counts.sort_values("Frames in Sequence")

    bar_color = "#15803d" if active_theme == "light" else "#10b981"

    fig = go.Figure(go.Bar(
        x=counts["Frames in Sequence"].astype(str) + " Frames",
        y=counts["Sequence Count"],
        marker_color=bar_color,
        hovertemplate="Burst Length: %{x}<br>Total Sequences: %{y:,}<extra></extra>"
    ))

    apply_theme_layout(fig, title="Camera Trap Motion Burst Sequence Distribution", height=350, theme=active_theme)
    fig.update_layout(xaxis_title="Sequence Burst Length", yaxis_title="Number of Unique Sequences")
    return fig


def create_confusion_matrix_chart(
    cm: np.ndarray,
    labels: List[str],
    theme: Optional[str] = None
) -> go.Figure:
    """Plot interactive confusion matrix for ML evaluation."""
    active_theme = theme or get_current_theme()
    # Normalize for hover percentages
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    text_annotations = []
    for i in range(len(labels)):
        row = []
        for j in range(len(labels)):
            val = cm[i, j]
            pct = cm_norm[i, j] * 100
            row.append(f"{val}<br>({pct:.1f}%)")
        text_annotations.append(row)

    if active_theme == "light":
        colorscale = [
            [0.0, "#f0fdf4"],
            [0.5, "#86efac"],
            [1.0, "#15803d"]
        ]
        text_color = "#0c2417"
    else:
        colorscale = [
            [0.0, "#0d1b13"],
            [0.5, "#15803d"],
            [1.0, "#4ade80"]
        ]
        text_color = "white"

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        text=text_annotations,
        texttemplate="%{text}",
        textfont=dict(size=11, color=text_color),
        colorscale=colorscale,
        colorbar=dict(title="Predictions"),
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z:,}<extra></extra>"
    ))

    apply_theme_layout(fig, title="Model Confusion Matrix (Validation Set)", height=500, theme=active_theme)
    fig.update_layout(
        xaxis_title="Predicted Label",
        yaxis_title="True Ground-Truth Label"
    )
    return fig


def create_feature_importance_chart(
    feature_names: List[str],
    importances: np.ndarray,
    top_n: int = 12,
    theme: Optional[str] = None
) -> go.Figure:
    """Plot Random Forest feature importance bar chart."""
    active_theme = theme or get_current_theme()
    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=True).tail(top_n)

    if active_theme == "light":
        colorscale = [
            [0.0, "#d97706"],
            [1.0, "#15803d"]
        ]
    else:
        colorscale = [
            [0.0, "#eab308"],
            [1.0, "#22c55e"]
        ]

    fig = go.Figure(go.Bar(
        x=df_imp["Importance"],
        y=df_imp["Feature"],
        orientation="h",
        marker=dict(
            color=df_imp["Importance"],
            colorscale=colorscale
        ),
        hovertemplate="<b>%{y}</b><br>Importance Weight: %{x:.4f}<extra></extra>"
    ))

    apply_theme_layout(fig, title=f"Top {top_n} Predictive Features (Mean Decrease in Impurity)", height=400, theme=active_theme)
    fig.update_layout(xaxis_title="Relative Feature Importance", yaxis_title="Feature Name")
    return fig


def create_risk_distribution_chart(
    risk_df: pd.DataFrame,
    theme: Optional[str] = None
) -> go.Figure:
    """Generate doughnut chart of stations by Prototype Risk Priority Level."""
    active_theme = theme or get_current_theme()
    if risk_df is None or risk_df.empty or "risk_level" not in risk_df.columns:
        return go.Figure()

    counts = risk_df["risk_level"].value_counts().reset_index()
    counts.columns = ["Risk Tier", "Stations"]

    color_map = {
        "High": "#ef4444" if active_theme == "dark" else "#dc2626",
        "Medium": "#f59e0b" if active_theme == "dark" else "#d97706",
        "Low": "#22c55e" if active_theme == "dark" else "#15803d",
        "Insufficient Data": "#64748b" if active_theme == "dark" else "#94a3b8"
    }

    fig = go.Figure(data=[go.Pie(
        labels=counts["Risk Tier"],
        values=counts["Stations"],
        hole=0.45,
        marker=dict(colors=[color_map.get(k, "#94a3b8") for k in counts["Risk Tier"]]),
        textinfo="label+value",
        hovertemplate="<b>%{label}</b><br>Station Count: %{value} (%{percent})<extra></extra>"
    )])

    apply_theme_layout(fig, title="Camera Station Risk Priority Distribution", height=350, theme=active_theme)
    fig.update_layout(showlegend=True)
    return fig


def create_activity_trend_comparison_chart(
    trends_df: pd.DataFrame,
    theme: Optional[str] = None
) -> go.Figure:
    """Generate multi-line comparison of multi-year activity trends."""
    active_theme = theme or get_current_theme()
    if trends_df is None or trends_df.empty:
        return go.Figure()

    year_cols = [c for c in [2010, 2011, 2012, 2013, 2014, 2015] if c in trends_df.columns]
    if not year_cols:
        return go.Figure()

    palette = [
        "#22c55e", "#38bdf8", "#f59e0b", "#f97316",
        "#a78bfa", "#4ade80", "#ec4899", "#14b8a6"
    ] if active_theme == "dark" else [
        "#15803d", "#0284c7", "#d97706", "#ea580c",
        "#7c3aed", "#16a34a", "#db2777", "#0f766e"
    ]

    fig = go.Figure()
    for idx, (_, row) in enumerate(trends_df.head(8).iterrows()):
        loc_id = row["location_id"]
        y_vals = [row[y] for y in year_cols]
        fig.add_trace(go.Scatter(
            x=[str(y) for y in year_cols],
            y=y_vals,
            mode="lines+markers",
            name=f"Station #{loc_id}",
            line=dict(width=2.5, color=palette[idx % len(palette)]),
            marker=dict(size=6),
            hovertemplate=f"<b>Station #{loc_id}</b><br>Year %{{x}}: %{{y:,}} observations<extra></extra>"
        ))

    apply_theme_layout(fig, title="Multi-Year Observation Trajectories (2010–2015) Across Top Stations", height=380, theme=active_theme)
    fig.update_layout(xaxis_title="Calendar Year", yaxis_title="Annual Captures")
    return fig

