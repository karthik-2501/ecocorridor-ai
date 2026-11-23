"""
Unit tests for centralized theme management, session state handling,
CSS generation, and Plotly theme layout application.
"""

import pytest
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from modules.theme import (
    THEME_CONFIG,
    init_theme_state,
    get_theme_config,
    get_current_theme,
    generate_theme_css,
    apply_plotly_theme,
    THEME_OPTIONS,
    THEME_MAP
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
    create_feature_importance_chart
)


@pytest.fixture(autouse=True)
def clean_streamlit_session():
    """Ensure session state is clean before and after each test."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    yield
    for key in list(st.session_state.keys()):
        del st.session_state[key]


@pytest.fixture
def sample_chart_dataframe():
    """Create representative camera trap dataframe for chart testing."""
    return pd.DataFrame({
        "image_id": ["img1", "img2", "img3", "img4", "img5"],
        "category": ["deer", "deer", "coyote", "empty", "car"],
        "is_animal": [True, True, True, False, False],
        "location": ["1", "1", "2", "2", "3"],
        "datetime": pd.to_datetime([
            "2013-10-01 12:00:00",
            "2013-10-02 23:00:00",
            "2013-11-01 15:00:00",
            "2013-11-02 02:00:00",
            "2013-12-01 08:00:00"
        ]),
        "time_of_day": ["Day", "Night", "Day", "Night", "Day"],
        "hour": [12, 23, 15, 2, 8],
        "seq_id": ["s1", "s2", "s3", "s4", "s5"],
        "seq_num_frames": [1, 2, 1, 3, 1]
    })


def test_theme_config_structure():
    """Verify both dark and light configurations contain all required visual tokens."""
    assert "dark" in THEME_CONFIG
    assert "light" in THEME_CONFIG

    required_keys = [
        "bg_primary", "bg_surface", "bg_card", "bg_sidebar",
        "border_color", "border_accent", "forest_green",
        "text_primary", "text_secondary", "text_muted", "metric_value",
        "hero_bg", "hero_title_gradient", "notice_bg", "notice_text",
        "button_bg", "button_text", "input_bg",
        "plotly_template", "plotly_paper", "plotly_plot", "plotly_title",
        "plotly_text", "plotly_grid"
    ]

    for mode in ["dark", "light"]:
        cfg = THEME_CONFIG[mode]
        for key in required_keys:
            assert key in cfg, f"Missing required token '{key}' in theme '{mode}'"


def test_init_theme_state():
    """Test session state initialization and persistence."""
    assert "theme" not in st.session_state

    # First initialization should default to dark
    theme = init_theme_state("dark")
    assert theme == "dark"
    assert st.session_state["theme"] == "dark"
    assert st.session_state["theme_toggle_radio"] == "🌙 Dark Mode"

    # Subsequent call should preserve existing theme
    st.session_state["theme"] = "light"
    preserved = init_theme_state("dark")
    assert preserved == "light"
    assert st.session_state["theme"] == "light"


def test_generate_theme_css():
    """Verify generated CSS includes correct variables and selectors for both themes."""
    dark_css = generate_theme_css("dark")
    light_css = generate_theme_css("light")

    assert "--bg-primary: #09120c" in dark_css
    assert "--bg-primary: #f5f8f5" in light_css

    assert "[data-testid=\"stSidebar\"]" in dark_css
    assert ".metric-card" in dark_css
    assert ".stButton > button" in dark_css
    assert "[data-testid=\"stMetric\"]" in dark_css
    assert "div[data-baseweb=\"select\"]" in dark_css
    assert ".notice-box" in dark_css


def test_apply_plotly_theme():
    """Test Plotly layout theme updates for dark and light modes."""
    fig = go.Figure(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]))

    # Apply dark theme
    apply_plotly_theme(fig, theme="dark", title="Dark Test Chart", height=350)
    layout = fig.layout
    assert layout.template.layout.paper_bgcolor is not None or layout.paper_bgcolor is not None
    assert layout.height == 350
    assert "Dark Test Chart" in layout.title.text
    assert layout.font.color == THEME_CONFIG["dark"]["plotly_text"]

    # Apply light theme
    apply_plotly_theme(fig, theme="light", title="Light Test Chart", height=420)
    layout = fig.layout
    assert layout.height == 420
    assert "Light Test Chart" in layout.title.text
    assert layout.font.color == THEME_CONFIG["light"]["plotly_text"]
    assert layout.paper_bgcolor == THEME_CONFIG["light"]["plotly_paper"]


def test_all_chart_generators_with_themes(sample_chart_dataframe):
    """Verify that all chart generator functions execute correctly under both dark and light modes."""
    df = sample_chart_dataframe

    for mode in ["dark", "light"]:
        # 1. Species distribution
        fig1 = create_species_distribution_chart(df, top_n=5, animals_only=True, theme=mode)
        assert isinstance(fig1, go.Figure)

        # 2. Activity timeline
        fig2 = create_activity_timeline_chart(df, freq="ME", theme=mode)
        assert isinstance(fig2, go.Figure)

        # 3. Location activity
        fig3 = create_location_activity_chart(df, top_n=5, theme=mode)
        assert isinstance(fig3, go.Figure)

        # 4. Diurnal vs nocturnal
        fig4 = create_diurnal_nocturnal_chart(df, top_n=5, theme=mode)
        assert isinstance(fig4, go.Figure)

        # 5. Hourly heatmap
        fig5 = create_hourly_activity_heatmap(df, top_locations_n=5, theme=mode)
        assert isinstance(fig5, go.Figure)

        # 6. Sequence length histogram
        fig6 = create_sequence_length_histogram(df, theme=mode)
        assert isinstance(fig6, go.Figure)

        # 7. Confusion matrix
        cm = np.array([[5, 1], [0, 4]])
        fig7 = create_confusion_matrix_chart(cm, labels=["ClassA", "ClassB"], theme=mode)
        assert isinstance(fig7, go.Figure)

        # 8. Feature importance
        fig8 = create_feature_importance_chart(["feat1", "feat2"], np.array([0.7, 0.3]), top_n=2, theme=mode)
        assert isinstance(fig8, go.Figure)


def test_app_theme_toggle_flow():
    """Integration test using Streamlit AppTest to verify toggle and persistence across pages."""
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parent.parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=30).run()
    assert not at.exception, f"Initial run failed: {at.exception}"
    assert at.session_state["theme"] == "dark"

    # Switch to Light Mode
    theme_radio = at.sidebar.radio(key="theme_toggle_radio")
    theme_radio.set_value("☀️ Light Mode").run()
    assert not at.exception, f"Switching to light failed: {at.exception}"
    assert at.session_state["theme"] == "light"

    # Test page navigation in light mode
    pages_to_test = [
        "🔍 Dataset Explorer",
        "🐾 Wildlife Species Analysis",
        "📷 Camera Location Monitoring",
        "ℹ️ About the Project"
    ]
    for page in pages_to_test:
        nav_radio = [r for r in at.sidebar.radio if any("Home Dashboard" in opt for opt in r.options)][0]
        nav_radio.set_value(page).run()
        assert not at.exception, f"Page '{page}' failed: {at.exception}"
        assert at.session_state["theme"] == "light", f"Theme lost on page '{page}'"

    # Switch back to Dark Mode
    theme_radio = at.sidebar.radio(key="theme_toggle_radio")
    theme_radio.set_value("🌙 Dark Mode").run()
    assert not at.exception, f"Switching back to dark failed: {at.exception}"
    assert at.session_state["theme"] == "dark"

