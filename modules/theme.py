"""
Centralized Theme Management Module for AI-Powered Wildlife Corridor Monitoring.
Provides centralized theme tokens, dynamic CSS injection, and Plotly theme helpers
for seamless Dark Forest and Light Day mode switching.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import streamlit as st
import plotly.graph_objects as go

import config


# -----------------------------------------------------------------------------
# Centralized Theme Configurations
# -----------------------------------------------------------------------------
THEME_CONFIG: Dict[str, Dict[str, Any]] = {
    "dark": {
        "name": "Dark Forest",
        "icon": "🌙",
        "toggle_label": "🌙 Dark Mode",
        # Backgrounds
        "bg_primary": "#09120c",
        "bg_surface": "#101c14",
        "bg_card": "#15241b",
        "bg_card_hover": "#1b2f23",
        "bg_sidebar": "#0b150f",
        # Borders
        "border_color": "#213c2c",
        "border_accent": "#2e593e",
        "border_subtle": "#1e3526",
        # Accents & Palette
        "forest_green": "#22c55e",
        "emerald_glow": "#10b981",
        "sage": "#86efac",
        "earth_amber": "#f59e0b",
        "terracotta": "#f97316",
        "danger_red": "#ef4444",
        "water_blue": "#38bdf8",
        # Typography
        "text_primary": "#f8fafc",
        "text_secondary": "#94a3b8",
        "text_muted": "#64748b",
        "metric_value": "#ffffff",
        # Gradients & Containers
        "hero_bg": "linear-gradient(135deg, rgba(16, 28, 20, 0.95) 0%, rgba(21, 36, 27, 0.85) 100%)",
        "hero_title_gradient": "linear-gradient(135deg, #ffffff 40%, #a7f3d0 100%)",
        # Notice boxes
        "notice_bg": "rgba(15, 23, 42, 0.65)",
        "notice_forest_bg": "rgba(16, 28, 20, 0.75)",
        "notice_info_bg": "rgba(15, 23, 42, 0.75)",
        "notice_text": "#e2e8f0",
        # Buttons
        "button_bg": "linear-gradient(135deg, #1b3826 0%, #102418 100%)",
        "button_hover_bg": "linear-gradient(135deg, #234d34 0%, #153322 100%)",
        "button_text": "#f8fafc",
        "button_border": "#2e593e",
        # Form inputs
        "input_bg": "#101c14",
        "input_border": "#213c2c",
        "input_text": "#f8fafc",
        # Chips & Placeholders
        "chip_bg": "rgba(255, 255, 255, 0.06)",
        "chip_border": "rgba(255, 255, 255, 0.12)",
        "chip_text": "#e2e8f0",
        "camera_card_bg": "#0f1d15",
        "camera_card_border": "#213c2c",
        # Badges
        "badge_success_bg": "rgba(34, 197, 94, 0.15)",
        "badge_success_text": "#4ade80",
        "badge_success_border": "rgba(34, 197, 94, 0.35)",
        "badge_warning_bg": "rgba(245, 158, 11, 0.15)",
        "badge_warning_text": "#fbbf24",
        "badge_warning_border": "rgba(245, 158, 11, 0.35)",
        "badge_danger_bg": "rgba(239, 68, 68, 0.15)",
        "badge_danger_text": "#f87171",
        "badge_danger_border": "rgba(239, 68, 68, 0.35)",
        "badge_info_bg": "rgba(56, 189, 248, 0.15)",
        "badge_info_text": "#38bdf8",
        "badge_info_border": "rgba(56, 189, 248, 0.35)",
        "badge_neutral_bg": "rgba(148, 163, 184, 0.15)",
        "badge_neutral_text": "#cbd5e1",
        "badge_neutral_border": "rgba(148, 163, 184, 0.3)",
        # Shadows
        "shadow_subtle": "0 4px 20px rgba(0, 0, 0, 0.4)",
        "shadow_glow": "0 0 25px rgba(34, 197, 94, 0.15)",
        # Plotly Tokens
        "plotly_template": "plotly_dark",
        "plotly_paper": "rgba(18, 32, 23, 0.6)",
        "plotly_plot": "rgba(10, 19, 14, 0.7)",
        "plotly_title": "#f8fafc",
        "plotly_text": "#94a3b8",
        "plotly_grid": "rgba(46, 89, 62, 0.25)",
        "plotly_line": "rgba(46, 89, 62, 0.45)",
        "plotly_zeroline": "rgba(46, 89, 62, 0.35)",
        "plotly_legend_bg": "rgba(18, 32, 23, 0.7)",
        "plotly_legend_border": "rgba(46, 89, 62, 0.4)",
        "plotly_legend_font": "#f8fafc",
        "plotly_hover_bg": "#15241b",
        "plotly_hover_border": "#2e593e",
        "plotly_hover_font": "#f8fafc",
    },
    "light": {
        "name": "Light Day",
        "icon": "☀️",
        "toggle_label": "☀️ Light Mode",
        # Backgrounds
        "bg_primary": "#f5f8f5",
        "bg_surface": "#ffffff",
        "bg_card": "#ffffff",
        "bg_card_hover": "#f0fdf4",
        "bg_sidebar": "#eaf2ec",
        # Borders
        "border_color": "#d0ded4",
        "border_accent": "#16a34a",
        "border_subtle": "#e2e8f0",
        # Accents & Palette
        "forest_green": "#15803d",
        "emerald_glow": "#059669",
        "sage": "#166534",
        "earth_amber": "#d97706",
        "terracotta": "#ea580c",
        "danger_red": "#dc2626",
        "water_blue": "#0284c7",
        # Typography
        "text_primary": "#0c2417",
        "text_secondary": "#334155",
        "text_muted": "#64748b",
        "metric_value": "#092316",
        # Gradients & Containers
        "hero_bg": "linear-gradient(135deg, #eaf5ee 0%, #f4faf6 100%)",
        "hero_title_gradient": "linear-gradient(135deg, #092617 30%, #15803d 100%)",
        # Notice boxes
        "notice_bg": "#f8fafc",
        "notice_forest_bg": "#f0fdf4",
        "notice_info_bg": "#f0f9ff",
        "notice_text": "#1e293b",
        # Buttons
        "button_bg": "linear-gradient(135deg, #16a34a 0%, #15803d 100%)",
        "button_hover_bg": "linear-gradient(135deg, #15803d 0%, #166534 100%)",
        "button_text": "#ffffff",
        "button_border": "#15803d",
        # Form inputs
        "input_bg": "#ffffff",
        "input_border": "#cbd5e1",
        "input_text": "#0c2417",
        # Chips & Placeholders
        "chip_bg": "#eef4f0",
        "chip_border": "#cbd5e1",
        "chip_text": "#0f291e",
        "camera_card_bg": "#f0fdf4",
        "camera_card_border": "#bbf7d0",
        # Badges
        "badge_success_bg": "#dcfce7",
        "badge_success_text": "#15803d",
        "badge_success_border": "#86efac",
        "badge_warning_bg": "#fef3c7",
        "badge_warning_text": "#b45309",
        "badge_warning_border": "#fcd34d",
        "badge_danger_bg": "#fee2e2",
        "badge_danger_text": "#b91c1c",
        "badge_danger_border": "#fca5a5",
        "badge_info_bg": "#e0f2fe",
        "badge_info_text": "#0369a1",
        "badge_info_border": "#7dd3fc",
        "badge_neutral_bg": "#f1f5f9",
        "badge_neutral_text": "#475569",
        "badge_neutral_border": "#cbd5e1",
        # Shadows
        "shadow_subtle": "0 2px 12px rgba(0, 0, 0, 0.06)",
        "shadow_glow": "0 0 20px rgba(22, 163, 74, 0.12)",
        # Plotly Tokens
        "plotly_template": "plotly_white",
        "plotly_paper": "#ffffff",
        "plotly_plot": "#f8faf8",
        "plotly_title": "#0c2417",
        "plotly_text": "#334155",
        "plotly_grid": "rgba(203, 213, 225, 0.6)",
        "plotly_line": "rgba(100, 116, 139, 0.4)",
        "plotly_zeroline": "rgba(100, 116, 139, 0.3)",
        "plotly_legend_bg": "rgba(255, 255, 255, 0.95)",
        "plotly_legend_border": "rgba(208, 222, 212, 0.9)",
        "plotly_legend_font": "#0c2417",
        "plotly_hover_bg": "#ffffff",
        "plotly_hover_border": "#15803d",
        "plotly_hover_font": "#0c2417",
    }
}

THEME_OPTIONS = ["🌙 Dark Mode", "☀️ Light Mode"]
THEME_MAP = {
    "🌙 Dark Mode": "dark",
    "☀️ Light Mode": "light"
}
REVERSE_THEME_MAP = {
    "dark": "🌙 Dark Mode",
    "light": "☀️ Light Mode"
}


def get_theme_config(theme: Optional[str] = None) -> Dict[str, Any]:
    """Retrieve theme token dictionary for a specific theme or active session theme."""
    if theme is None:
        theme = get_current_theme()
    return THEME_CONFIG.get(theme, THEME_CONFIG["dark"])


def get_current_theme() -> str:
    """Return the currently selected theme from session state, defaulting to 'dark'."""
    try:
        if "theme" in st.session_state:
            val = st.session_state["theme"]
            if val in ("dark", "light"):
                return val
    except Exception:
        pass
    return "dark"


def init_theme_state(default_theme: str = "dark") -> str:
    """
    Initialize theme in session state only once, preserving existing choice.
    Returns the active theme string ('dark' or 'light').
    """
    if "theme" not in st.session_state:
        st.session_state["theme"] = default_theme

    theme = st.session_state["theme"]
    if theme not in ("dark", "light"):
        theme = default_theme
        st.session_state["theme"] = theme

    # Synchronize toggle widget state if not present
    expected_choice = REVERSE_THEME_MAP.get(theme, "🌙 Dark Mode")
    if "theme_toggle_radio" not in st.session_state:
        st.session_state["theme_toggle_radio"] = expected_choice

    return theme


def on_theme_change():
    """Callback triggered whenever user interacts with the theme toggle radio."""
    choice = st.session_state.get("theme_toggle_radio")
    if choice in THEME_MAP:
        st.session_state["theme"] = THEME_MAP[choice]


def render_theme_toggle(sidebar: bool = True) -> str:
    """
    Render working theme toggle in the sidebar with clear icons:
    🌙 Dark Mode and ☀️ Light Mode.
    Preserves user choice across reruns and immediately updates the application.
    """
    target = st.sidebar if sidebar else st
    current = get_current_theme()
    current_idx = 0 if current == "dark" else 1

    expected_choice = REVERSE_THEME_MAP.get(current, "🌙 Dark Mode")
    if st.session_state.get("theme_toggle_radio") != expected_choice:
        st.session_state["theme_toggle_radio"] = expected_choice

    target.markdown(
        f"""
        <div style='display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px; padding: 0 2px;'>
            <span style='font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: var(--sage);'>
                Theme Mode
            </span>
            <span style='font-size: 0.74rem; color: var(--text-muted); font-weight: 500;'>
                {'Dark Forest' if current == 'dark' else 'Light Day'}
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    selected = target.radio(
        "Visual Theme",
        options=THEME_OPTIONS,
        index=current_idx,
        key="theme_toggle_radio",
        on_change=on_theme_change,
        horizontal=True,
        label_visibility="collapsed"
    )

    # In case on_change was bypassed, ensure sync and immediate update
    selected_theme = THEME_MAP.get(selected, "dark")
    if selected_theme != st.session_state.get("theme"):
        st.session_state["theme"] = selected_theme
        st.rerun()

    return st.session_state["theme"]


def generate_theme_css(theme: Optional[str] = None) -> str:
    """Generate dynamic CSS variable definitions and overrides for the active theme."""
    cfg = get_theme_config(theme)
    is_dark = (theme or get_current_theme()) == "dark"

    css = f"""
    /* Theme Variables Root Injection */
    :root {{
        --bg-primary: {cfg['bg_primary']};
        --bg-surface: {cfg['bg_surface']};
        --bg-card: {cfg['bg_card']};
        --bg-card-hover: {cfg['bg_card_hover']};
        --bg-sidebar: {cfg['bg_sidebar']};
        --border-color: {cfg['border_color']};
        --border-accent: {cfg['border_accent']};
        --border-subtle: {cfg['border_subtle']};
        --forest-green: {cfg['forest_green']};
        --emerald-glow: {cfg['emerald_glow']};
        --sage: {cfg['sage']};
        --earth-amber: {cfg['earth_amber']};
        --terracotta: {cfg['terracotta']};
        --danger-red: {cfg['danger_red']};
        --water-blue: {cfg['water_blue']};
        --text-primary: {cfg['text_primary']};
        --text-secondary: {cfg['text_secondary']};
        --text-muted: {cfg['text_muted']};
        --metric-val-color: {cfg['metric_value']};
        --hero-bg: {cfg['hero_bg']};
        --hero-title-gradient: {cfg['hero_title_gradient']};
        --notice-bg: {cfg['notice_bg']};
        --notice-forest-bg: {cfg['notice_forest_bg']};
        --notice-info-bg: {cfg['notice_info_bg']};
        --notice-text: {cfg['notice_text']};
        --button-bg: {cfg['button_bg']};
        --button-hover-bg: {cfg['button_hover_bg']};
        --button-text: {cfg['button_text']};
        --button-border: {cfg['button_border']};
        --input-bg: {cfg['input_bg']};
        --input-border: {cfg['input_border']};
        --input-text: {cfg['input_text']};
        --chip-bg: {cfg['chip_bg']};
        --chip-border: {cfg['chip_border']};
        --chip-text: {cfg['chip_text']};
        --camera-card-bg: {cfg['camera_card_bg']};
        --camera-card-border: {cfg['camera_card_border']};
        --badge-success-bg: {cfg['badge_success_bg']};
        --badge-success-text: {cfg['badge_success_text']};
        --badge-success-border: {cfg['badge_success_border']};
        --badge-warning-bg: {cfg['badge_warning_bg']};
        --badge-warning-text: {cfg['badge_warning_text']};
        --badge-warning-border: {cfg['badge_warning_border']};
        --badge-danger-bg: {cfg['badge_danger_bg']};
        --badge-danger-text: {cfg['badge_danger_text']};
        --badge-danger-border: {cfg['badge_danger_border']};
        --badge-info-bg: {cfg['badge_info_bg']};
        --badge-info-text: {cfg['badge_info_text']};
        --badge-info-border: {cfg['badge_info_border']};
        --badge-neutral-bg: {cfg['badge_neutral_bg']};
        --badge-neutral-text: {cfg['badge_neutral_text']};
        --badge-neutral-border: {cfg['badge_neutral_border']};
        --shadow-subtle: {cfg['shadow_subtle']};
        --shadow-glow: {cfg['shadow_glow']};
    }}

    /* Streamlit App Container Overrides */
    .stApp {{
        background-color: var(--bg-primary) !important;
        color: var(--text-primary) !important;
    }}

    /* Global Typography */
    html, body, [class*="css"], .stMarkdown, .stText {{
        color: var(--text-primary) !important;
    }}

    h1, h2, h3, h4, h5, h6 {{
        color: var(--text-primary) !important;
    }}

    p, span, label {{
        color: var(--text-primary);
    }}

    [data-testid="stCaptionContainer"] p, .stCaption {{
        color: var(--text-muted) !important;
    }}

    /* Custom Hero Container */
    .hero-container {{
        background: var(--hero-bg) !important;
        border: 1px solid var(--border-color) !important;
        box-shadow: var(--shadow-subtle), var(--shadow-glow) !important;
    }}

    .hero-title {{
        background: var(--hero-title-gradient) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
    }}

    .hero-subtitle {{
        color: var(--text-secondary) !important;
    }}

    .hero-meta-row {{
        border-top: 1px solid var(--border-color) !important;
        color: var(--text-muted) !important;
    }}

    .hero-meta-item strong {{
        color: var(--text-primary) !important;
    }}

    /* Custom Metric Cards */
    .metric-card {{
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        box-shadow: var(--shadow-subtle) !important;
    }}

    .metric-card:hover {{
        background: var(--bg-card-hover) !important;
        border-color: var(--border-accent) !important;
    }}

    .metric-label {{
        color: var(--text-secondary) !important;
    }}

    .metric-value {{
        color: var(--metric-val-color) !important;
    }}

    .metric-delta {{
        color: var(--text-muted) !important;
    }}

    /* Native Streamlit Metric Cards */
    [data-testid="stMetric"] {{
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
        box-shadow: var(--shadow-subtle) !important;
    }}

    [data-testid="stMetricLabel"] p {{
        color: var(--text-secondary) !important;
        font-weight: 600 !important;
    }}

    [data-testid="stMetricValue"] div {{
        color: var(--metric-val-color) !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
    }}

    [data-testid="stMetricDelta"] svg {{
        fill: currentColor !important;
    }}

    /* Sidebar Overrides */
    [data-testid="stSidebar"], section[data-testid="stSidebar"] {{
        background-color: var(--bg-sidebar) !important;
        border-right: 1px solid var(--border-color) !important;
    }}

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {{
        color: var(--sage) !important;
    }}

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {{
        color: var(--text-primary) !important;
    }}

    /* Section Header & Content Boxes */
    .section-header {{
        border-bottom: 1px solid var(--border-color) !important;
    }}

    .section-title {{
        color: var(--text-primary) !important;
    }}

    .section-subtitle {{
        color: var(--text-secondary) !important;
    }}

    .content-box {{
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-primary) !important;
    }}

    /* Notice & Disclaimer Boxes */
    .notice-box {{
        background: var(--notice-bg) !important;
        color: var(--notice-text) !important;
        border-left: 4px solid var(--earth-amber) !important;
    }}

    .notice-box.forest {{
        background: var(--notice-forest-bg) !important;
        border-left-color: var(--forest-green) !important;
    }}

    .notice-box.info {{
        background: var(--notice-info-bg) !important;
        border-left-color: var(--water-blue) !important;
    }}

    /* Buttons */
    .stButton > button,
    .stDownloadButton > button {{
        background: var(--button-bg) !important;
        color: var(--button-text) !important;
        border: 1px solid var(--button-border) !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.12) !important;
    }}

    .stButton > button:hover,
    .stDownloadButton > button:hover {{
        background: var(--button-hover-bg) !important;
        border-color: var(--forest-green) !important;
        box-shadow: 0 4px 14px rgba(22, 163, 74, 0.25) !important;
    }}

    /* Form Controls & Inputs */
    div[data-baseweb="select"] > div {{
        background-color: var(--input-bg) !important;
        border-color: var(--border-color) !important;
        color: var(--input-text) !important;
    }}

    div[data-baseweb="select"] span {{
        color: var(--input-text) !important;
    }}

    div[data-baseweb="popover"],
    div[data-baseweb="menu"],
    ul[data-testid="stSelectboxVirtualDropdown"] {{
        background-color: var(--bg-surface) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-primary) !important;
    }}

    li[data-baseweb="menu-item"] {{
        color: var(--text-primary) !important;
        background-color: transparent !important;
    }}

    li[data-baseweb="menu-item"]:hover {{
        background-color: var(--bg-card-hover) !important;
    }}

    input[type="text"],
    input[type="number"],
    .stTextInput input,
    .stNumberInput input {{
        background-color: var(--input-bg) !important;
        color: var(--input-text) !important;
        border-color: var(--border-color) !important;
    }}

    div[data-baseweb="tag"] {{
        background-color: var(--chip-bg) !important;
        border: 1px solid var(--chip-border) !important;
    }}

    div[data-baseweb="tag"] span {{
        color: var(--chip-text) !important;
    }}

    div[data-testid="stRadio"] label,
    div[data-testid="stCheckbox"] label {{
        color: var(--text-primary) !important;
    }}

    /* Tabs */
    button[data-baseweb="tab"] {{
        color: var(--text-secondary) !important;
    }}

    button[data-baseweb="tab"][aria-selected="true"] {{
        color: var(--forest-green) !important;
        font-weight: 700 !important;
    }}

    div[data-baseweb="tab-highlight"] {{
        background-color: var(--forest-green) !important;
    }}

    div[data-baseweb="tab-border"] {{
        background-color: var(--border-color) !important;
    }}

    /* Expanders */
    [data-testid="stExpander"] {{
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
    }}

    [data-testid="stExpander"] summary {{
        color: var(--text-primary) !important;
    }}

    [data-testid="stExpander"] summary:hover {{
        color: var(--forest-green) !important;
    }}

    /* Dataframe & Tables */
    [data-testid="stDataFrame"] {{
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        background-color: var(--bg-card) !important;
    }}

    /* Badges */
    .badge-success {{
        background: var(--badge-success-bg) !important;
        color: var(--badge-success-text) !important;
        border: 1px solid var(--badge-success-border) !important;
    }}

    .badge-warning {{
        background: var(--badge-warning-bg) !important;
        color: var(--badge-warning-text) !important;
        border: 1px solid var(--badge-warning-border) !important;
    }}

    .badge-danger {{
        background: var(--badge-danger-bg) !important;
        color: var(--badge-danger-text) !important;
        border: 1px solid var(--badge-danger-border) !important;
    }}

    .badge-info {{
        background: var(--badge-info-bg) !important;
        color: var(--badge-info-text) !important;
        border: 1px solid var(--badge-info-border) !important;
    }}

    .badge-neutral {{
        background: var(--badge-neutral-bg) !important;
        color: var(--badge-neutral-text) !important;
        border: 1px solid var(--badge-neutral-border) !important;
    }}

    /* Chips & Placeholders */
    .id-chip {{
        background: var(--chip-bg) !important;
        border: 1px solid var(--chip-border) !important;
        color: var(--chip-text) !important;
    }}

    .camera-placeholder-card {{
        background: var(--camera-card-bg) !important;
        border: 2px dashed var(--camera-card-border) !important;
        color: var(--text-secondary) !important;
    }}

    /* Dividers */
    hr {{
        border-color: var(--border-color) !important;
    }}

    /* Theme-aware Sidebar Branding & Footer Classes */
    .sidebar-header-box {{
        padding: 10px 0 16px 0;
        border-bottom: 1px solid var(--border-color);
    }}

    .sidebar-title {{
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 1.15rem;
        color: var(--text-primary);
        line-height: 1.2;
    }}

    .sidebar-sub {{
        font-size: 0.76rem;
        color: var(--sage);
        font-weight: 600;
    }}

    .sidebar-academic {{
        font-size: 0.72rem;
        color: var(--text-muted);
        margin-top: 3px;
    }}

    .footer-text {{
        text-align: center;
        font-size: 0.8rem;
        color: var(--text-muted);
        padding: 10px 0 20px 0;
    }}
    """
    return css


def apply_theme(theme: Optional[str] = None) -> None:
    """
    Inject base stylesheet and active theme-specific dynamic CSS into Streamlit page.
    """
    active = theme or get_current_theme()

    # Load base stylesheet if present
    css_path = config.ASSETS_DIR / "style.css"
    base_css = ""
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            base_css = f.read()

    theme_css = generate_theme_css(active)

    combined_css = f"<style>\n{base_css}\n{theme_css}\n</style>"
    st.markdown(combined_css, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Plotly Chart Theme Helpers
# -----------------------------------------------------------------------------
def apply_plotly_theme(
    fig: go.Figure,
    theme: Optional[str] = None,
    title: str = "",
    height: int = 400
) -> go.Figure:
    """
    Apply theme layout, background, grid, legend, and typography to any Plotly figure.
    Dynamically respects dark forest vs. light day mode.
    """
    cfg = get_theme_config(theme)

    fig.update_layout(
        template=cfg["plotly_template"],
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(family="Outfit, sans-serif", size=16, color=cfg["plotly_title"]),
            x=0.02,
            y=0.96
        ),
        paper_bgcolor=cfg["plotly_paper"],
        plot_bgcolor=cfg["plotly_plot"],
        font=dict(family="Inter, sans-serif", color=cfg["plotly_text"], size=12),
        margin=dict(l=40, r=30, t=50, b=40),
        height=height,
        hoverlabel=dict(
            bgcolor=cfg["plotly_hover_bg"],
            font_size=12,
            font_family="Inter, sans-serif",
            font_color=cfg["plotly_hover_font"],
            bordercolor=cfg["plotly_hover_border"]
        ),
        xaxis=dict(
            gridcolor=cfg["plotly_grid"],
            linecolor=cfg["plotly_line"],
            zerolinecolor=cfg["plotly_zeroline"],
            tickfont=dict(color=cfg["plotly_text"]),
            title_font=dict(color=cfg["plotly_title"])
        ),
        yaxis=dict(
            gridcolor=cfg["plotly_grid"],
            linecolor=cfg["plotly_line"],
            zerolinecolor=cfg["plotly_zeroline"],
            tickfont=dict(color=cfg["plotly_text"]),
            title_font=dict(color=cfg["plotly_title"])
        ),
        legend=dict(
            font=dict(color=cfg["plotly_legend_font"]),
            bgcolor=cfg["plotly_legend_bg"],
            bordercolor=cfg["plotly_legend_border"],
            borderwidth=1
        )
    )
    return fig
