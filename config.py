"""
Configuration file for AI-Powered Wildlife Corridor Monitoring and Protection.
Defines paths, category groupings, styling constants, and default parameters.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"
DEFAULT_CSV_NAME = "caltech_camera_traps_image_level(2).csv"
FALLBACK_CSV_NAME = "caltech_camera_traps_image_level.csv"

# Potential dataset locations in priority order
DATASET_CANDIDATE_PATHS = [
    DATA_DIR / DEFAULT_CSV_NAME,
    DATA_DIR / FALLBACK_CSV_NAME,
    Path(r"E:\jassem\caltech_camera_traps_image_level.csv"),
    Path(r"E:\jassem\wildlife_corridor_ai\wildlife_corridor_ai\data\caltech_camera_traps_image_level.csv"),
]

# Project Metadata
PROJECT_TITLE = "AI-Powered Wildlife Corridor Monitoring and Protection"
PROJECT_SUBTITLE = "AI-powered insights for wildlife activity monitoring, corridor awareness, and conservation support."
PROJECT_VERSION = "2.0.0"
PROJECT_ACADEMIC_YEAR = "2025-2026"
PROJECT_TAGLINE = "B.Tech Final-Year Capstone Project"

# Category Classifications
NON_ANIMAL_CATEGORIES = {"empty", "car", "multiple"}

VALID_ANIMAL_SPECIES = [
    "deer", "opossum", "coyote", "raccoon", "rabbit",
    "bobcat", "cat", "bird", "squirrel", "rodent",
    "dog", "cow", "fox", "skunk", "lizard",
    "mountain_lion", "badger", "pig", "insect", "bat"
]

PREDATOR_SPECIES = {"mountain_lion", "bobcat", "coyote", "fox", "badger"}
PREY_HERBIVORE_SPECIES = {"deer", "rabbit", "squirrel", "rodent"}
HUMAN_DOMESTIC_CATEGORIES = {"car", "dog", "cat", "cow", "pig"}

# Expected CSV Columns
EXPECTED_COLUMNS = [
    "image_id", "file_name", "date_captured", "location",
    "seq_id", "frame_num", "seq_num_frames", "width",
    "height", "category", "labels", "num_annotations", "rights_holder"
]

# Theme-Aware Color Palettes (Forest & Earth Tones for Plotly & UI)
PALETTES = {
    "dark": {
        "bg_dark": "#0a130e",
        "card_dark": "#122017",
        "card_border": "#1e3526",
        "forest_green": "#22c55e",
        "emerald": "#10b981",
        "sage": "#86efac",
        "earth_amber": "#f59e0b",
        "terracotta": "#f97316",
        "alert_red": "#ef4444",
        "water_blue": "#38bdf8",
        "muted_gray": "#94a3b8",
        "text_light": "#f8fafc",
        "text_muted": "#94a3b8",
    },
    "light": {
        "bg_dark": "#f5f8f5",
        "card_dark": "#ffffff",
        "card_border": "#d0ded4",
        "forest_green": "#15803d",
        "emerald": "#059669",
        "sage": "#166534",
        "earth_amber": "#d97706",
        "terracotta": "#ea580c",
        "alert_red": "#dc2626",
        "water_blue": "#0284c7",
        "muted_gray": "#64748b",
        "text_light": "#0c2417",
        "text_muted": "#64748b",
    }
}

# Default Palette (Backwards compatibility)
PALETTE = PALETTES["dark"]


# Species Color Map for Visual Consistency across Charts
SPECIES_COLORS = {
    "deer": "#34d399",
    "coyote": "#f59e0b",
    "bobcat": "#fb923c",
    "mountain_lion": "#ef4444",
    "raccoon": "#38bdf8",
    "opossum": "#a78bfa",
    "fox": "#eab308",
    "rabbit": "#4ade80",
    "bird": "#67e8f9",
    "squirrel": "#ca8a04",
    "rodent": "#9ca3af",
    "badger": "#f472b6",
    "skunk": "#94a3b8",
    "cat": "#fbbf24",
    "dog": "#a3e635",
    "cow": "#818cf8",
    "pig": "#f43f5e",
    "lizard": "#2dd4bf",
    "bat": "#64748b",
    "insect": "#a1a1aa",
    "car": "#f43f5e",
    "empty": "#475569",
    "multiple": "#c084fc",
}
