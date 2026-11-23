"""
Preprocessing and Feature Engineering Module.
Performs data sanitization, label categorization, temporal feature extraction,
and scikit-learn pipeline preparation for machine learning.
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import config


def parse_and_validate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitize category and label columns.
    Ensures empty, whitespace, or malformed entries are standardized.
    Distinguishes verified animal species from background or human vehicles.
    """
    df = df.copy()

    if "category" in df.columns:
        df["category"] = df["category"].fillna("unknown").astype(str).str.strip().str.lower()
    else:
        df["category"] = "unknown"

    if "labels" in df.columns:
        df["labels"] = df["labels"].fillna("unknown").astype(str).str.strip().str.lower()
    else:
        df["labels"] = df["category"]

    # Flag verified animal vs non-animal
    df["is_animal"] = ~df["category"].isin(config.NON_ANIMAL_CATEGORIES)
    df["is_vehicle"] = df["category"] == "car"
    df["is_empty"] = df["category"] == "empty"

    return df


def engineer_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract diurnal/nocturnal indicators, seasonal indices, and hourly cycles
    from valid datetime values.
    """
    df = df.copy()

    if "datetime" not in df.columns and "date_captured" in df.columns:
        df["datetime"] = pd.to_datetime(df["date_captured"], errors="coerce")

    if "datetime" in df.columns:
        dt = df["datetime"]
        df["hour"] = dt.dt.hour
        df["month"] = dt.dt.month
        df["year"] = dt.dt.year
        df["day_of_week"] = dt.dt.dayofweek
        df["day_name"] = dt.dt.day_name()
        df["date"] = dt.dt.date

        # Diurnal (Day: 06:00-18:00) vs Nocturnal (Night: 18:00-06:00)
        df["is_night"] = df["hour"].apply(
            lambda h: 1 if pd.notna(h) and (h < 6 or h >= 18) else (0 if pd.notna(h) else np.nan)
        )
        df["time_of_day"] = df["is_night"].apply(
            lambda n: "Night" if n == 1 else ("Day" if n == 0 else "Unknown")
        )

        # Meteorological seasons (Northern Hemisphere)
        def get_season(month):
            if pd.isna(month):
                return "Unknown"
            m = int(month)
            if m in [12, 1, 2]:
                return "Winter"
            elif m in [3, 4, 5]:
                return "Spring"
            elif m in [6, 7, 8]:
                return "Summer"
            else:
                return "Autumn"

        df["season"] = df["month"].apply(get_season)

        # Cyclical temporal encoding for ML (sin/cos of hour and month)
        valid_hour = df["hour"].fillna(12)
        df["hour_sin"] = np.sin(2 * np.pi * valid_hour / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * valid_hour / 24.0)

        valid_month = df["month"].fillna(6)
        df["month_sin"] = np.sin(2 * np.pi * valid_month / 12.0)
        df["month_cos"] = np.cos(2 * np.pi * valid_month / 12.0)

    return df


def engineer_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute sequence-level metadata metrics such as frame progress ratio,
    burst position, and sequence frequency.
    """
    df = df.copy()

    if "frame_num" in df.columns and "seq_num_frames" in df.columns:
        # Avoid division by zero
        seq_len = df["seq_num_frames"].replace(0, 1)
        df["frame_ratio"] = (df["frame_num"] / seq_len).clip(0.0, 1.0)
        df["is_single_frame_seq"] = (df["seq_num_frames"] == 1).astype(int)
    else:
        df["frame_ratio"] = 1.0
        df["is_single_frame_seq"] = 1

    # Image aspect ratio
    if "width" in df.columns and "height" in df.columns:
        height_safe = df["height"].replace(0, 1)
        df["aspect_ratio"] = (df["width"] / height_safe).round(3)

    return df


def prepare_full_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run full sequential preprocessing and feature engineering on dataset."""
    df = parse_and_validate_labels(df)
    df = engineer_temporal_features(df)
    df = engineer_sequence_features(df)
    return df


def build_ml_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str]
) -> ColumnTransformer:
    """
    Create a scikit-learn ColumnTransformer pipeline for machine learning.
    Imputes missing numeric features with median and scales them.
    Imputes missing categorical features with mode and one-hot encodes them.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )

    return preprocessor
