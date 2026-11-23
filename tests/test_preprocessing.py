"""
Unit tests for data preprocessing and feature engineering pipelines.
"""

import pandas as pd
import numpy as np
import pytest

from modules.preprocessing import (
    parse_and_validate_labels,
    engineer_temporal_features,
    engineer_sequence_features,
    prepare_full_pipeline,
    build_ml_preprocessor
)


@pytest.fixture
def sample_raw_dataframe():
    """Create a representative mock camera-trap dataframe."""
    return pd.DataFrame({
        "image_id": ["img_01", "img_02", "img_03", "img_04"],
        "file_name": ["img_01.jpg", "img_02.jpg", "img_03.jpg", "img_04.jpg"],
        "date_captured": [
            "2013-10-04 13:31:53",  # Day
            "2013-11-04 22:37:07",  # Night
            "11 11",                # Malformed date
            "2012-05-09 07:33:45"   # Day
        ],
        "location": ["26", "26", "38", "38"],
        "seq_id": ["seq_a", "seq_b", "seq_c", "seq_c"],
        "frame_num": [1, 1, 1, 2],
        "seq_num_frames": [1, 1, 2, 2],
        "width": [2048, 2048, 1920, 1920],
        "height": [1494, 1494, 1080, 1080],
        "category": ["empty", "deer", "car", "coyote"],
        "labels": ["empty", "deer", "car", "coyote"],
        "num_annotations": [1, 1, 1, 1],
        "rights_holder": ["USGS", "Erin", "Justin", "Justin"]
    })


def test_parse_and_validate_labels(sample_raw_dataframe):
    processed = parse_and_validate_labels(sample_raw_dataframe)
    
    # Check is_animal flag
    assert processed.loc[processed["category"] == "deer", "is_animal"].iloc[0] == True
    assert processed.loc[processed["category"] == "coyote", "is_animal"].iloc[0] == True
    assert processed.loc[processed["category"] == "empty", "is_animal"].iloc[0] == False
    assert processed.loc[processed["category"] == "car", "is_animal"].iloc[0] == False
    
    # Check vehicle and empty flags
    assert processed.loc[processed["category"] == "car", "is_vehicle"].iloc[0] == True
    assert processed.loc[processed["category"] == "empty", "is_empty"].iloc[0] == True


def test_engineer_temporal_features(sample_raw_dataframe):
    processed = engineer_temporal_features(sample_raw_dataframe)
    
    assert "hour" in processed.columns
    assert "time_of_day" in processed.columns
    assert "hour_sin" in processed.columns
    assert "month_cos" in processed.columns
    
    # First row is 13:31:53 -> Day
    assert processed.loc[0, "time_of_day"] == "Day"
    # Second row is 22:37:07 -> Night
    assert processed.loc[1, "time_of_day"] == "Night"
    # Third row is malformed '11 11' -> NaT, hour is NaN
    assert pd.isna(processed.loc[2, "datetime"])


def test_engineer_sequence_features(sample_raw_dataframe):
    processed = engineer_sequence_features(sample_raw_dataframe)
    
    assert "frame_ratio" in processed.columns
    assert "aspect_ratio" in processed.columns
    
    # Single frame sequence ratio should be 1.0
    assert processed.loc[0, "frame_ratio"] == 1.0
    # Frame 1 of 2 should be 0.5
    assert processed.loc[2, "frame_ratio"] == 0.5
    # Frame 2 of 2 should be 1.0
    assert processed.loc[3, "frame_ratio"] == 1.0


def test_build_ml_preprocessor(sample_raw_dataframe):
    df_piped = prepare_full_pipeline(sample_raw_dataframe)
    num_cols = ["hour", "month", "frame_num", "seq_num_frames"]
    cat_cols = ["location"]
    
    preprocessor = build_ml_preprocessor(num_cols, cat_cols)
    transformed = preprocessor.fit_transform(df_piped[num_cols + cat_cols])
    
    assert transformed.shape[0] == len(sample_raw_dataframe)
    assert transformed.shape[1] > len(num_cols) # due to one-hot encoding
