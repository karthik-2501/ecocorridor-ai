"""
Unit tests for rule-based risk analysis and prototype alerts.
"""

import pandas as pd
import pytest

from modules.preprocessing import prepare_full_pipeline
from modules.risk_analysis import calculate_station_risk_scores, generate_prototype_alerts


@pytest.fixture
def sample_risk_dataframe():
    """Create a mock multi-station dataframe with known risk properties."""
    records = []
    # Station 1: Normal activity, all deer, daytime
    for i in range(10):
        records.append({
            "image_id": f"s1_{i}",
            "file_name": f"s1_{i}.jpg",
            "date_captured": "2013-10-04 12:00:00",
            "location": "1",
            "seq_id": f"seq_s1_{i}",
            "frame_num": 1,
            "seq_num_frames": 1,
            "width": 2048,
            "height": 1494,
            "category": "deer",
            "labels": "deer",
            "num_annotations": 1,
            "rights_holder": "USGS"
        })
    # Station 2: High vehicular disturbance, 20 cars, night
    for i in range(20):
        records.append({
            "image_id": f"s2_{i}",
            "file_name": f"s2_{i}.jpg",
            "date_captured": "2013-10-04 23:00:00",
            "location": "2",
            "seq_id": f"seq_s2_{i}",
            "frame_num": 1,
            "seq_num_frames": 1,
            "width": 2048,
            "height": 1494,
            "category": "car",
            "labels": "car",
            "num_annotations": 1,
            "rights_holder": "USGS"
        })
    df = pd.DataFrame(records)
    return prepare_full_pipeline(df)


def test_calculate_station_risk_scores(sample_risk_dataframe):
    risk_df = calculate_station_risk_scores(sample_risk_dataframe)
    
    assert len(risk_df) == 2
    assert "prototype_risk_score" in risk_df.columns
    assert "risk_level" in risk_df.columns
    assert "rationale" in risk_df.columns

    # Check bounds
    assert (risk_df["prototype_risk_score"] >= 0).all()
    assert (risk_df["prototype_risk_score"] <= 100).all()

    # Station 2 has 100% cars and 100% night, so it should have a higher risk score than Station 1
    score_s1 = risk_df.loc[risk_df["location_id"] == "1", "prototype_risk_score"].iloc[0]
    score_s2 = risk_df.loc[risk_df["location_id"] == "2", "prototype_risk_score"].iloc[0]
    assert score_s2 > score_s1


def test_generate_prototype_alerts(sample_risk_dataframe):
    risk_df = calculate_station_risk_scores(sample_risk_dataframe)
    alerts = generate_prototype_alerts(risk_df, sample_risk_dataframe)
    
    assert isinstance(alerts, pd.DataFrame)
    assert not alerts.empty
    assert "severity" in alerts.columns
    assert "alert_type" in alerts.columns
    
    # Station 2 should trigger a vehicle alert
    assert any("Vehicle" in t for t in alerts["alert_type"])


def test_insufficient_data_handling():
    # Station with only 2 records (< 5)
    records = [
        {
            "image_id": "sparse_1",
            "file_name": "sparse_1.jpg",
            "date_captured": "2013-10-04 12:00:00",
            "location": "999",
            "seq_id": "seq_sparse_1",
            "frame_num": 1,
            "seq_num_frames": 1,
            "width": 2048,
            "height": 1494,
            "category": "deer",
            "labels": "deer",
            "num_annotations": 1,
            "rights_holder": "USGS"
        },
        {
            "image_id": "sparse_2",
            "file_name": "sparse_2.jpg",
            "date_captured": "2013-10-04 12:01:00",
            "location": "999",
            "seq_id": "seq_sparse_2",
            "frame_num": 1,
            "seq_num_frames": 1,
            "width": 2048,
            "height": 1494,
            "category": "deer",
            "labels": "deer",
            "num_annotations": 1,
            "rights_holder": "USGS"
        }
    ]
    df = prepare_full_pipeline(pd.DataFrame(records))
    risk_df = calculate_station_risk_scores(df)

    assert len(risk_df) == 1
    assert risk_df.iloc[0]["risk_level"] == "Insufficient Data"
    assert "Sparse" in risk_df.iloc[0]["rationale"]


def test_generate_consolidated_report(sample_risk_dataframe):
    from modules.risk_analysis import generate_consolidated_corridor_report
    risk_df = calculate_station_risk_scores(sample_risk_dataframe)
    report = generate_consolidated_corridor_report(risk_df)

    assert isinstance(report, pd.DataFrame)
    assert not report.empty
    assert "Camera_Station_ID" in report.columns
    assert "Risk_Priority_Tier" in report.columns
    assert "badge_class" not in report.columns

