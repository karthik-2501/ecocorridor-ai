"""
Wildlife Activity and Explainable Prototype Risk Analysis Module.
Calculates transparent, rule-based Activity Risk Scores for camera stations
based on statistical baseline deviations, vehicle presence, nocturnal concentration,
and apex predator presence.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

import config


def calculate_station_risk_scores(
    df: pd.DataFrame,
    activity_weight: float = 0.35,
    vehicle_weight: float = 0.30,
    nocturnal_weight: float = 0.20,
    predator_weight: float = 0.15,
    high_threshold: float = 70.0,
    medium_threshold: float = 40.0,
    min_records_threshold: int = 5
) -> pd.DataFrame:
    """
    Calculate an explainable, rule-based Prototype Activity Risk Score
    for each camera location in the dataset.

    Formulas:
    1. Activity Intensity Subscore (0-100):
       Ratio of station capture count to network mean baseline (capped at 2.5x baseline).
    2. Anthropogenic Disturbance Subscore (0-100):
       Ratio of vehicle ('car') records to total records at the location.
    3. Nocturnal Concentration Subscore (0-100):
       Percentage of wildlife records captured during night hours (18:00 to 06:00).
    4. Predator Presence Subscore (0-100):
       Frequency of conflict-prone predator records (mountain_lion, bobcat, coyote, fox).
    5. Composite Prototype Score:
       Score = (w_act * Activity) + (w_veh * Disturbance) + (w_noc * Nocturnal) + (w_pred * Predator)

    Data Sufficiency Check:
    Stations with fewer than `min_records_threshold` records are marked as
    'Insufficient Data' rather than given a misleading risk prediction.

    Scientific Note: This score is an explainable analytical index indicating statistical
    deviation and potential anthropogenic conflict, NOT verified poaching, deforestation,
    or collision probability.
    """
    if df is None or df.empty or "location" not in df.columns:
        return pd.DataFrame()

    network_mean_activity = len(df) / df["location"].nunique()
    results = []

    for loc_id, group in df.groupby("location"):
        loc_total = len(group)
        vehicle_count = int((group["category"] == "car").sum()) if "category" in group.columns else 0
        empty_count = int((group["category"] == "empty").sum()) if "category" in group.columns else 0
        wildlife_count = loc_total - vehicle_count - empty_count

        # Count predator observations
        predator_count = 0
        if "category" in group.columns:
            predator_count = int(group["category"].isin(config.PREDATOR_SPECIES).sum())

        # Check for insufficient sample size
        if loc_total < min_records_threshold:
            results.append({
                "location_id": str(loc_id),
                "prototype_risk_score": 0.0,
                "risk_level": "Insufficient Data",
                "badge_class": "badge-neutral",
                "total_observations": loc_total,
                "wildlife_observations": wildlife_count,
                "vehicle_observations": vehicle_count,
                "predator_observations": predator_count,
                "nocturnal_percentage": 0.0,
                "activity_subscore": 0.0,
                "vehicle_subscore": 0.0,
                "nocturnal_subscore": 0.0,
                "predator_subscore": 0.0,
                "rationale": f"Sparse observation record ({loc_total} total captures); insufficient evidence for risk assessment",
                "recommended_action": "Deploy supplementary camera sensors or conduct habitat survey to gather baseline data."
            })
            continue

        # 1. Activity Intensity Factor (capped at 100)
        activity_ratio = loc_total / network_mean_activity if network_mean_activity > 0 else 1.0
        score_activity = min(100.0, (activity_ratio / 2.5) * 100.0)

        # 2. Anthropogenic / Vehicle Disturbance Factor (0-100)
        vehicle_ratio = vehicle_count / loc_total if loc_total > 0 else 0.0
        score_vehicle = min(100.0, vehicle_ratio * 400.0)  # 25% cars gives max 100

        # 3. Nocturnal Concentration Factor (0-100)
        nocturnal_count = 0
        if "time_of_day" in group.columns:
            nocturnal_count = int((group["time_of_day"] == "Night").sum())
        nocturnal_ratio = nocturnal_count / loc_total if loc_total > 0 else 0.0
        score_nocturnal = nocturnal_ratio * 100.0

        # 4. Predator Presence Factor (0-100)
        predator_ratio = predator_count / loc_total if loc_total > 0 else 0.0
        score_predator = min(100.0, predator_ratio * 300.0)

        # Composite score calculation
        total_weight = activity_weight + vehicle_weight + nocturnal_weight + predator_weight
        if total_weight <= 0:
            total_weight = 1.0

        w_act = activity_weight / total_weight
        w_veh = vehicle_weight / total_weight
        w_noc = nocturnal_weight / total_weight
        w_pred = predator_weight / total_weight

        composite_score = (
            (w_act * score_activity) +
            (w_veh * score_vehicle) +
            (w_noc * score_nocturnal) +
            (w_pred * score_predator)
        )
        composite_score = round(min(100.0, max(0.0, composite_score)), 1)

        # Risk Classification
        if composite_score >= high_threshold:
            category = "High"
            badge_class = "badge-danger"
            recommendation = "Prioritize station for field inspection; evaluate vehicular corridor fencing and trail encroachment."
        elif composite_score >= medium_threshold:
            category = "Medium"
            badge_class = "badge-warning"
            recommendation = "Maintain regular surveillance; monitor seasonal shifts in nocturnal and vehicular corridor usage."
        else:
            category = "Low"
            badge_class = "badge-success"
            recommendation = "Stable baseline activity; continue standard periodic camera maintenance."

        # Rationale builder
        reasons = []
        if activity_ratio > 1.5:
            reasons.append(f"High activity volume ({activity_ratio:.1f}x network baseline)")
        if vehicle_count > 0:
            reasons.append(f"Recorded {vehicle_count:,} vehicle encounters ({vehicle_ratio*100:.1f}%)")
        if nocturnal_ratio > 0.60:
            reasons.append(f"Heavy nocturnal concentration ({nocturnal_ratio*100:.1f}%)")
        if predator_count > 0:
            reasons.append(f"Observed {predator_count} apex predator encounters ({predator_ratio*100:.1f}%)")
        if not reasons:
            reasons.append("Stable activity consistent with network baseline")

        results.append({
            "location_id": str(loc_id),
            "prototype_risk_score": composite_score,
            "risk_level": category,
            "badge_class": badge_class,
            "total_observations": loc_total,
            "wildlife_observations": wildlife_count,
            "vehicle_observations": vehicle_count,
            "predator_observations": predator_count,
            "nocturnal_percentage": round(nocturnal_ratio * 100, 1),
            "activity_subscore": round(score_activity, 1),
            "vehicle_subscore": round(score_vehicle, 1),
            "nocturnal_subscore": round(score_nocturnal, 1),
            "predator_subscore": round(score_predator, 1),
            "rationale": "; ".join(reasons),
            "recommended_action": recommendation
        })

    risk_df = pd.DataFrame(results)
    risk_df = risk_df.sort_values("prototype_risk_score", ascending=False).reset_index(drop=True)
    return risk_df


def generate_prototype_alerts(
    risk_df: pd.DataFrame,
    df_raw: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate evidence-based prototype alert feed for conservation management.
    Alerts include:
    - Affected camera location
    - Most recent observation date / active date range from actual dataset
    - Alert reason with empirical statistics
    - Recommended conservation action
    - Transparent classification as a statistical prototype alert
    """
    alerts = []

    # Map latest timestamp and date range for each station
    date_lookup = {}
    if df_raw is not None and not df_raw.empty and "location" in df_raw.columns:
        if "datetime" in df_raw.columns:
            valid_dt_df = df_raw.dropna(subset=["datetime"])
            for loc_id, group in valid_dt_df.groupby("location"):
                latest_dt = group["datetime"].max()
                earliest_dt = group["datetime"].min()
                date_lookup[str(loc_id)] = {
                    "latest_date": str(latest_dt.date()) if pd.notna(latest_dt) else "2015-06-03",
                    "date_span": f"{earliest_dt.year}–{latest_dt.year}" if pd.notna(earliest_dt) and pd.notna(latest_dt) else "2010–2015"
                }

    for _, row in risk_df.iterrows():
        loc_id = row["location_id"]
        loc_dates = date_lookup.get(loc_id, {"latest_date": "Historical Record", "date_span": "2010–2015"})
        obs_date = loc_dates["latest_date"]

        # 1. Vehicle intrusion alert
        if row["vehicle_observations"] > 50:
            alerts.append({
                "alert_id": f"ALT-VEH-{loc_id}",
                "location_id": f"Loc #{loc_id}",
                "date_observed": obs_date,
                "severity": "High",
                "alert_type": "Anthropogenic Intrusion",
                "reason": (
                    f"Station recorded {row['vehicle_observations']:,} vehicle detections "
                    f"({row['vehicle_observations']/row['total_observations']*100:.1f}% of station total). "
                    f"Significant human-wildlife corridor overlap."
                ),
                "recommended_action": "Assess vehicle trail encroachment, inspect physical corridor fencing, and coordinate with rangers.",
                "alert_classification": "Prototype Demonstration Alert (Statistical Anomaly)"
            })
        elif row["vehicle_observations"] > 0:
            alerts.append({
                "alert_id": f"ALT-VEH-M-{loc_id}",
                "location_id": f"Loc #{loc_id}",
                "date_observed": obs_date,
                "severity": "Medium",
                "alert_type": "Vehicle Presence",
                "reason": f"{row['vehicle_observations']} vehicle instances observed along trail corridor.",
                "recommended_action": "Monitor seasonal vehicular trail usage and place wildlife crossing awareness signs.",
                "alert_classification": "Prototype Demonstration Alert (Statistical Anomaly)"
            })

        # 2. Activity spike alert
        if row["activity_subscore"] >= 80.0:
            alerts.append({
                "alert_id": f"ALT-ACT-{loc_id}",
                "location_id": f"Loc #{loc_id}",
                "date_observed": obs_date,
                "severity": "Medium" if row["vehicle_observations"] == 0 else "High",
                "alert_type": "Volume Anomaly",
                "reason": (
                    f"Activity volume ({row['total_observations']:,} captures) significantly exceeds corridor baseline "
                    f"(subscore: {row['activity_subscore']}/100)."
                ),
                "recommended_action": "Verify sensor false-trigger rate, clear wind-blown vegetation, and inspect SD card capacity.",
                "alert_classification": "Prototype Demonstration Alert (Statistical Anomaly)"
            })

        # 3. Apex Predator conflict indicator alert
        if row.get("predator_observations", 0) >= 5 and row["vehicle_observations"] > 0:
            alerts.append({
                "alert_id": f"ALT-PRED-{loc_id}",
                "location_id": f"Loc #{loc_id}",
                "date_observed": obs_date,
                "severity": "High",
                "alert_type": "Predator Co-occurrence",
                "reason": (
                    f"Recorded {row['predator_observations']} apex predator captures alongside "
                    f"{row['vehicle_observations']} vehicle records at this station."
                ),
                "recommended_action": "Conduct immediate trail review to evaluate human-carnivore conflict risk along public access paths.",
                "alert_classification": "Prototype Demonstration Alert (Statistical Anomaly)"
            })

        # 4. Coverage gap warning
        if row["risk_level"] == "Insufficient Data":
            alerts.append({
                "alert_id": f"ALT-GAP-{loc_id}",
                "location_id": f"Loc #{loc_id}",
                "date_observed": obs_date,
                "severity": "Low",
                "alert_type": "Sparse Coverage Warning",
                "reason": f"Only {row['total_observations']} record(s) logged at this location. Insufficient data for reliable assessment.",
                "recommended_action": "Check camera battery, mount orientation, and redeploy sensor to improve corridor coverage.",
                "alert_classification": "Prototype Demonstration Alert (Operational Notice)"
            })

    alerts_df = pd.DataFrame(alerts)
    if alerts_df.empty:
        alerts_df = pd.DataFrame(columns=[
            "alert_id", "location_id", "date_observed", "severity",
            "alert_type", "reason", "recommended_action", "alert_classification"
        ])

    return alerts_df


def generate_consolidated_corridor_report(
    risk_df: pd.DataFrame,
    df_raw: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Generate a comprehensive, downloadable corridor assessment report
    integrating station-level metrics, prototype risk scores, subscores,
    rationales, and recommended conservation actions.
    """
    if risk_df is None or risk_df.empty:
        return pd.DataFrame()

    report = risk_df.copy()
    report = report.rename(columns={
        "location_id": "Camera_Station_ID",
        "prototype_risk_score": "Prototype_Risk_Score_0_100",
        "risk_level": "Risk_Priority_Tier",
        "total_observations": "Total_Recorded_Captures",
        "wildlife_observations": "Wildlife_Captures",
        "vehicle_observations": "Vehicle_Captures",
        "predator_observations": "Predator_Captures",
        "nocturnal_percentage": "Nocturnal_Activity_Pct",
        "activity_subscore": "Activity_Volume_Subscore",
        "vehicle_subscore": "Vehicle_Disturbance_Subscore",
        "nocturnal_subscore": "Nocturnal_Concentration_Subscore",
        "predator_subscore": "Predator_Presence_Subscore",
        "rationale": "Score_Diagnostic_Rationale",
        "recommended_action": "Conservation_Recommendation"
    })

    # Drop internal styling columns
    if "badge_class" in report.columns:
        report = report.drop(columns=["badge_class"])

    return report
